import os
import numpy as np
from torch import nn
import random
from tqdm import tqdm
import logging
import time


import torch
from torch.utils.data import DataLoader

from utils.ad_dataset_2nd import TestDataset, TrainDataset
from utils.sampler import ApproximateGreedyCoresetSampler
from models.feature_extractor import RGBBackbone, XYZBackbone
from models.point_feature_alignment import PointFeatureAlignment
from models.common import FaissNNRGB, FaissNN3D, NearestNeighbourScorerRGB, NearestNeighbourScorer3D, RescaleSegmentor
from models.memory_bank import MMUpdateRGB, MMUpdate3D
# lora
import loralib as lora

from sklearn.metrics import roc_auc_score # AUROC-Score
from utils.metrics import calculate_au_pro # AUPRO-Score

import scipy.ndimage as ndimage
import warnings
warnings.filterwarnings("ignore")


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.deterministic = True


def replace_learnable(
    model,
    target_block_idxs: list[int],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    mode: str = "lora",
    modal: str = "rgb",
    init_lora_weights: bool = True
):
    block_list = list(model.blocks.children())
    for block_idx, backbone_block in enumerate(block_list):
        if block_idx not in target_block_idxs:
            continue
        if mode == "lora":
            for full_name, module in list(backbone_block.named_modules()):
                if full_name.split('.')[0] != 'attn':
                    continue
                if not isinstance(module, nn.Linear):
                    continue
                path = full_name.split('.')
                parent = backbone_block
                for key in path[:-1]:
                    parent = getattr(parent, key)
                attr_name = path[-1]
                orig: nn.Linear = getattr(parent, attr_name)
                if orig.out_features == orig.in_features * 3:
                    lora_layer = lora.MergedLinear(
                        in_features=orig.in_features,
                        out_features=orig.out_features,
                        r=r,
                        lora_alpha=alpha,
                        lora_dropout=dropout,
                        enable_lora=[True, False, True],
                        bias=(orig.bias is not None)
                    )
                else:
                    continue
                
                with torch.no_grad():
                    lora_layer.weight.copy_(orig.weight)
                    if orig.bias is not None:
                        lora_layer.bias.copy_(orig.bias)
                lora_layer.weight.requires_grad = False
                if orig.bias is not None:
                    lora_layer.bias.requires_grad = False
                if init_lora_weights:
                    lora_layer.reset_parameters()
                setattr(parent, attr_name, lora_layer)
            
        elif mode == "fulllora":
            for full_name, module in list(backbone_block.named_modules()):
                if not isinstance(module, nn.Linear):
                    continue
                path = full_name.split('.')
                parent = backbone_block
                for key in path[:-1]:
                    parent = getattr(parent, key)
                attr_name = path[-1]
                orig: nn.Linear = getattr(parent, attr_name)
                if orig.out_features == orig.in_features * 3:
                    lora_layer = lora.MergedLinear(
                        in_features=orig.in_features,
                        out_features=orig.out_features,
                        r=r,
                        lora_alpha=alpha,
                        lora_dropout=dropout,
                        enable_lora=[True, False, True],
                        bias=(orig.bias is not None)
                    )
                else:
                    lora_layer = lora.Linear(
                        in_features=orig.in_features,
                        out_features=orig.out_features,
                        r=r,
                        lora_alpha=alpha,
                        lora_dropout=dropout,
                        bias=(orig.bias is not None)
                    )
                with torch.no_grad():
                    lora_layer.weight.copy_(orig.weight)
                    if orig.bias is not None:
                        lora_layer.bias.copy_(orig.bias)
                lora_layer.weight.requires_grad = False
                if orig.bias is not None:
                    lora_layer.bias.requires_grad = False
                if init_lora_weights:
                    lora_layer.reset_parameters()
                setattr(parent, attr_name, lora_layer)
        else:
            # Just simply unfreeze
            if modal == "rgb":
                for p in backbone_block.parameters():
                    p.requires_grad=True
            if modal == "xyz":
                for p in backbone_block.parameters():
                    p.requires_grad=True
    
    return model


def run(args):
    """Execute the original full-VIC algorithm for one configured class."""
    seed_everything(seed=args.seed)

    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id 
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)


    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path {args.dataset_path} does not exist.")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Set the route
    dataroute = args.dataset_path
    camera_route = os.path.join(args.camera_param_route, f"{args.class_name}/calibration/camera_parameters.json")

    # *** Point Feature Alignment (PFA) steps ***
    # 1. Feature Extraction (Pretrained DINO, Pretrained Point-MAE: like other papers)
    # 2. Point Feature Alignment
    visual_extractor_ckpt = args.visual_encoder_ckpt_pth
    point_extractor_ckpt = args.point_encoder_ckpt_pth
    rgb_context = RGBBackbone(device=device,
                              rgb_checkpoint_path=visual_extractor_ckpt)
    for p in rgb_context.parameters():
        p.requires_grad = False
    """
    returns as follows: (Context: 1 ~ 8, Target: 1 ~ 4)
    1. x,
    2. center, 
    3. ori_idx, 
    4. center_idx, 
    5. target_blocks, 
    6. target_indices -> compare with context + predictor
    7. context_tokens -> for predictor
    8. context_centers -> context index centers
    """
    xyz_context = XYZBackbone(device=device,
                              xyz_checkpoint_path=point_extractor_ckpt,
                              group_size=args.group_size,
                              num_group=args.num_group)
    # Freeze all context blocks
    for p in xyz_context.parameters():
        p.requires_grad = False
    
    if args.mode == "combined" or args.mode == "only_jepa":
        # Replace with LoRA Linear Layers
        rgb_context.rgb_backbone = replace_learnable( 
            rgb_context.rgb_backbone,
            target_block_idxs=[3,7,11],  # reinit 4th, 8th, last layer
            r=2,
            alpha=8,
            dropout=0.1,
            init_lora_weights=True,
            modal="rgb",
            mode="full"
        )
        
        xyz_context.xyz_backbone.blocks = replace_learnable( 
            xyz_context.xyz_backbone.blocks,
            target_block_idxs=[3,7,11],  # reinit 4th, 8th, last layer
            r=2,
            alpha=8,
            dropout=0.1,
            init_lora_weights=True,
            modal="xyz",
            mode="full"
        )
    rgb_params = torch.load(f"{args.stage1_dir}/{args.class_name}_best_jepa_rgb_dino_main_{args.npred}blocks_full_{args.vic_rate}.pth", map_location="cpu")
    xyz_params = torch.load(f"{args.stage1_dir}/{args.class_name}_best_jepa_xyz_pmae_main_{args.npred}blocks_full_{args.vic_rate}.pth", map_location="cpu")
    
    rgb_context.load_state_dict(rgb_params)
    xyz_context.load_state_dict(xyz_params)

    rgb_context = rgb_context.to(device)
    xyz_context = xyz_context.to(device)

    for n, p in rgb_context.named_parameters():
        p.requires_grad = False
    for n, p in xyz_context.named_parameters():
        p.requires_grad = False
        
    # Point Feature Alignment Module
    pfa = PointFeatureAlignment(patch_size=args.patch_size)

    # Initialize the dataset
    # need to align Point Clouds and RGB images (Point Feature Alignment)
    trainset = TrainDataset(args.class_name, args.img_size, dataroute, dtd_dataset_path=args.dtd_dataset_path, k_shot=args.k_shot)
    testset = TestDataset(args.class_name, args.img_size, dataroute)
    # dataloader: train -> customized_collate_fn: also returns context and target blocks

    trainloader = DataLoader(dataset=trainset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             worker_init_fn=worker_init_fn,
                             drop_last=False,
                             pin_memory=True)
    
    testloader = DataLoader(dataset=testset, 
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            worker_init_fn=worker_init_fn,
                            drop_last=False,
                            pin_memory=True)

    def build_coreset_sampler(embeddings, sampler):
        # _, _, d = embeddings.shape
        # embeddings = embeddings.view(-1, d)
        memory_bank, idxs = sampler.run(embeddings)
        return memory_bank, idxs

    # RGB Memory Bank
    nearest_search_rgb_l2 = FaissNNRGB(True, 4, mode="dist")
    anomaly_scorer_rgb = NearestNeighbourScorerRGB(n_nearest_neighbours=5,
                                                   nn_method_l2=nearest_search_rgb_l2)
    # 3D Memory Bank
    nearest_search_xyz_l2 = FaissNN3D(True, 4, mode="dist")
    anomaly_scorer_xyz = NearestNeighbourScorer3D(n_nearest_neighbours=5,
                                                  nn_method_l2=nearest_search_xyz_l2)
    
    # RGB Memory Bank
    nearest_search_rgb_l2_org = FaissNNRGB(True, 4, mode="dist")
    anomaly_scorer_rgb_org = NearestNeighbourScorerRGB(n_nearest_neighbours=5,
                                                       nn_method_l2=nearest_search_rgb_l2_org)
    # 3D Memory Bank
    nearest_search_xyz_l2_org = FaissNN3D(True, 4, mode="dist")
    anomaly_scorer_xyz_org = NearestNeighbourScorer3D(n_nearest_neighbours=5,
                                                      nn_method_l2=nearest_search_xyz_l2_org)
    
    # Anomaly Map Rescaler
    anomaly_segmentor = RescaleSegmentor(
        device=torch.device("cpu"), 
        target_size=args.img_size
    )
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Build memory bank
    anomaly_masks = []
    rgb_original, rgb_anomaly = [], []
    xyz_original, xyz_anomaly = [], []
    for iteration, data in tqdm(enumerate(trainloader)):
        # load data (keep this)
        rgb = data["data"][0] # [bsz, 3, 224, 224]
        xyz = data["data"][1] # [bsz, 3, 224, 224] -> need to reshpae into [bsz, 3, 224*224]
        depth = data["data"][2] # [bsz, 3, 224, 224] # depth image shape (Repeat [1, 224, 224])
        label = data["label"]

        # Anomaly RGB + Depth Texture (Pair 1)
        # Anomaly 3D + Depth Geometric (Pair 2)
        anomaly_rgb = data["anomaly_data"][0]
        anomaly_3d = data["anomaly_data"][1]
        anomaly_depth1 = data["anomaly_data"][2]
        anomaly_depth2 = data["anomaly_data"][3]
        anomaly_mask = data["anomaly_data"][5]

        # append pseudo anomaly mask
        anomaly_masks.append(anomaly_mask)
        # RGB Images
        rgb_original.append(rgb)
        rgb_anomaly.append(anomaly_rgb)
        # 3D Point Clouds
        xyz_original.append(xyz)
        xyz_anomaly.append(anomaly_3d)
            
    # For Coreset Sampling
    with torch.no_grad():
        rgb_context.eval()
        xyz_context.eval()
        rgb_only_embs, rgb_anomaly_embs, xyz_only_embs, xyz_anomaly_embs = [], [], [], []
        rgb_original = torch.cat(rgb_original, dim=0)
        xyz_original = torch.cat(xyz_original, dim=0)
        rgb_anomaly = torch.cat(rgb_anomaly, dim=0)
        xyz_anomaly = torch.cat(xyz_anomaly, dim=0)
        bg_indices_normal, bg_indices_abnormal = [], []
        bsz = rgb_original.shape[0]
        for i in tqdm(range(bsz), desc="Unimodal Feature Extraction Step ..."):
            rgb = rgb_original[i,:,:,:].unsqueeze(0).to(device)
            xyz = xyz_original[i,:,:,:].unsqueeze(0).to(device)

            # Anomaly 
            anomaly_rgb = rgb_anomaly[i,:,:,:].unsqueeze(0).to(device)
            anomaly_xyz = xyz_anomaly[i,:,:,:].unsqueeze(0).to(device)

            # Augmented data
            # Original Points
            B,C,H,W = xyz.shape
            xyz = xyz.view(B, C, H*W)
            B,C,H,W = anomaly_xyz.shape
            anomaly_xyz = anomaly_xyz.view(B, C, H*W)

            # Extract valid points only (Normal)
            xyz_t = xyz.permute(0, 2, 1) # [1, H*W, 3]
            non_zero_mask = torch.all(xyz_t[0] != 0, dim=1) 
            nonzero_indices = torch.nonzero(non_zero_mask, as_tuple=True)[0]
            original_points_valid = xyz_t[:, nonzero_indices, :]

            # Extract valid points only (Anomaly)
            anomaly_xyz_t = anomaly_xyz.permute(0, 2, 1)
            non_zero_mask_anom = torch.all(anomaly_xyz_t[0] != 0, dim=1)
            nonzero_indices_anom = torch.nonzero(non_zero_mask_anom, as_tuple=True)[0]
            original_points_anom_valid = anomaly_xyz_t[:, nonzero_indices_anom, :]

            # for coreset sampler: put -> context parameters
            h_rgb_coreset = rgb_context(rgb, bg_indices=None, context_mask=None)
            h_rgb_anomaly = rgb_context(anomaly_rgb, bg_indices=None, context_mask=None)
            # # for memory bank
            rgb_only_embs.append(h_rgb_coreset.detach().cpu())
            rgb_anomaly_embs.append(h_rgb_anomaly.detach().cpu())
            # 3D Modality
            xyz_features, center, \
                ori_idx, center_idx = xyz_context(xyz, mode="target")
            # Point-Feature Alignment
            xyz_features1 = pfa(group_features=xyz_features.permute(0,2,1), 
                                group_centers=center,         
                                original_points=original_points_valid, 
                                nonzero_indices=nonzero_indices,       
                                kernel_size=args.patch_size//2)
            B,D,P,P = xyz_features1.shape
            xyz_features1 = xyz_features1.permute(0,2,3,1).reshape(B, P*P, D)

            xyz_only_embs.append(xyz_features1.detach().cpu())

            xyz_features2 = pfa(group_features=xyz_features.permute(0,2,1), 
                                group_centers=center,         
                                original_points=original_points_valid,
                                nonzero_indices=nonzero_indices,
                                kernel_size=args.patch_size)
            B,D,P,P = xyz_features2.shape
            xyz_features2 = xyz_features2.permute(0,2,3,1).reshape(B, P*P, D)
            bg_idx = (xyz_features2.sum(axis = -1)  == 0)
            bg_indices_normal.append(bg_idx.detach().cpu())

            xyz_features_anomaly, center_anomaly, \
                ori_idx_anomaly, center_idx_anomaly = xyz_context(anomaly_xyz, mode="target")
            # Point-Feature Alignment
            xyz_features_anomaly_ = pfa(group_features=xyz_features_anomaly.permute(0,2,1), 
                                        group_centers=center_anomaly,         
                                        original_points=original_points_anom_valid,
                                        nonzero_indices=nonzero_indices_anom,
                                        kernel_size=args.patch_size//2)
            B,D,P,P = xyz_features_anomaly_.shape
            xyz_features_anomaly_ = xyz_features_anomaly_.permute(0,2,3,1).reshape(B, P*P, D)
            xyz_anomaly_embs.append(xyz_features_anomaly_.detach().cpu())

            xyz_features_ab = pfa(group_features=xyz_features_anomaly.permute(0,2,1), 
                                    group_centers=center_anomaly,         
                                    original_points=original_points_anom_valid,
                                    nonzero_indices=nonzero_indices_anom,
                                    kernel_size=args.patch_size)
            B,D,P,P = xyz_features_ab.shape
            xyz_features_ab = xyz_features_ab.permute(0,2,3,1).reshape(B, P*P, D)
            bg_idx = (xyz_features_ab.sum(axis = -1)  == 0)
            bg_indices_abnormal.append(bg_idx.detach().cpu())
        
        rgb_only_embs = torch.cat(rgb_only_embs, dim=0)
        xyz_only_embs = torch.cat(xyz_only_embs, dim=0)

        rgb_anomaly_embs = torch.cat(rgb_anomaly_embs, dim=0)
        xyz_anomaly_embs = torch.cat(xyz_anomaly_embs, dim=0)
        
        anomaly_masks = torch.cat(anomaly_masks, dim=0)

        bg_indices_normal = torch.cat(bg_indices_normal, dim=0)
        bg_indices_abnormal = torch.cat(bg_indices_abnormal, dim=0)

    # RGB Coreset Sampling
    b, p_rgb, d_rgb = rgb_only_embs.shape
    rgb_only_embs_bank = rgb_only_embs.reshape(b*p_rgb, d_rgb)
    sampler_rgb = ApproximateGreedyCoresetSampler(percentage=args.coreset_percentage_rgb,
                                                  device=device)
    coreset_samples_rgb_only, _ = build_coreset_sampler(embeddings=rgb_only_embs_bank, sampler=sampler_rgb)
    coreset_samples_rgb_only = coreset_samples_rgb_only.detach().cpu().numpy().astype("float32")
    
    # XYZ Coreset Sampling
    b, p_xyz, d_xyz = xyz_only_embs.shape
    xyz_only_embs_bank = xyz_only_embs.reshape(b*p_xyz, d_xyz)
    sampler_xyz = ApproximateGreedyCoresetSampler(percentage=args.coreset_percentage_xyz,
                                                    device=device)
    coreset_samples_xyz_only, _ = build_coreset_sampler(embeddings=xyz_only_embs_bank, sampler=sampler_xyz)
    coreset_samples_xyz_only = coreset_samples_xyz_only.detach().cpu().numpy().astype("float32")

    # Without Noise Suppression Style
    anomaly_scorer_rgb_org.fit(detection_features=[coreset_samples_rgb_only])
    anomaly_scorer_xyz_org.fit(detection_features=[coreset_samples_xyz_only])

    # Seed setting
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)
    
    # Update Memory Bank
    rgb_bank = MMUpdateRGB(init_memory_bank=torch.from_numpy(coreset_samples_rgb_only),
                           embed_dim=2304*2)
    xyz_bank = MMUpdate3D(init_memory_bank=torch.from_numpy(coreset_samples_xyz_only),
                          embed_dim=1152*2)
    
    n = int(args.coreset_percentage_rgb * 100)
    rgb_bank_params = torch.load(f"{args.stage2_dir}/{args.class_name}_rgb_bank_main_{args.npred}blocks_full_{args.vic_rate}.pth", map_location="cpu")
    xyz_bank_params = torch.load(f"{args.stage2_dir}/{args.class_name}_xyz_bank_main_{args.npred}blocks_full_{args.vic_rate}.pth", map_location="cpu")

    rgb_bank.load_state_dict(rgb_bank_params)
    xyz_bank.load_state_dict(xyz_bank_params)

    rgb_bank = rgb_bank.to(device)
    xyz_bank = xyz_bank.to(device)
    
    # Fix memory bank
    coreset_samples_rgb_only = rgb_bank.memory_bank.detach().cpu().numpy()
    coreset_samples_xyz_only = xyz_bank.memory_bank.detach().cpu().numpy()
    
    anomaly_scorer_rgb.fit(detection_features=[coreset_samples_rgb_only])
    anomaly_scorer_xyz.fit(detection_features=[coreset_samples_xyz_only])
    

    @torch.no_grad()
    def eval_anomaly(rgb_ftrs,
                     xyz_ftrs,
                    #  mm_ftrs,
                     l):
        print(f"label is as: {l}")
        bsz, npatches_rgb, _ = rgb_ftrs.shape
        rgb_ftrs = rgb_ftrs.reshape(bsz*npatches_rgb, -1).detach().cpu()
        bsz, npatches_xyz, _ = xyz_ftrs.shape
        xyz_ftrs = xyz_ftrs.reshape(bsz*npatches_xyz, -1).detach().cpu()

        # 1. Memory Bank Update Style
        confidence_rgb, _ = rgb_bank(rgb_ftrs.to(device))
        confidence_xyz, _ = xyz_bank(xyz_ftrs.to(device))
        # Best Scenario: logit -> exponential 
        # Range of Confidence: [-1, 1]
        # Effect: noise suppression
        confidence_rgb = torch.exp(confidence_rgb.reshape(-1)).detach().cpu().reshape(224, 224).numpy()
        confidence_xyz = torch.exp(confidence_xyz.reshape(-1)).detach().cpu().reshape(224, 224).numpy()
        _, dscore_rgb_only, _, _ = anomaly_scorer_rgb.predict([rgb_ftrs.detach().cpu().contiguous().numpy()], max_val=None, foreground_mask=None)
        _, dscore_xyz_only, _, _ = anomaly_scorer_xyz.predict([xyz_ftrs.detach().cpu().contiguous().numpy()], max_val=None, foreground_mask=None)

        # 1.1. make Segmentation Maps
        dscore_rgb_only = torch.tensor(dscore_rgb_only).unsqueeze(0).reshape(-1, args.img_size//args.patch_size, args.img_size//args.patch_size)
        anomaly_map_rgb = anomaly_segmentor.convert_to_segmentation(dscore_rgb_only)[0]
        anomaly_map_rgb = ndimage.gaussian_filter(anomaly_map_rgb, sigma=4)
        anomaly_map_rgb_combined = anomaly_map_rgb * confidence_rgb
        # RGB Only Detection Score
        anomaly_rgb_only_mem_update = np.max(anomaly_map_rgb_combined.reshape(1, -1), axis=-1) 

        dscore_xyz_only = torch.tensor(dscore_xyz_only).unsqueeze(0).reshape(-1, args.img_size//(args.patch_size//2), args.img_size//(args.patch_size//2))
        anomaly_map_xyz = anomaly_segmentor.convert_to_segmentation(dscore_xyz_only)[0]
        anomaly_map_xyz = ndimage.gaussian_filter(anomaly_map_xyz, sigma=4)
        anomaly_map_xyz_combined = anomaly_map_xyz * confidence_xyz
        # 3D Only Detection Score
        anomaly_xyz_only_mem_update = np.max(anomaly_map_xyz_combined.reshape(1, -1), axis=-1)
        
        # Fusion Map Score (Z-score Normalization + Maximum Fusion)
        anomaly_rgb = anomaly_map_rgb_combined.reshape(-1)
        anomaly_rgb = (anomaly_rgb - np.mean(anomaly_rgb, axis=-1)) / (np.std(anomaly_rgb, axis=-1) + 1e-8)
        anomaly_xyz = anomaly_map_xyz_combined.reshape(-1)
        anomaly_xyz = (anomaly_xyz - np.mean(anomaly_xyz, axis=-1)) / (np.std(anomaly_xyz, axis=-1) + 1e-8)
        # Just RGB + 3D 
        anomaly_map_fusion_2_banks = np.maximum(anomaly_rgb, anomaly_xyz).reshape(224, 224)
        anomaly_fusion_2_banks = anomaly_rgb_only_mem_update * anomaly_xyz_only_mem_update * np.max(anomaly_map_fusion_2_banks.reshape(1, -1), axis=-1)

        # 2. Just JEPA Only
        _, dscore_rgb_only_org, _, _ = anomaly_scorer_rgb_org.predict([rgb_ftrs.detach().cpu().contiguous().numpy()], max_val=None, foreground_mask=None)
        _, dscore_xyz_only_org, _, _ = anomaly_scorer_xyz_org.predict([xyz_ftrs.detach().cpu().contiguous().numpy()], max_val=None, foreground_mask=None)

        # 2.1. make Segmentation Maps
        dscore_rgb_only_org = torch.tensor(dscore_rgb_only_org).unsqueeze(0).reshape(-1, args.img_size//args.patch_size, args.img_size//args.patch_size)
        anomaly_map_rgb_jepa_only = anomaly_segmentor.convert_to_segmentation(dscore_rgb_only_org)[0]
        anomaly_map_rgb_jepa_only = ndimage.gaussian_filter(anomaly_map_rgb_jepa_only, sigma=4)
        # RGB Only Detection Score
        anomaly_rgb_jepa_only_score = np.max(anomaly_map_rgb_jepa_only.reshape(1, -1), axis=-1) * np.max(dscore_rgb_only_org.numpy().reshape(1, -1), axis=-1)

        dscore_xyz_only_org = torch.tensor(dscore_xyz_only_org).unsqueeze(0).reshape(-1, args.img_size//(args.patch_size//2), args.img_size//(args.patch_size//2))
        anomaly_map_xyz_jepa_only = anomaly_segmentor.convert_to_segmentation(dscore_xyz_only_org)[0]
        anomaly_map_xyz_jepa_only = ndimage.gaussian_filter(anomaly_map_xyz_jepa_only, sigma=4)
        anomaly_xyz_jepa_only_score = np.max(anomaly_map_xyz_jepa_only.reshape(1, -1), axis=-1) * np.max(dscore_xyz_only_org.numpy().reshape(1, -1), axis=-1)

        # Fusion Map Score (Z-score Normalization + Maximum Fusion)
        anomaly_rgb_jepa_only = anomaly_map_rgb_jepa_only.reshape(-1)
        anomaly_rgb_jepa_only = (anomaly_rgb_jepa_only - np.mean(anomaly_rgb_jepa_only, axis=-1)) / (np.std(anomaly_rgb_jepa_only, axis=-1) + 1e-8)
        anomaly_xyz_jepa_only = anomaly_map_xyz_jepa_only.reshape(-1)
        anomaly_xyz_jepa_only = (anomaly_xyz_jepa_only - np.mean(anomaly_xyz_jepa_only, axis=-1)) / (np.std(anomaly_xyz_jepa_only, axis=-1) + 1e-8)

        # Just RGB + 3D
        anomaly_map_fusion_2_banks_jepa_only = np.maximum(anomaly_rgb_jepa_only, anomaly_xyz_jepa_only).reshape(224, 224)
        anomaly_fusion_2_banks_jepa_only = anomaly_rgb_jepa_only_score * anomaly_xyz_jepa_only_score * np.max(anomaly_map_fusion_2_banks_jepa_only.reshape(1, -1), axis=-1)

        return anomaly_rgb_only_mem_update, anomaly_xyz_only_mem_update, anomaly_fusion_2_banks, \
               anomaly_map_rgb_combined, anomaly_map_xyz_combined, anomaly_map_fusion_2_banks, \
               anomaly_rgb_jepa_only_score, anomaly_xyz_jepa_only_score, anomaly_fusion_2_banks_jepa_only, \
               anomaly_map_rgb_jepa_only, anomaly_map_xyz_jepa_only, anomaly_map_fusion_2_banks_jepa_only
       

    # 2. Check anomaly detection score
    # Following as M3DM(CVPR 2023) and LSFA(ECCV 2024), we also adopt the off-the-shelf PatchCore algorithm
    anomaly_scores_rgb1 = []
    anomaly_scores_xyz1 = []
    anomaly_scores_rgb2 = []
    anomaly_scores_xyz2 = []
    anomaly_map_rgb1 = []
    anomaly_map_xyz1 = []
    anomaly_map_rgb2 = []
    anomaly_map_xyz2 = []
    anomaly_map_mm1 = []
    anomaly_map_mm2 = []
    anomaly_map_mm3 = []
    anomaly_map_mm4 = []
    anomaly_scores_mm1 = [] 
    anomaly_scores_mm2 = []
    anomaly_scores_mm3 = [] 
    anomaly_scores_mm4 = [] 
    anomaly_maps = []
    anomaly_labels = []
    anomaly_gt_maps = []
    vis_rgb, vis_xyz = [], [] 
    latency_times = [] 
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True) 
    for iteration, data in tqdm(enumerate(testloader), desc="Inference Step ..."):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            starter.record()
        else:
            start_time = time.time()
        rgb_context.eval()
        xyz_context.eval()
        # Memory bank update
        rgb_bank.eval()
        xyz_bank.eval()

        rgb = data["data"][0] # [1, 3, 224, 224]
        rgb_background = data["data"][3] # [1, 1, 224, 224]
        xyz = data["data"][1] # [1, 3, 224, 224] -> need to reshpae into [1, 3, 224*224]
        depth = data["data"][2] # [1, 1, 224, 224]

        label = data["label"]
        anomaly_map = data["anomaly_map"]

        B,C,H,W = xyz.shape
        original_points = xyz.permute(0,2,3,1).view(B, H*W, C)
        xyz = xyz.view(B, C, H*W)

        rgb = rgb.to(device)
        xyz = xyz.to(device)
        depth = depth.to(device)

        rgb_img = rgb.clone()

        xyz_t = xyz.permute(0, 2, 1) # [B, N, 3]
        non_zero_mask = torch.all(xyz_t[0] != 0, dim=1)
        nonzero_indices = torch.nonzero(non_zero_mask, as_tuple=True)[0]
        original_points_valid = xyz_t[:, nonzero_indices, :] # [1, N_valid, 3]

        # # 1. XYZ target encoder extraction
        xyz_feats, centers, \
            ori_idx_tgt, center_idx_tgt = xyz_context(xyz, mode="target")
        
        # 1.1 Point Feature Alignment (interpolation + projection)
        xyz_to_2d = pfa(group_features=xyz_feats.permute(0,2,1), 
                        group_centers=centers,         
                        original_points=original_points_valid,
                        nonzero_indices=nonzero_indices,    
                        kernel_size=args.patch_size//2)
        
        B,D,P,P = xyz_to_2d.shape
        xyz_features = xyz_to_2d.permute(0,2,3,1).reshape(B, P*P, D)

        # 1.1 Point Feature Alignment (interpolation + projection)
        xyz_to_2d2 = pfa(group_features=xyz_feats.permute(0,2,1), 
                         group_centers=centers,         
                         original_points=original_points_valid,
                         nonzero_indices=nonzero_indices,    
                         kernel_size=args.patch_size)
        
        B,D,P,P = xyz_to_2d2.shape
        xyz_features2 = xyz_to_2d2.permute(0,2,3,1).reshape(B, P*P, D)

        # RGB Modality
        h_rgb = rgb_context(rgb, context_mask=None) # B, P*P, D

        # Multi-modality
        bg_indices = (xyz_features2.sum(axis=-1) == 0)

        pred_anomaly_score_rgb_mem_update, \
        pred_anomaly_score_xyz_mem_update, \
        pred_anomaly_score_mm_mem_update, \
        pred_anomaly_map_rgb_only_mm_update, \
        pred_anomaly_map_xyz_only_mm_update, \
        pred_anomaly_map_mm_update, \
        pred_anomaly_score_rgb_jepa_only, \
        pred_anomaly_score_xyz_jepa_only, \
        pred_anomaly_score_mm_jepa_only, \
        pred_anomaly_map_rgb_jepa_only, \
        pred_anomaly_map_xyz_jepa_only, \
        pred_anomaly_map_mm_jepa_only  = eval_anomaly(rgb_ftrs=h_rgb,
                                                      xyz_ftrs=xyz_features,
                                                      l=data['label'].item())
        
        if torch.cuda.is_available():
            ender.record()
            torch.cuda.synchronize()
            curr_latency = starter.elapsed_time(ender) 
        else:
            curr_latency = (time.time() - start_time)

        if iteration >= 10:
            latency_times.append(curr_latency)
        
        # # For I-AUROC
        anomaly_scores_rgb1.append(pred_anomaly_score_rgb_mem_update)
        anomaly_scores_xyz1.append(pred_anomaly_score_xyz_mem_update)
        anomaly_scores_rgb2.append(pred_anomaly_score_rgb_jepa_only)
        anomaly_scores_xyz2.append(pred_anomaly_score_xyz_jepa_only)
        anomaly_scores_mm1.append(pred_anomaly_score_mm_mem_update)
        anomaly_scores_mm2.append(pred_anomaly_score_mm_jepa_only)

        anomaly_map_rgb1.append(np.array(pred_anomaly_map_rgb_only_mm_update))
        anomaly_map_xyz1.append(np.array(pred_anomaly_map_xyz_only_mm_update))
        anomaly_map_rgb2.append(np.array(pred_anomaly_map_rgb_jepa_only))
        anomaly_map_xyz2.append(np.array(pred_anomaly_map_xyz_jepa_only))
        anomaly_map_mm1.append(np.array(pred_anomaly_map_mm_update)) 
        anomaly_map_mm2.append(np.array(pred_anomaly_map_mm_jepa_only))

        anomaly_labels.append(data["label"].item())
        anomaly_gt_maps.append(np.array(data["anomaly_map"].squeeze(0).squeeze(0)))
    
    anomaly_scores_rgb1 = np.array(anomaly_scores_rgb1)
    anomaly_scores_xyz1 = np.array(anomaly_scores_xyz1)
    anomaly_scores_rgb2 = np.array(anomaly_scores_rgb2)
    anomaly_scores_xyz2 = np.array(anomaly_scores_xyz2)
    anomaly_scores_mm1 = np.array(anomaly_scores_mm1)
    anomaly_scores_mm2 = np.array(anomaly_scores_mm2)
    
    # Image-level AUROC (Memory bank update)
    image_auroc_rgb1 = roc_auc_score(y_true=np.stack(anomaly_labels), y_score=np.stack(anomaly_scores_rgb1))
    image_auroc_xyz1 = roc_auc_score(y_true=np.stack(anomaly_labels), y_score=np.stack(anomaly_scores_xyz1))
    # Image-level AUROC (JEPA Only)
    image_auroc_rgb2 = roc_auc_score(y_true=np.stack(anomaly_labels), y_score=np.stack(anomaly_scores_rgb2))
    image_auroc_xyz2 = roc_auc_score(y_true=np.stack(anomaly_labels), y_score=np.stack(anomaly_scores_xyz2))
    # 1. memory bank update, 2. JEPA Only, 3. RGB + 3D (Mem update) 4. RGB + 3D (Only JEPA)
    image_auroc_fused1 = roc_auc_score(y_true=np.stack(anomaly_labels), y_score=np.stack(anomaly_scores_mm1))
    image_auroc_fused2 = roc_auc_score(y_true=np.stack(anomaly_labels), y_score=np.stack(anomaly_scores_mm2))
    # Each Modality P-AUROC (Memory bank update)
    pixel_auroc_rgb1 = roc_auc_score(y_true=np.stack(anomaly_gt_maps).flatten(), y_score=np.stack(anomaly_map_rgb1).flatten())
    pixel_auroc_xyz1 = roc_auc_score(y_true=np.stack(anomaly_gt_maps).flatten(), y_score=np.stack(anomaly_map_xyz1).flatten())
    # Each Modality P-AUROC (JEPA Only)
    pixel_auroc_rgb2 = roc_auc_score(y_true=np.stack(anomaly_gt_maps).flatten(), y_score=np.stack(anomaly_map_rgb2).flatten())
    pixel_auroc_xyz2 = roc_auc_score(y_true=np.stack(anomaly_gt_maps).flatten(), y_score=np.stack(anomaly_map_xyz2).flatten())
    # Multimodal P-AUROC
    pixel_auroc_fused1 = roc_auc_score(y_true=np.stack(anomaly_gt_maps).flatten(), y_score=np.stack(anomaly_map_mm1).flatten())
    pixel_auroc_fused2 = roc_auc_score(y_true=np.stack(anomaly_gt_maps).flatten(), y_score=np.stack(anomaly_map_mm2).flatten())
    
    # AUPRO 30%
    aupro_rgb30_1, _ = calculate_au_pro(anomaly_gt_maps, anomaly_map_rgb1, integration_limit=0.3)
    aupro_xyz30_1, _ = calculate_au_pro(anomaly_gt_maps, anomaly_map_xyz1, integration_limit=0.3)
    aupro_rgb30_2, _ = calculate_au_pro(anomaly_gt_maps, anomaly_map_rgb2, integration_limit=0.3)
    aupro_xyz30_2, _ = calculate_au_pro(anomaly_gt_maps, anomaly_map_xyz2, integration_limit=0.3)
    aupro_mm30_1, _ = calculate_au_pro(anomaly_gt_maps, anomaly_map_mm1, integration_limit=0.3)
    aupro_mm30_2, _ = calculate_au_pro(anomaly_gt_maps, anomaly_map_mm2, integration_limit=0.3)

    # AUPRO 10%
    aupro_rgb10_1, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_rgb1, integration_limit=0.1)
    aupro_xyz10_1, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_xyz1, integration_limit=0.1)
    aupro_rgb10_2, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_rgb2, integration_limit=0.1)
    aupro_xyz10_2, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_xyz2, integration_limit=0.1)
    aupro_mm10_1, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_mm1, integration_limit=0.1) # Memory Update
    aupro_mm10_2, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_mm2, integration_limit=0.1) # JEPA Only

    # AUPRO 5%
    aupro_rgb05_1, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_rgb1, integration_limit=0.05)
    aupro_xyz05_1, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_xyz1, integration_limit=0.05)
    aupro_rgb05_2, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_rgb2, integration_limit=0.05)
    aupro_xyz05_2, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_xyz2, integration_limit=0.05)
    aupro_mm05_1, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_mm1, integration_limit=0.05) # Memory Update
    aupro_mm05_2, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_mm2, integration_limit=0.05) # JEPA Only
    
    # AUPRO 1%
    aupro_rgb01_1, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_rgb1, integration_limit=0.01)
    aupro_xyz01_1, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_xyz1, integration_limit=0.01)
    aupro_rgb01_2, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_rgb2, integration_limit=0.01)
    aupro_xyz01_2, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_xyz2, integration_limit=0.01)
    aupro_mm01_1, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_mm1, integration_limit=0.01) # Memory Update
    aupro_mm01_2, _ = calculate_au_pro(gts=anomaly_gt_maps, predictions=anomaly_map_mm2, integration_limit=0.01) # JEPA Only
    
    # 30%
    aupro_rgb30_1 = aupro_rgb30_1[0]
    aupro_xyz30_1 = aupro_xyz30_1[0]
    aupro_rgb30_2 = aupro_rgb30_2[0]
    aupro_xyz30_2 = aupro_xyz30_2[0]
    aupro_mm30_1 = aupro_mm30_1[0]
    aupro_mm30_2 = aupro_mm30_2[0]

    # 10%
    aupro_rgb10_1 = aupro_rgb10_1[0]
    aupro_xyz10_1 = aupro_xyz10_1[0]
    aupro_rgb10_2 = aupro_rgb10_2[0]
    aupro_xyz10_2 = aupro_xyz10_2[0]
    aupro_mm10_1 = aupro_mm10_1[0]
    aupro_mm10_2 = aupro_mm10_2[0]

    # 5%
    aupro_rgb05_1 = aupro_rgb05_1[0]
    aupro_xyz05_1 = aupro_xyz05_1[0]
    aupro_rgb05_2 = aupro_rgb05_2[0]
    aupro_xyz05_2 = aupro_xyz05_2[0]
    aupro_mm05_1 = aupro_mm05_1[0]
    aupro_mm05_2 = aupro_mm05_2[0]

    aupro_rgb01_1 = aupro_rgb01_1[0]
    aupro_xyz01_1 = aupro_xyz01_1[0]
    aupro_rgb01_2 = aupro_rgb01_2[0]
    aupro_xyz01_2 = aupro_xyz01_2[0]
    aupro_mm01_1 = aupro_mm01_1[0]
    aupro_mm01_2 = aupro_mm01_2[0]
    
    if len(latency_times) > 0:
        avg_latency_ms = np.mean(latency_times)
        fps = 1000 / avg_latency_ms
        logging.info(f"========================================================")
        logging.info(f"[Latency and Frame Rate Measurement]")
        logging.info(f"Average Latency: {avg_latency_ms:.4f} ms")
        logging.info(f"Frame Rate (FPS): {fps:.2f} FPS")
        logging.info(f"========================================================")
    logging.info("----------------------------------------- ************* With Memory bank update ************* ----------------------------------------------------")
    logging.info("-----------------------------------------------** RGB Modality Scores **-----------------------------------------------------------")
    logging.info(f"@ RGB I-AUROC Score: {image_auroc_rgb1:.3f} @")
    logging.info(f"@ RGB P-AUROC Score: {pixel_auroc_rgb1:.3f} @")
    logging.info(f"@ RGB AUPRO30 Score: {aupro_rgb30_1:.3f} @")
    logging.info(f"@ RGB AUPRO01 Score: {aupro_rgb01_1:.3f} @")
    logging.info("-----------------------------------------------** 3D Modality Scores **------------------------------------------------------------")
    logging.info(f"@ XYZ I-AUROC Score: {image_auroc_xyz1:.3f} @")
    logging.info(f"@ XYZ P-AUROC Score: {pixel_auroc_xyz1:.3f} @")
    logging.info(f"@ XYZ AUPRO30 Score: {aupro_xyz30_1:.3f} @")
    logging.info(f"@ XYZ AUPRO01 Score: {aupro_xyz01_1:.3f} @")
    logging.info("------------------------------------------------** Multimodal Scores ** ----------------------------------------------------------")
    logging.info(f"@ MM I-AUROC Score: {image_auroc_fused1:.3f} @ ")
    logging.info(f"@ MM P-AUROC Score: {pixel_auroc_fused1:.3f} @ - Maximum Fusion")
    logging.info(f"@ MM AUPRO30 Score: {aupro_mm30_1:.3f} @ - Maximum Fusion")
    logging.info(f"@ MM AUPRO10 Score: {aupro_mm10_1:.3f} @ - Maximum Fusion")
    logging.info(f"@ MM AUPRO05 Score: {aupro_mm05_1:.3f} @ - Maximum Fusion")
    logging.info(f"@ MM AUPRO01 Score: {aupro_mm01_1:.3f} @ - Maximum Fusion")
    logging.info("------------------------------------------------------------------------------------------------------------------------------------------------")
    logging.info("----------------------------------------- ************* Just 1st stage (JEPA) ************* ----------------------------------------------------")
    logging.info("-----------------------------------------------** RGB Modality Scores **-----------------------------------------------------------")
    logging.info(f"@ RGB I-AUROC Score: {image_auroc_rgb2:.3f} @")
    logging.info(f"@ RGB P-AUROC Score: {pixel_auroc_rgb2:.3f} @")
    logging.info(f"@ RGB AUPRO30 Score: {aupro_rgb30_2:.3f} @")
    logging.info(f"@ RGB AUPRO01 Score: {aupro_rgb01_2:.3f} @")
    logging.info("-----------------------------------------------** 3D Modality Scores **------------------------------------------------------------")
    logging.info(f"@ XYZ I-AUROC Score: {image_auroc_xyz2:.3f} @")
    logging.info(f"@ XYZ P-AUROC Score: {pixel_auroc_xyz2:.3f} @")
    logging.info(f"@ XYZ AUPRO30 Score: {aupro_xyz30_2:.3f} @")
    logging.info(f"@ XYZ AUPRO01 Score: {aupro_xyz01_2:.3f} @")
    logging.info("------------------------------------------------** Multimodal Scores ** ----------------------------------------------------------")
    logging.info(f"@ MM I-AUROC Score: {image_auroc_fused2:.3f} @ ")
    logging.info(f"@ MM P-AUROC Score: {pixel_auroc_fused2:.3f} @ - Maximum Fusion")
    logging.info(f"@ MM AUPRO30 Score: {aupro_mm30_2:.3f} @ - Maximum Fusion")
    logging.info(f"@ MM AUPRO10 Score: {aupro_mm10_2:.3f} @ - Maximum Fusion")
    logging.info(f"@ MM AUPRO05 Score: {aupro_mm05_2:.3f} @ - Maximum Fusion")
    logging.info(f"@ MM AUPRO01 Score: {aupro_mm01_2:.3f} @ - Maximum Fusion")
    logging.info("---------------------------------------------------------------------------------------------------------------------------------------------------")
