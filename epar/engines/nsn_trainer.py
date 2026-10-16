"""2nd Stage Training for Multimodal Anomaly Detection (EPAR)"""

import os
import numpy as np
from torch import nn
import random
import matplotlib.patches as patches
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.ad_dataset_2nd import TestDataset, TrainDataset, ValidationDataset
from utils.sampler import ApproximateGreedyCoresetSampler
from models.feature_extractor import RGBBackbone, XYZBackbone
from models.point_feature_alignment import PointFeatureAlignment
from models.common import FaissNNRGB, FaissNN3D, NearestNeighbourScorerRGB, NearestNeighbourScorer3D, RescaleSegmentor
from models.memory_bank import MMUpdateRGB, MMUpdate3D, FocalLoss
# lora
import loralib as lora
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
    trainloader = DataLoader(dataset=trainset,
                             batch_size=args.batch_size,
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
    
    
    def image_to_patches(image, patch_size):
        """
        image: (3, H, W) -> RGB input
        output: (num_patches, patch_dim)
        """
        if image.shape[0] == 3: 
            image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            image = image.unsqueeze(0)
        
        C, H, W = image.shape
        patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(C, -1, patch_size, patch_size)
        patches = patches.permute(1, 0, 2, 3)  # (num_patches, 1, patch_size, patch_size)

        N, C, P, P = patches.shape
        patches = patches.reshape(N, C * P * P)  # (N, patch_dim)
        
        return patches
    
    def find_significant_patches(patches, bg_threshold, bg_ratio_threshold):
        N, patch_dim = patches.shape
        P = int((patch_dim) ** 0.5)
        patches = patches.view(N, 1, P, P)
        
        # background pixel mask
        bg_pixel_mask = (patches <= bg_threshold).float()
        bg_proportion = bg_pixel_mask.mean(dim=[1,2,3])  # (N,)
        
        significant_indices = torch.where(bg_proportion < bg_ratio_threshold)[0].tolist()
        return significant_indices

    # RGB Memory Bank
    nearest_search_rgb_l2 = FaissNNRGB(True, 4, mode="dist")
    anomaly_scorer_rgb = NearestNeighbourScorerRGB(n_nearest_neighbours=5,
                                                   nn_method_l2=nearest_search_rgb_l2)
    # 3D Memory Bank
    nearest_search_xyz_l2 = FaissNN3D(True, 4, mode="dist")
    anomaly_scorer_xyz = NearestNeighbourScorer3D(n_nearest_neighbours=5,
                                                  nn_method_l2=nearest_search_xyz_l2)
    
    # Anomaly Map Rescaler
    anomaly_segmentor = RescaleSegmentor(
        device=torch.device("cpu"), 
        target_size=args.img_size
    )
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # train mode
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

    # Seed setting
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)
    
    # Update Memory Bank
    rgb_bank = MMUpdateRGB(init_memory_bank=torch.from_numpy(coreset_samples_rgb_only),
                           embed_dim=2304*2)
    xyz_bank = MMUpdate3D(init_memory_bank=torch.from_numpy(coreset_samples_xyz_only),
                          embed_dim=1152*2)
        
    rgb_bank = rgb_bank.to(device)
    xyz_bank = xyz_bank.to(device)
    opt_rgb_bank = torch.optim.AdamW(rgb_bank.parameters(),
                                     lr=args.bank_lr_rgb)
    opt_xyz_bank = torch.optim.AdamW(xyz_bank.parameters(),
                                     lr=args.bank_lr_xyz)
    focal_loss = FocalLoss()
    for ep in tqdm(range(args.num_epochs), desc="bank update ..."):
        nsamples = xyz_only_embs.shape[0]
        loss_rgb, loss_xyz, loss_mm = 0.0, 0.0, 0.0
        for i in range(nsamples):
            # GT
            rgb_no = rgb_original[i,:,:,:].unsqueeze(0)
            xyz_no = xyz_original[i,:,:,:].unsqueeze(0)
            rgb_ano = rgb_anomaly[i,:,:,:].unsqueeze(0)
            xyz_ano = xyz_anomaly[i,:,:,:].unsqueeze(0)
            mask = anomaly_masks[i,:,:,:].unsqueeze(0)
            # # Get Features
            rgb_ftr_ab = rgb_anomaly_embs[i,:,:]
            xyz_ftr_ab = xyz_anomaly_embs[i,:,:]
            # bg_anomaly = (xyz_ftr_ab.sum(axis = -1) == 0)
            bg_anomaly = bg_indices_abnormal[i,:]
            bg_normal = bg_indices_normal[i,:]
            rgb_ftr = rgb_only_embs[i,:,:]
            xyz_ftr = xyz_only_embs[i,:,:]
            # bg_normal = (xyz_ftr.sum(axis=-1)==0)
            rgb_res_abnormal1, rgb_res_abnormal2 = rgb_bank(rgb_ftr_ab.to(device))
            xyz_res_abnormal1, xyz_res_abnormal2 = xyz_bank(xyz_ftr_ab.to(device))
            rgb_loss = focal_loss(rgb_res_abnormal1.reshape(1,-1),
                                  mask.reshape(1,-1).to(device)) + F.smooth_l1_loss(rgb_res_abnormal1.reshape(1, -1), mask.reshape(1,-1).to(device))
            xyz_loss = focal_loss(xyz_res_abnormal1.reshape(1,-1),
                                  mask.reshape(1,-1).to(device)) + F.smooth_l1_loss(xyz_res_abnormal1.reshape(1, -1), mask.reshape(1,-1).to(device))
                
            # optimization
            opt_rgb_bank.zero_grad()
            rgb_loss.backward()
            opt_rgb_bank.step()
            opt_xyz_bank.zero_grad()
            xyz_loss.backward()
            opt_xyz_bank.step()
            loss_rgb += rgb_loss.item()
            loss_xyz += xyz_loss.item()

        print(f'At epoch: {ep} RGB Loss: {loss_rgb / nsamples} | 3D Loss: {loss_xyz / nsamples}')
        
    # Save Memory Bank
    torch.save(rgb_bank.state_dict(), f"{args.stage2_dir}/{args.class_name}_rgb_bank_main_{args.npred}blocks_full_{args.vic_rate}.pth")
    torch.save(xyz_bank.state_dict(), f"{args.stage2_dir}/{args.class_name}_xyz_bank_main_{args.npred}blocks_full_{args.vic_rate}.pth")