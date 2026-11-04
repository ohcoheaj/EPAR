import os
import numpy as np
from torch import nn
import random
import matplotlib.patches as patches
import copy
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.ad_dataset_1st import TrainDataset, ValidationDataset
from utils.loss import CauchyLoss
from models.feature_extractor import RGBBackbone, XYZBackbone
from models import encoder as vit
from models.point_transformer import TransformerPredictor
from models.encoder import VIT_EMBED_DIMS
from models.src.blocks_new_v2 import MaskCollator
from models.src.apply_masks import apply_masks
from models.src.tensors import trunc_normal_
from models.src.schedulers import CosineWDSchedule, WarmupCosineSchedule
# from models.refiner import FocalLoss
# from models.adapter import Adapter
# lora
import loralib as lora


import warnings
warnings.filterwarnings("ignore")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def init_model(device,
               num_patches,
               embed_dim,
               pred_depth,
               num_heads,
               drop_rate=0.0,
               pred_emb_dim=384):
    predictor = vit.__dict__['vit_predictor'](
        num_patches=num_patches,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=num_heads)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in predictor.modules():
        init_weights(m)

    predictor.to(device)
    
    return predictor

def init_opt(
    encoder,
    adapter,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.0
):
    layer_list = [encoder] + predictor
    if adapter is not None:
        layer_list.append(adapter)
    w_decay, wo_decay = [], []
    for layer in layer_list:
        for n, p in layer.named_parameters():
            if n.endswith(".bias") or p.ndim==1: # no decay on layernorm and bias
                wo_decay.append(p)
            else:
                w_decay.append(p)
    param_groups = [{'params': w_decay, 'weight_decay': wd},
                    {'params': wo_decay, 'weight_decay': 0.0}]
        
    optimizer = optim.AdamW(param_groups, lr=ref_lr)
    # Warmingup Schedule
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    # Cosine Weight Decay scheduling
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler


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

    def worker_init_fn(worker_id, seed=42):
        worker_seed = seed
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
    rgb_target = copy.deepcopy(rgb_context)
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
    xyz_target = copy.deepcopy(xyz_context)

    # Unfreeze the Encoder and Positional Embedding of 3D Point Encoder is appropriate
    for n, p in xyz_context.named_parameters():
        if n.split(".")[1] == "encoder":
            p.requires_grad = True
        elif n.split(".")[1] == "pos_embed":
            p.requires_grad = True
        else:
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
    rgb_target = copy.deepcopy(rgb_context)
    
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
    xyz_target = copy.deepcopy(xyz_context)
    xyz_target = xyz_target.to(device)

    for n, p in rgb_target.named_parameters():
        p.requires_grad = False
    for p in xyz_target.parameters():
        p.requires_grad = False
    
    num_params = sum(p.numel() for p in rgb_target.parameters() if p.requires_grad)
            
    rgb_context = rgb_context.to(device)
    xyz_context = xyz_context.to(device)
    rgb_target = rgb_target.to(device)
    xyz_target = xyz_target.to(device)
    
    rgb_predictor = init_model(device,
                               num_patches=((args.img_size // args.patch_size) * (args.img_size // args.patch_size)),
                               embed_dim=768,
                               pred_depth=args.predictor_depth,
                               num_heads=6,
                               drop_rate=0.1,
                               pred_emb_dim=384)
    rgb_predictor = rgb_predictor.to(device)

    xyz_predictor = TransformerPredictor(
        embed_dim=384,
        predictor_embed_dim=192,
        depth=args.predictor_depth,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.05,
        drop_path_rate=0.10,
        add_pos_at_every_layer=True,
        add_target_pos=True
    )
    xyz_predictor = xyz_predictor.to(device)

    # Initialize the dataset
    # need to align Point Clouds and RGB images (Point Feature Alignment)
    trainset = TrainDataset(args.class_name, args.img_size, dataroute, dtd_dataset_path=args.dtd_dataset_path, k_shot=args.k_shot)
    validset = ValidationDataset(args.class_name, args.img_size, dataroute, dtd_dataset_path=args.dtd_dataset_path)

    trainloader = DataLoader(dataset=trainset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             worker_init_fn=worker_init_fn,
                             drop_last=False,
                             pin_memory=True)
    
    # Context, Target Encoder
    encoder_emb_dim = VIT_EMBED_DIMS[args.encoder_size]
    cos_sim = nn.CosineEmbeddingLoss()

    # update for ema
    ema_rgb = args.ema_params_rgb
    ema_3d = args.ema_params_3d
    ipe = len(trainloader)
    ipe_scale = 1.0
    # EMA Scheduler
    momentum_scheduler_rgb = (ema_rgb[0] + i*(ema_rgb[1]-ema_rgb[0])/(ipe*args.num_epochs*ipe_scale)
                              for i in range(int(ipe*args.num_epochs*ipe_scale)+1))
    momentum_scheduler_3d = (ema_3d[0] + i*(ema_3d[1]-ema_3d[0])/(ipe*args.num_epochs*ipe_scale)
                             for i in range(int(ipe*args.num_epochs*ipe_scale)+1))
    
    
    # init optimizer
    optimizer_rgb, scaler_rgb, scheduler_rgb, wd_scheduler_rgb = init_opt(encoder=rgb_context,
                                                                          adapter=None,
                                                                          predictor=[rgb_predictor],
                                                                          wd=args.weight_decay,
                                                                          iterations_per_epoch=ipe,
                                                                          final_wd=args.final_weight_decay,
                                                                          start_lr=args.start_lr_rgb,
                                                                          ref_lr=args.lr_rgb,
                                                                          final_lr=args.final_lr_rgb,
                                                                          warmup=args.warmup_epochs,
                                                                          num_epochs=args.num_epochs,
                                                                          use_bfloat16=args.usage_half_precision)
    optimizer_xyz, scaler_xyz, scheduler_xyz, wd_scheduler_xyz = init_opt(encoder=xyz_context,
                                                                          adapter=None,
                                                                          predictor=[xyz_predictor],
                                                                          wd=args.weight_decay,
                                                                          iterations_per_epoch=ipe,
                                                                          final_wd=args.final_weight_decay,
                                                                          start_lr=args.start_lr_xyz,
                                                                          ref_lr=args.lr_xyz,
                                                                          final_lr=args.final_lr_xyz,
                                                                          warmup=args.warmup_epochs,
                                                                          num_epochs=args.num_epochs,
                                                                          use_bfloat16=args.usage_half_precision)
    
    def loss_fn(z, h, loss_type):
        if loss_type == "smooth":
            loss = F.smooth_l1_loss(z, h) # Smooth L1 Loss
            return loss
        if loss_type == "cosine":
            nl = z.size(0)
            loss = 0.0
            for i in range(nl):
                z_ = F.normalize(z[i,:,:], p=2, dim=1)
                h_ = F.normalize(h[i,:,:], p=2, dim=1)
                loss += F.cosine_embedding_loss(z_, h_, torch.ones(z.size(1)).to(z.device))
            return loss / nl
    
    def make_tgt_block(bsz, patch_size, foreground_indices=None, background_indices=None, pred_mask_scale=None, mode=None):
        # Target block complements
        target_block_complements = torch.zeros([bsz, args.img_size // patch_size, args.img_size // patch_size])
        min_row, max_row, min_col, max_col = 9999, 0, 9999, 0
        min_keep = (args.img_size//patch_size) * (args.img_size//patch_size)
        for i in range(bsz):
            f_idx = np.array(foreground_indices[i])
            min_keep = min(int(len(f_idx) * pred_mask_scale[0]), min_keep)
            f_rows = f_idx // (args.img_size//patch_size)
            f_cols = f_idx % (args.img_size//patch_size)
            # make target block complements
            target_block_complements[i][f_rows, f_cols] = 1.0
            # get indices
            idxs = np.array([[j//(args.img_size//patch_size), j%(args.img_size//patch_size)] for j in foreground_indices[i]])
            min_row = min(min_row, idxs[:,0].min())
            max_row = max(max_row, idxs[:,0].max())
            min_col = min(min_col, idxs[:,1].min())
            max_col = max(max_col, idxs[:,1].max())
        
        # calculate along batch
        row_range = [min_row, max_row]
        col_range = [min_col, max_col]

        # Rectangular like target block complements
        for b in range(len(target_block_complements)):
            for i in range(row_range[0], row_range[1]+1):
                for j in range(col_range[0], col_range[1]+1):
                    target_block_complements[b][i, j] = 1

        mask_collator = MaskCollator(input_size=(args.img_size, args.img_size),
                                     patch_size=patch_size,
                                     pred_mask_scale=pred_mask_scale,
                                     aspect_ratio=args.aspect_ratio,
                                     npred=args.npred,
                                     min_keep=min_keep,
                                     allow_overlap=False,
                                     mode=mode)
        
        context_indices = torch.arange((args.img_size//patch_size)*(args.img_size//patch_size))

        # Just exclude the background indices from context indices (2026.1.11)
        exclude_indices = torch.tensor(background_indices).flatten()
        mask = torch.ones(context_indices.shape[0], dtype=torch.bool)
        mask[exclude_indices] = False
        context_indices = context_indices[mask]

        context_blocks, target_blocks = mask_collator(bsz=bsz,
                                                      row_range=row_range,
                                                      col_range=col_range,
                                                      target_acceptable_regions=target_block_complements,
                                                      background_indices=background_indices,
                                                      context_indices=context_indices)
        
        return context_blocks, target_blocks, target_block_complements.reshape(bsz, (args.img_size//patch_size)*(args.img_size//patch_size))
    
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

    def image_to_grayscale(image):
        """Convert RGB image to grayscale. image: (3, H, W) -> (1, H, W)"""
        if image.shape[0] == 3:
            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            return gray.unsqueeze(0)
        return image
    def otsu_threshold(gray_image):
        """Compute Otsu's threshold for a grayscale image tensor.
        gray_image: (1, H, W) with values in [0, 255]
        Returns: optimal threshold (float)
        """
        pixels = gray_image.flatten().float()
        hist = torch.histc(pixels, bins=256, min=0, max=255)
        total = pixels.numel()
        
        sum_total = torch.dot(torch.arange(256, dtype=torch.float32, device=pixels.device), hist)
        
        sum_bg = 0.0
        weight_bg = 0
        max_variance = 0.0
        threshold = 0
        
        for t in range(256):
            weight_bg += hist[t].item()
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            
            sum_bg += t * hist[t].item()
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total.item() - sum_bg) / weight_fg
            
            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if variance > max_variance:
                max_variance = variance
                threshold = t
        
        return float(threshold)
    
    def find_significant_patches(patches, bg_threshold, bg_ratio_threshold):
        N, patch_dim = patches.shape
        P = int((patch_dim) ** 0.5)
        patches = patches.view(N, 1, P, P)
        
        # background pixel mask
        bg_pixel_mask = (patches <= bg_threshold).float()
        bg_proportion = bg_pixel_mask.mean(dim=[1,2,3])  # (N,)
        
        significant_indices = torch.where(bg_proportion < bg_ratio_threshold)[0].tolist()
        return significant_indices

    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    # Training loop (JEPA Training Loop)
    vic_rate = args.vic_rate
    for epoch in range(args.num_epochs):
        print(f"@@@@@@@@@@@ Class: {args.class_name}, Epoch: {epoch+1}/{args.num_epochs} Training Starts @@@@@@@@@@@")
        # train mode
        rgb_context.train()
        rgb_predictor.train()
        rgb_target.train()
        xyz_context.train()
        xyz_predictor.train()
        xyz_target.train()
        # Training loop
        for iteration, data in tqdm(enumerate(trainloader)):
            # load data (keep this)
            rgb = data["data"][0] # [bsz, 3, 224, 224]
            xyz = data["data"][1] # [bsz, 3, 224, 224] -> need to reshpae into [bsz, 3, 224*224]
            depth = data["data"][2] # [bsz, 3, 224, 224] # depth image shape (Repeat [1, 224, 224])
            label = data["label"]

            B,C,H,W = xyz.shape
            xyz = xyz.view(B, C, H*W)

            rgb = rgb.to(device)
            xyz = xyz.to(device)
            
            # 1. XYZ Context encoder, target encoder extraction (Normal Point Clouds, depthr)
            xyz_features_ctx, center_ctx,\
                ori_idx_ctx, center_idx_ctx, \
                    tgt_blocks_ctx, target_indices, \
                        context_tokens_ctx, context_centers_ctx, \
                            original_center, context_indices_xyz = xyz_context(xyz, npred=args.npred, mode="context")
            xyz_features_tgt, center_tgt, \
                ori_idx_tgt, center_idx_tgt = xyz_target(xyz, mode="target")

            # control context
            xyz_features_ctx = torch.cat([xyz_features_ctx[:,:384,:].unsqueeze(0), \
                                          xyz_features_ctx[:,384:768,:].unsqueeze(0), \
                                          xyz_features_ctx[:,768:,:].unsqueeze(0)], dim=0)

            xyz_features_ctx = xyz_features_ctx.permute(1,0,3,2) # bsz, num_level, n_ctxt, dim

            # control target
            xyz_features_tgt = torch.cat([xyz_features_tgt[:,:384,:].unsqueeze(0), \
                                          xyz_features_tgt[:,384:768,:].unsqueeze(0), \
                                          xyz_features_tgt[:,768:,:].unsqueeze(0)], dim=0)
            xyz_features_tgt = xyz_features_tgt.permute(1,0,3,2) # bsz, num_level, n_ctxt, dim

            # invariant loss (Invariant Regularization)
            b, n, p, d = xyz_features_tgt.shape
            xyz_features_tgt_invar = xyz_features_tgt.permute(0,2,1,3).reshape(b,p,n*d)
            xyz_features_tgt_invar = xyz_features_tgt_invar[:, context_indices_xyz, :]
            b, n, p, d = xyz_features_ctx.shape
            xyz_features_ctx_invar = xyz_features_ctx.permute(0,2,1,3).reshape(b,p,n*d)
            local_invariant_loss_xyz = 0.0
            bsz, npatches, d = xyz_features_ctx_invar.shape
            for b in range(bsz):
                s_xyz = xyz_features_ctx_invar[b,:,:]
                t_xyz = xyz_features_tgt_invar[b,:,:]
                local_invariant_loss_xyz += cos_sim(F.normalize(s_xyz, dim=1),
                                                    F.normalize(t_xyz, dim=1).detach(),
                                                    target = torch.ones(npatches).to(device))
            local_invariant_loss_xyz /= bsz

            # Variance-Covariance Regularizer
            bsz, nctxt, d = xyz_features_ctx_invar.shape
            var_loss, cov_loss = 0.0, 0.0
            for i in range(bsz):
                xyz_invar = xyz_features_ctx_invar[i,:,:] - xyz_features_ctx_invar[i,:,:].mean(dim=0)
                std_xyz = torch.sqrt(xyz_invar.var(dim=0) + 1e-4) # epsilon for numerical stability
                var_loss += torch.mean(F.relu(1 - std_xyz))
                cov_xyz = (xyz_invar.T @ xyz_invar) / (nctxt - 1)
                cov_loss += off_diagonal(cov_xyz).pow_(2).sum().div(d)
            var_loss /= bsz
            cov_loss /= bsz
            vic_reg_xyz = 15*local_invariant_loss_xyz + 15*var_loss + cov_loss
      
            # Num-level: 3
            bsz, n_l, p, d = xyz_features_tgt.shape
            # calc loss
            loss_xyz_jepa = 0.0
            for i in range(bsz): # Multi-level prediction
                xyz_features_ctx_batch = xyz_features_ctx[i,:,:,:]
                xyz_features_ctx_batch = [xyz_features_ctx_batch[k,:,:].unsqueeze(0) for k in range(n_l)]
                xyz_features_ctx_batch = torch.cat(xyz_features_ctx_batch, dim=2) # [1, n_target, dim*3]
                xyz_features_tgt_batch = xyz_features_tgt[i,:,:,:]
                t_l = target_indices.shape[0]
                jepa_loss = 0.0
                for m in range(t_l): # iteration through the number of target blocks
                    tgt_indices = target_indices[m]
                    tgt_center = original_center[i, tgt_indices, :] # bsz, n_tgt_blocks, 3
                    xyz_pred = xyz_predictor(
                        xyz_features_ctx_batch, context_centers_ctx[i].unsqueeze(0), tgt_center.unsqueeze(0)
                    )
                    xyz_tgt = xyz_features_tgt_batch[:,tgt_indices,:]
                    jepa_loss += loss_fn(z=xyz_pred, h=xyz_tgt.detach(), loss_type="smooth")
                loss_xyz_jepa += jepa_loss
            loss_xyz_jepa /= (bsz*t_l)

            foreground_idxs = []
            background_idxs = []
            for img in rgb:
                img_check = img.clone().detach().cpu()
                IMAGENET_MEAN = [0.485, 0.456, 0.406] # ImageNet mean
                IMAGENET_STD = [0.229, 0.224, 0.225] # ImageNet std
                img_check = img_check * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                img_check = (img_check * 255.0).to(torch.int)
                gray = image_to_grayscale(img_check)
                auto_threshold = otsu_threshold(gray)
                patch = image_to_patches(img_check, args.patch_size)
                f_idxs = find_significant_patches(
                    patch, auto_threshold, args.bg_ratio_threshold)
                all_idxs = set(range((args.img_size // args.patch_size) ** 2))
                background_idxs.append(list(all_idxs - set(f_idxs)))
                foreground_idxs.append(f_idxs)
            
            # RGB Modality Foreground Selection
            rgb_context_blocks, rgb_target_blocks, masks = make_tgt_block(bsz=args.batch_size,
                                                                          patch_size=args.patch_size,
                                                                          foreground_indices=foreground_idxs,
                                                                          background_indices=background_idxs,
                                                                          pred_mask_scale=args.pred_mask_scale,
                                                                          mode="train")
            
            z_rgb = rgb_context(rgb,
                                context_mask=rgb_context_blocks)
            z_rgb_invar = z_rgb.clone()

            # Predictor Pass (RGB Modality)
            z_rgb = rgb_predictor(z_rgb, rgb_context_blocks, rgb_target_blocks, device=device)
            
            # Make Target (RGB)
            h_rgb = rgb_target(rgb,
                               context_mask=None)
            h_rgb = h_rgb.detach()
            
            _, _, dim = h_rgb.shape
            h_rgb = torch.cat([h_rgb[:, :, :dim//3].unsqueeze(0), \
                               h_rgb[:, :, dim//3:(dim//3)*2].unsqueeze(0), \
                               h_rgb[:, :, (dim//3)*2:].unsqueeze(0)], dim=0) # [3, bsz, n_tgt, d]
            target_rgb, invar_rgb = [], []
            for i in range(len(h_rgb)):
                irgb = h_rgb[i]        
                trgb = apply_masks(irgb, rgb_target_blocks, device=device)
                crgb = apply_masks(irgb, rgb_context_blocks, device=device)
                target_rgb.append(trgb.unsqueeze(0))
                invar_rgb.append(crgb.unsqueeze(0))
            target_rgb = torch.cat(target_rgb, dim=0) # [3, bsz, n_tgt, d]
            invar_rgb = torch.cat(invar_rgb, dim=0)

            # Invariance Regularizer (RGB Modality)
            nl, bsz, ntgt, d = invar_rgb.shape
            invar_rgb = invar_rgb.permute(1,2,0,3)
            invar_rgb = invar_rgb.reshape(bsz, ntgt, nl*d)
            local_invariant_loss_rgb = 0.0
            bsz, npatches, d = z_rgb_invar.shape
            for b in range(bsz):
                s_rgb = z_rgb_invar[b,:,:]
                t_rgb = invar_rgb[b,:,:]
                local_invariant_loss_rgb += cos_sim(F.normalize(s_rgb, dim=1),
                                                    F.normalize(t_rgb, dim=1),
                                                    target = torch.ones(npatches).to(device))
            local_invariant_loss_rgb /= bsz
            
            # RGB Variance-Covariance Regularizer (prevent Representation Collapse)
            bsz, nctxt, d = z_rgb_invar.shape
            var_loss, cov_loss = 0.0, 0.0
            for i in range(bsz):
                rgb_invar = z_rgb_invar[i,:,:] - z_rgb_invar[i,:,:].mean(dim=0)
                std_rgb = torch.sqrt(rgb_invar.var(dim=0) + 1e-4) # epsilon for numerical stability
                var_loss += torch.mean(F.relu(1 - std_rgb))
                cov_rgb = (rgb_invar.T @ rgb_invar) / (nctxt - 1)
                cov_loss += off_diagonal(cov_rgb).pow_(2).sum().div(d)
            var_loss /= bsz
            cov_loss /= bsz
            vic_reg_rgb = 15*local_invariant_loss_rgb + 15*var_loss + cov_loss
            # RGB JEPA Loss
            loss_rgb_jepa = 0.0
            z_rgb = z_rgb.permute(1,0,2,3) # [4*bsz, nlevel, ntgt, d]
            target_rgb = target_rgb.permute(1,0,2,3) # [4*bsz, nlevel, n_tgt, d]
            for i in range(len(target_rgb)):
                loss_rgb_jepa += loss_fn(z=z_rgb[i], h=target_rgb[i].detach(), loss_type="smooth")
            loss_rgb_jepa /= len(target_rgb)

            step_loss_rgb = (loss_rgb_jepa + vic_rate*(vic_reg_rgb))
            step_loss_xyz = (loss_xyz_jepa + vic_rate*(vic_reg_xyz))
        
            print(f"XYZ JEPA Loss is as: {loss_xyz_jepa} | RGB JEPA Loss is as: {loss_rgb_jepa} | VICREG loss XYZ: {vic_reg_xyz} | VICREG loss RGB: {vic_reg_rgb} |")
            # optimization (RGB)
            optimizer_rgb.zero_grad()
            if args.usage_half_precision:
                scaler_rgb.scale(step_loss_rgb).backward()
            else:
                step_loss_rgb.backward()
            optimizer_rgb.step()
            # Scheduler update (after an epoch)
            scheduler_rgb.step()
            wd_scheduler_rgb.step()
            
            # optimization (3D)
            optimizer_xyz.zero_grad()
            if args.usage_half_precision:
                scaler_xyz.scale(step_loss_xyz).backward()
            else:
                step_loss_xyz.backward(retain_graph=True)
            optimizer_xyz.step()
            # Scheduler update (after an epoch)
            scheduler_xyz.step()
            wd_scheduler_xyz.step()
            
            # Momentum encoder update: Exponential Moving Average
            with torch.no_grad():
                m = next(momentum_scheduler_rgb)
                for q, k in zip(rgb_context.parameters(), rgb_target.parameters()):
                    k.data.mul_(m).add_((1 - m) * q.detach().data)
            
            with torch.no_grad():
                m = next(momentum_scheduler_3d)
                for q, k in zip(xyz_context.parameters(), xyz_target.parameters()):
                    k.data.mul_(m).add_((1 - m) * q.detach().data)
            
        if (epoch+1) == args.num_epochs:
            # Save Last epoch (1st JEPA)
            torch.save(rgb_context.state_dict(), f"{args.stage1_dir}/{args.class_name}_best_jepa_rgb_dino_main_{args.npred}blocks_full_{vic_rate}.pth")
            torch.save(xyz_context.state_dict(), f"{args.stage1_dir}/{args.class_name}_best_jepa_xyz_pmae_main_{args.npred}blocks_full_{vic_rate}.pth")
