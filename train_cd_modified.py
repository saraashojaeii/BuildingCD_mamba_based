import torch
import os
# Set CUDA memory management before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch.optim as optim
from tqdm import tqdm
import data as Data
import models as Model
import torch.nn as nn
import argparse
import logging
import core.logger as Logger
import numpy as np
from misc.metric_tools import ConfuseMatrixMeter
from models.loss import *
from collections import OrderedDict
import core.metrics as Metrics
from misc.torchutils import get_scheduler, save_network
import wandb
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
from itertools import islice

def normalize_change_target(seg1: torch.Tensor | None,
                            seg2: torch.Tensor | None,
                            change_gt: torch.Tensor | None) -> torch.Tensor:
    """Return binary change target of shape [B, H, W] (dtype long, {0,1}).

    Handles cases where:
    - change_gt is provided in various formats (NCHW with C=1, NHWC RGB, or [B,H,W] int/float)
    - or must be derived from seg1 and seg2 which may be RGB (NHWC), one-hot/logits (NCHW, C>1),
      or single-channel (NCHW, C=1 or [B,H,W]).
    """
    def _to_index_mask(x: torch.Tensor) -> torch.Tensor:
        # Convert arbitrary segmentation label tensor to [B,H,W] integer indices
        if x.dim() == 4:
            # NCHW
            if x.size(1) == 1:
                return x.squeeze(1).long()
            # NHWC (e.g., RGB mask)
            if x.size(-1) == 3 and x.shape[1] != 3:
                # Assume channels-last
                return x.any(dim=-1).long()  # fallback to binary presence per-pixel
            # Multi-channel: take argmax as class indices
            return torch.argmax(x, dim=1).long()
        elif x.dim() == 3:
            # Already [B,H,W]
            return x.long()
        else:
            raise ValueError(f"Unsupported seg shape: {tuple(x.shape)}")

    if change_gt is not None:
        c = change_gt
        # If NHWC RGB -> any over last channel
        if c.dim() == 4 and c.size(-1) == 3 and (c.shape[1] != 3):
            c = c.any(dim=-1).long()
        elif c.dim() == 4 and c.size(1) == 1:
            c = c.squeeze(1).long()
        elif c.dim() == 3:
            # [B,H,W] possibly float/binary
            c = (c > 0).long()
        else:
            # As a conservative fallback
            c = _to_index_mask(c)
            c = (c > 0).long()
        return c

    # Derive from seg1 and seg2
    if seg1 is None or seg2 is None:
        raise ValueError("seg1/seg2 required when change_gt is None")

    # Handle RGB NHWC
    if seg1.dim() == 4 and seg1.size(-1) == 3 and (seg1.shape[1] != 3) and \
       seg2.dim() == 4 and seg2.size(-1) == 3 and (seg2.shape[1] != 3):
        change = (seg1 != seg2).any(dim=-1).long()
        return change

    # Convert to class indices if needed
    s1 = _to_index_mask(seg1)
    s2 = _to_index_mask(seg2)
    change = (s1 != s2).long()
    return change

def create_color_mask(tensor, num_classes: int = 10):
    """Convert a 2-D label tensor/ndarray to an RGB image with a categorical colormap.

    This is used for logging multi-class segmentation masks to wandb so that they
    appear in color instead of a binary/grayscale mask.
    """
    import numpy as _np
    import matplotlib as _mpl

    # Convert to numpy array
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = _np.asarray(tensor)

    # Remove singleton dimensions if they exist (e.g. 1×H×W)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = _np.squeeze(arr, axis=0)
    
    # Handle case where ground truth is already RGB (H, W, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Already an RGB image, return as uint8
        return arr.astype(_np.uint8)
    
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D mask or 3-D RGB image, got shape {arr.shape}")

    h, w = arr.shape
    unique_vals = _np.unique(arr)
    
    # Fix matplotlib deprecation warning and ensure class 0 is visible
    cmap = _mpl.colormaps.get_cmap('tab10')
    if hasattr(cmap, 'resampled'):
        cmap = cmap.resampled(num_classes)
    rgb = _np.zeros((h, w, 3), dtype=_np.uint8)
    
    # Custom color mapping to ensure class 0 is visible (not black)
    colors = []
    for i in range(num_classes):
        color = _np.array(cmap(i)[:3]) * 255
        # If color is too dark (close to black), make it brighter
        if _np.sum(color) < 50:  # Very dark color
            color = _np.array([255, 0, 0])  # Make it red instead
        colors.append(color.astype(_np.uint8))
    
    # Apply color mapping
    for cls in range(num_classes):
        if cls in unique_vals:
            rgb[arr == cls] = colors[cls]
    
    return rgb

if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/saraashojaeii/git/BuildingCD_mamba_based/config/second_cdmamba/second_cdmamba.json',
                        help='JSON file for configuration')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'], help='Run either train(training + validation) or testing',)
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('-log_eval', action='store_true')
    # Accept naming-related args so CLI doesn't error (used only for run naming)
    parser.add_argument('--model', type=str, default='', help='Model name (for run naming only)')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name (for run naming only)')
    parser.add_argument('--tag', type=str, default='', help='Optional custom tag (for run naming only)')
    # Accept seed here as well (even though seeding uses early_args)
    parser.add_argument('--seed', type=int, default=None, help='Optional; accepted for compatibility')
    # Limits for overfitting/quick runs
    parser.add_argument('--max_train_batches', type=int, default=0, help='Limit number of training batches per epoch (0 = no limit)')
    parser.add_argument('--max_val_batches', type=int, default=0, help='Limit number of validation batches per epoch (0 = no limit)')
    parser.add_argument('--max_test_batches', type=int, default=0, help='Limit number of test batches (0 = no limit)')
    # Threshold for converting probs to binary mask (class-1)
    parser.add_argument('--change_threshold', type=float, default=0.5, help='Probability threshold for change class (class-1) binarization')
    # Auxiliary self-supervised loss weight
    parser.add_argument('--aux_recon_weight', type=float, default=0.1, help='Weight for auxiliary reconstruction loss')

    # Parse config
    args = parser.parse_args()
    opt = Logger.parse(args)

    #Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # Create a unique timestamped experiment subfolder for logs/results/checkpoints
    exp_timestamp = datetime.now().strftime('%Y%m%d_%H')
    exp_name = opt.get('name', 'experiment')
    dataset_suffix = getattr(args, 'dataset', None) or ''
    tag_suffix = getattr(args, 'tag', None) or ''
    suffix_parts = []
    if dataset_suffix:
        suffix_parts.append(str(dataset_suffix))
    if tag_suffix:
        suffix_parts.append(str(tag_suffix))
    suffix = '_'.join(suffix_parts)
    if suffix:
        exp_folder = f"{exp_name}_{exp_timestamp}_{suffix}"
    else:
        exp_folder = f"{exp_name}_{exp_timestamp}"
    for k in ['log', 'result', 'checkpoint']:
        if k in opt['path_cd'] and isinstance(opt['path_cd'][k], str):
            base_dir = opt['path_cd'][k]
            stamped = os.path.join(base_dir, exp_folder)
            opt['path_cd'][k] = stamped
            os.makedirs(stamped, exist_ok=True)
    # Keep the subfolder name for reference
    opt['path_cd']['exp_folder'] = exp_folder

    #logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(logger_name=None, root=opt['path_cd']['log'], phase='train',
                        level=logging.INFO, screen=True)
    Logger.setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test',
                        level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # Set device with comprehensive debugging
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # GPU Debugging Information
    logger.info(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'CUDA Device Count: {torch.cuda.device_count()}')
        logger.info(f'Current CUDA Device: {torch.cuda.current_device()}')
        logger.info(f'CUDA Device Name: {torch.cuda.get_device_name()}')
        logger.info(f'CUDA Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB')
        logger.info(f'CUDA Memory Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB')
    else:
        logger.warning('CUDA is not available! Training will run on CPU (very slow)')

    # Initialize wandb only on main process
    if opt.get('wandb') and opt['wandb'].get('project'):
        # Compose run name with dataset/tag suffixes for wandb as well
        run_name = exp_folder
        wandb.init(project=opt['wandb']['project'], config=opt, name=run_name)
        try:
            if hasattr(wandb, 'run') and wandb.run is not None:
                wandb.run.name = run_name
        except Exception:
            pass
    else:
        wandb.init(mode="disabled")

    #dataset
    for phase, dataset_opt in opt['datasets'].items(): #train train{}
        #print(" phase is {}, dataopt is {}".format(phase, dataset_opt))
        if phase == 'train' and args.phase != 'test':
            print("Creat [train] change-detection dataloader")
            train_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

        elif phase == 'val' and args.phase != 'test':
            print("Creat [val] change-detection dataloader")
            val_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            val_loader = Data.create_cd_dataloader(val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)

        # elif phase == 'test' and args.phase == 'test':
        elif phase == 'test':
            print("Creat [test] change-detection dataloader")
            test_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)

    logger.info('Initial Dataset Finished')

    #Create cd model
    cd_model = Model.create_CD_model(opt)
    
    # Initialize model weights to prevent NaN loss - more conservative
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight, gain=0.1)  # Very small gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.001)  # Very small std
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    cd_model.apply(init_weights)
    cd_model.to(device)
    logger.info(f'CD Model moved to device: {device}')
    
    # Verify model is actually on GPU
    if torch.cuda.is_available():
        model_device = next(cd_model.parameters()).device
        logger.info(f'Model parameters are on device: {model_device}')
        if model_device.type != 'cuda':
            logger.error('WARNING: Model parameters are NOT on GPU!')
        else:
            logger.info('✓ Model successfully moved to GPU')

    # Enable gradient checkpointing if available to save memory
    if hasattr(cd_model, 'gradient_checkpointing_enable'):
        cd_model.gradient_checkpointing_enable()

    num_classes = opt['model']['n_classes']
    logger.info(f"Number of classes for loss function: {num_classes}")

    #Create criterion (segmentation losses use semantic num_classes; change head will use 2)
    if opt['model']['loss'] == 'ce_dice':
        loss_fun = CEDiceLoss(num_classes=num_classes)
        loss_fun_change = CEDiceLoss(num_classes=2)
    elif opt['model']['loss'] == 'ce':
        # CrossEntropy can be used as a function or nn.Module. Using function for now.
        loss_fun = cross_entropy_loss_fn
        loss_fun_change = cross_entropy_loss_fn
    elif opt['model']['loss'] == 'dice':
        loss_fun = DiceOnlyLoss(num_classes=num_classes)
        loss_fun_change = DiceOnlyLoss(num_classes=2)
    elif opt['model']['loss'] == 'ce2_dice1':
        loss_fun = CE2Dice1Loss(num_classes=num_classes)
        loss_fun_change = CE2Dice1Loss(num_classes=2)
    elif opt['model']['loss'] == 'ce1_dice2':
        loss_fun = CE1Dice2Loss(num_classes=num_classes)
        loss_fun_change = CE1Dice2Loss(num_classes=2)
    # Add other loss types if needed, e.g., for 'ce_scl'
    elif opt['model']['loss'] == 'extended_triplet':
        # Extended multi-task loss: seg(t1)+seg(t2)+change + cross-time consistency + coupling
        base_seg = CEDiceLoss(num_classes=num_classes)
        cfg = opt['model'].get('extended_triplet', {})
        loss_fun = TripletChangeSegLoss(
            seg_loss_fn=base_seg,
            lambda_seg=cfg.get('lambda_seg', 1.0),
            lambda_cd=cfg.get('lambda_cd', 1.0),
            lambda_unch=cfg.get('lambda_unch', 0.2),
            lambda_ch=cfg.get('lambda_ch', 0.2),
            lambda_cpl=cfg.get('lambda_cpl', 0.5),
            T=cfg.get('T', 4.0),
            margin=cfg.get('margin', 0.3)
        )
    elif opt['model']['loss'] == 'multi_class_cd':
        loss_fun = MultiClassCDLoss(num_classes=num_classes, loss_weights=opt['model'].get('loss_weights'))
    # elif opt['model']['loss'] == 'ce_scl':
    #     loss_fun = CEDiceLoss(num_classes=num_classes) # Or a specific SCL loss class
    else:
        raise ValueError(f"Unsupported loss function type: {opt['model']['loss']}")

    # If losses are nn.Module, move them to the device
    if isinstance(loss_fun, nn.Module):
        loss_fun.to(device)
    if 'loss_fun_change' in locals() and isinstance(loss_fun_change, nn.Module):
        loss_fun_change.to(device)
    # Fallback: if loss_fun_change wasn't defined (e.g., for unsupported options), reuse loss_fun
    if 'loss_fun_change' not in locals():
        loss_fun_change = loss_fun

    #Create optimizer
    if opt['train']["optimizer"]["type"] == 'adam':
        optimer = optim.Adam(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'adamw':
        optimer = optim.AdamW(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'sgd':
        optimer = optim.SGD(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"],
                            momentum=0.9, weight_decay=5e-4)

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    metric = ConfuseMatrixMeter(n_class=2)  # For binary change detection (change/no-change)
    log_dict = OrderedDict()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # if you really want this
    # Remove any empty_cache()/synchronize() calls outside diagnostics

    #################
    # Training loop #
    #################
    if opt['phase'] == 'train':
        best_mF1 = 0.0
        epoch_losses = []
        for current_epoch in range(0, opt['train']['n_epoch']):
            print("......Begin Training......")
            metric.clear()
            cd_model.train()
            train_result_path = '{}/train/{}'.format(opt['path_cd']['result'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)

            #################
            #    Training   #
            #################
            message = 'lr: %0.7f\n \n' % optimer.param_groups[0]['lr']
            logger.info(message)

            epoch_loss = 0

            # Initial memory cleanup
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()

            # Reduce gradient accumulation for memory savings
            accumulation_steps = 2  # Effective batch size = 1 * 2 = 2
            
            # Set memory fraction to avoid fragmentation (more conservative)
            torch.cuda.set_per_process_memory_fraction(0.8)
            _max_train = getattr(args, 'max_train_batches', 0) or 0
            _train_total = min(len(train_loader), _max_train) if _max_train > 0 else len(train_loader)
            _train_iter = islice(train_loader, _max_train) if _max_train > 0 else train_loader
            for current_step, train_data in enumerate(tqdm(_train_iter, total=_train_total, desc=f"Train {current_epoch}/{opt['train']['n_epoch']}")):
                # Aggressive memory cleanup at start of each step
                # torch.cuda.empty_cache()
                # torch.cuda.synchronize()
                
                # Move data to GPU manually
                train_im1 = train_data['A'].to(device)
                train_im2 = train_data['B'].to(device)
                # Robust label extraction and move to device
                seg_t1 = (train_data['L1'] if 'L1' in train_data else train_data['L']).to(device)
                seg_t2 = (train_data['L2'] if 'L2' in train_data else train_data['L']).to(device)
                change = (train_data['change'] if 'change' in train_data else train_data['L']).to(device)

                # Use gradient checkpointing to save memory
                # Temporarily disable mixed precision to debug NaN
                outputs = cd_model(train_im1, train_im2)
                
                # Debug: Check for NaN in model outputs
                # for i, output in enumerate(outputs):
                #     if torch.isnan(output).any() or torch.isinf(output).any():
                #         logger.warning(f"NaN/Inf detected in model output {i}: nan={torch.isnan(output).sum()}, inf={torch.isinf(output).sum()}")
                #         logger.warning(f"Output {i} stats: min={output.min()}, max={output.max()}, mean={output.mean()}")
                
                # Clear input tensors from memory immediately after forward pass
                del train_im1, train_im2
                # torch.cuda.empty_cache()

                if opt['model']['loss'] == 'multi_class_cd':
                    # Debug: Check input data for NaN/Inf
                    # for name, tensor in [('seg_t1', seg_t1), ('seg_t2', seg_t2), ('change', change)]:
                    #     if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    #         logger.warning(f"NaN/Inf detected in {name}: nan={torch.isnan(tensor).sum()}, inf={torch.isinf(tensor).sum()}")
                    #         logger.warning(f"{name} stats: min={tensor.min()}, max={tensor.max()}, shape={tensor.shape}")
                    
                    labels = {'seg_t1': seg_t1, 'seg_t2': seg_t2, 'change': change}
                    # Temporarily disable mixed precision to debug NaN
                    train_loss, loss_dict = loss_fun(outputs, labels)
                    
                    # Debug: Check individual loss components
                    # logger.info(f"Step {current_step}: Loss components - seg_t1: {loss_dict['seg_t1']:.6f}, seg_t2: {loss_dict['seg_t2']:.6f}, change: {loss_dict['change']:.6f}, total: {train_loss.item():.6f}")
                    # Scale loss for gradient accumulation
                    train_loss = train_loss / accumulation_steps
                    seg_logits_t1, seg_logits_t2, change_pred = outputs
                    with torch.no_grad():
                        pred_seg_t1 = torch.argmax(seg_logits_t1, dim=1)
                        pred_seg_t2 = torch.argmax(seg_logits_t2, dim=1)
                        pred_change = torch.argmax(change_pred, dim=1)
                    
                    # Log input images to wandb (first batch of each epoch)
                    train_img1_np = train_data['A'][0].detach().cpu()
                    train_img2_np = train_data['B'][0].detach().cpu()
                    def norm_img(img):
                        img = img
                        if img.min() < 0:
                            img = (img + 1.0) / 2.0
                        img = (img * 255.0).clamp(0, 255).byte()
                        return img.permute(1,2,0).numpy() if img.ndim == 3 else img.numpy()
                    wandb.log({
                        "train/input_T1": [wandb.Image(norm_img(train_img1_np), caption="Train Input T1")],
                        "train/input_T2": [wandb.Image(norm_img(train_img2_np), caption="Train Input T2")],
                    }, commit=False)


                    # Log masks to wandb (log only for the first batch of each epoch to avoid excessive logging)
                    if current_step == 0 and current_epoch % 1 == 0:
                        # Handle ground truth masks - check if they're already RGB or need color mapping
                        seg_t1_np = seg_t1[0].detach().cpu().numpy()
                        seg_t2_np = seg_t2[0].detach().cpu().numpy()
                        
                        # If ground truth is already RGB (3 channels), scale it properly
                        if seg_t1_np.ndim == 3 and seg_t1_np.shape[2] == 3:
                            # Scale from 0-max_val to 0-255 for proper display
                            max_val = seg_t1_np.max()
                            if max_val > 0:
                                gt_seg_t1_img = ((seg_t1_np / max_val) * 255).astype(np.uint8)
                            else:
                                gt_seg_t1_img = seg_t1_np.astype(np.uint8)
                        else:
                            gt_seg_t1_img = create_color_mask(seg_t1[0], num_classes=opt['model']['n_classes'])
                        
                        if seg_t2_np.ndim == 3 and seg_t2_np.shape[2] == 3:
                            # Scale from 0-max_val to 0-255 for proper display
                            max_val = seg_t2_np.max()
                            if max_val > 0:
                                gt_seg_t2_img = ((seg_t2_np / max_val) * 255).astype(np.uint8)
                            else:
                                gt_seg_t2_img = seg_t2_np.astype(np.uint8)
                        else:
                            gt_seg_t2_img = create_color_mask(seg_t2[0], num_classes=opt['model']['n_classes'])
                        
                        # Also log probability maps for debugging
                        seg_t1_probs = torch.softmax(seg_logits_t1[0], dim=0)
                        seg_t2_probs = torch.softmax(seg_logits_t2[0], dim=0)
                        change_probs = torch.softmax(change_pred[0], dim=0)
                        
                        # Create probability visualizations (show max probability across classes)
                        seg_t1_max_prob = torch.max(seg_t1_probs, dim=0)[0].detach().cpu().numpy()
                        seg_t2_max_prob = torch.max(seg_t2_probs, dim=0)[0].detach().cpu().numpy()
                        change_max_prob = torch.max(change_probs, dim=0)[0].detach().cpu().numpy()
                        
                        # For change, visualize as 2-class (class-1 prob heatmap + argmax)
                        change_prob = change_probs[1].detach().cpu().numpy()
                        wandb.log({
                            "train/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=opt['model']['n_classes']), caption="Pred Seg T1 (multi-class)")],
                            "train/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=opt['model']['n_classes']), caption="Pred Seg T2 (multi-class)")],
                            "train/pred_change": [wandb.Image(create_color_mask(pred_change[0], num_classes=2), caption="Pred Change (binary)")],
                            "train/pred_seg_t1_prob": [wandb.Image(seg_t1_max_prob, caption="Pred Seg T1 Max Probability")],
                            "train/pred_seg_t2_prob": [wandb.Image(seg_t2_max_prob, caption="Pred Seg T2 Max Probability")],
                            "train/pred_change_prob": [wandb.Image(change_prob, caption="Pred Change Class-1 Probability")],
                            "train/gt_seg_t1": [wandb.Image(gt_seg_t1_img, caption="GT Seg T1")],
                            "train/gt_seg_t2": [wandb.Image(gt_seg_t2_img, caption="GT Seg T2")],
                            "train/gt_change": [wandb.Image(create_color_mask(change[0], num_classes=2), caption="GT Change (binary color)")],
                            "train/input_T1": [wandb.Image(norm_img(train_img1_np), caption="Train Input T1")],
                            "train/input_T2": [wandb.Image(norm_img(train_img2_np), caption="Train Input T2")],
                            "global_step": current_epoch * len(train_loader) + current_step
                        })


                elif opt['model']['loss'] == 'extended_triplet':
                    # Extended multi-task loss branch
                    seg_logits_t1, seg_logits_t2, change_pred = outputs
                    # TripletChangeSegLoss expects a single-channel change logit
                    u = change_pred if change_pred.shape[1] == 1 else change_pred[:, 1:2]
                    change_bin = normalize_change_target(seg_t1, seg_t2, change)
                    preds = (seg_logits_t1, seg_logits_t2, u)
                    labels = {'seg_t1': seg_t1, 'seg_t2': seg_t2, 'change': change_bin}
                    train_loss, ext_loss_dict = loss_fun(preds, labels)
                    train_loss = train_loss / accumulation_steps
                    # for compatibility with downstream logging keys
                    loss_dict = {'seg_t1': float('nan'), 'seg_t2': float('nan'), 'change': ext_loss_dict['cd']}
                    # Optional: log components
                    if current_step == 0 and current_epoch % 1 == 0:
                        wandb.log({
                            'train/extended/seg_total': ext_loss_dict['seg'],
                            'train/extended/unch_kl': ext_loss_dict['unch_kl'],
                            'train/extended/ch_div': ext_loss_dict['ch_div'],
                            'train/extended/couple': ext_loss_dict['couple'],
                        })

                else:
                    # Binary change detection branch (2-channel change head)
                    if isinstance(outputs, tuple) and len(outputs) >= 3:
                        seg_logits_t1, seg_logits_t2, change_pred = outputs
                    else:
                        change_pred = outputs
                        # Create dummy segmentation logits for logging with correct class dimension
                        b, _, h, w = change_pred.shape
                        seg_logits_t1 = torch.zeros((b, num_classes, h, w), device=change_pred.device, dtype=change_pred.dtype)
                        seg_logits_t2 = torch.zeros_like(seg_logits_t1)
                    # Create binary ground truth robustly: [B,H,W] long
                    change_bin = normalize_change_target(seg_t1, seg_t2, change)
                    # Compute loss against binary targets (use 2-class criterion)
                    train_loss = loss_fun_change(change_pred, change_bin)
                    # Scale loss for gradient accumulation
                    train_loss = train_loss / accumulation_steps
                    # Create a dummy loss_dict for logging consistency
                    loss_dict = {'seg_t1': 0, 'seg_t2': 0, 'change': train_loss.item()}
                
                # Convert logits to predicted masks for logging
                with torch.no_grad():
                    pred_seg_t1 = torch.argmax(seg_logits_t1, dim=1)
                    pred_seg_t2 = torch.argmax(seg_logits_t2, dim=1)
                    pred_change = torch.argmax(change_pred, dim=1)
                
                # Log masks to wandb (log only for the first batch of each epoch to avoid excessive logging)
                if current_step == 0 and current_epoch % 1 == 0:
                    # Debug: Check prediction values
                    print(f"\n=== TRAINING PREDICTIONS DEBUG (Epoch {current_epoch}) ===")
                    print(f"pred_seg_t1 shape: {pred_seg_t1.shape}, unique values: {torch.unique(pred_seg_t1[0])}")
                    print(f"pred_seg_t2 shape: {pred_seg_t2.shape}, unique values: {torch.unique(pred_seg_t2[0])}")
                    print(f"pred_change shape: {pred_change.shape}, unique values: {torch.unique(pred_change[0])}")
                    print(f"seg_logits_t1 shape: {seg_logits_t1.shape}, min: {seg_logits_t1.min():.4f}, max: {seg_logits_t1.max():.4f}")
                    print(f"seg_logits_t2 shape: {seg_logits_t2.shape}, min: {seg_logits_t2.min():.4f}, max: {seg_logits_t2.max():.4f}")
                    print(f"change_pred shape: {change_pred.shape}, min: {change_pred.min():.4f}, max: {change_pred.max():.4f}")
                    # Handle ground truth masks - check if they're already RGB or need color mapping
                    seg_t1_np = seg_t1[0].detach().cpu().numpy()
                    seg_t2_np = seg_t2[0].detach().cpu().numpy()
                    
                    # If ground truth is already RGB (3 channels), scale it properly
                    if seg_t1_np.ndim == 3 and seg_t1_np.shape[2] == 3:
                        # Scale from 0-max_val to 0-255 for proper display
                        max_val = seg_t1_np.max()
                        if max_val > 0:
                            gt_seg_t1_img = ((seg_t1_np / max_val) * 255).astype(np.uint8)
                        else:
                            gt_seg_t1_img = seg_t1_np.astype(np.uint8)
                    else:
                        gt_seg_t1_img = create_color_mask(seg_t1[0], num_classes=num_classes)
                    
                    if seg_t2_np.ndim == 3 and seg_t2_np.shape[2] == 3:
                        # Scale from 0-max_val to 0-255 for proper display
                        max_val = seg_t2_np.max()
                        if max_val > 0:
                            gt_seg_t2_img = ((seg_t2_np / max_val) * 255).astype(np.uint8)
                        else:
                            gt_seg_t2_img = seg_t2_np.astype(np.uint8)
                    else:
                        gt_seg_t2_img = create_color_mask(seg_t2[0], num_classes=num_classes)
                    
                    # Prepare binary GT change as a binary mask (0/1) for visualization
                    train_gt_change_bin = normalize_change_target(seg_t1, seg_t2, change)
                    # Ensure mask is [H, W] for visualization
                    if train_gt_change_bin.dim() == 3 and train_gt_change_bin.size(0) == 1:
                        train_gt_change_bin_vis = train_gt_change_bin.squeeze(0)
                    elif train_gt_change_bin.dim() == 3:
                        train_gt_change_bin_vis = train_gt_change_bin[0]
                    else:
                        train_gt_change_bin_vis = train_gt_change_bin
                    
                    # Debug what we have
                    print(f"\ntrain_gt_change_bin_vis shape: {train_gt_change_bin_vis.shape}, dtype: {train_gt_change_bin_vis.dtype}")
                    print(f"train_gt_change_bin_vis unique values: {torch.unique(train_gt_change_bin_vis)}")
                    
                    # Guaranteed binary mask creation
                    if train_gt_change_bin_vis.dtype == torch.float32:
                        # Convert floating point to binary 0/1
                        train_change_vis_binary = (train_gt_change_bin_vis > 0.5).int()
                    else:
                        # For int types, ensure only 0/1 values
                        train_change_vis_binary = (train_gt_change_bin_vis > 0).int()
                        
                    print(f"train_change_vis_binary unique values: {torch.unique(train_change_vis_binary)}")
                    
                    # Convert to numpy and ensure proper shape for colormapping
                    binary_mask_np = train_change_vis_binary.cpu().numpy()
                    print(f"binary_mask_np shape: {binary_mask_np.shape}, dtype: {binary_mask_np.dtype}, values: {np.unique(binary_mask_np)}")
                    
                    # Create high contrast colormap (bright red on black)
                    h, w = binary_mask_np.shape
                    train_gt_change_color = np.zeros((h, w, 3), dtype=np.uint8)
                    # Use bright red color [255,0,0] for all change pixels (binary_mask_np > 0)
                    mask = binary_mask_np > 0
                    train_gt_change_color[mask] = [255, 0, 0]
                    
                    # Log input images to wandb (first batch of each epoch)
                    train_img1_np = train_data['A'][0].detach().cpu()
                    train_img2_np = train_data['B'][0].detach().cpu()
                    
                    def norm_img(img):
                        if img.min() < 0:
                            img = (img + 1.0) / 2.0
                        img = (img * 255.0).clamp(0, 255).byte()
                        return img.permute(1,2,0).numpy() if img.ndim == 3 else img.numpy()
                    
                    wandb.log({
                        # Input images
                        "train/input_T1": [wandb.Image(norm_img(train_img1_np), caption="Train Input T1")],
                        "train/input_T2": [wandb.Image(norm_img(train_img2_np), caption="Train Input T2")],
                        # Predictions
                        "train/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=num_classes), caption="Pred Seg T1 (multi-class)")],
                        "train/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=num_classes), caption="Pred Seg T2 (multi-class)")],
                        "train/pred_change": [wandb.Image(create_color_mask(pred_change[0], num_classes=2), caption="Pred Change (binary)")],
                        # Ground truth
                        "train/gt_seg_t1": [wandb.Image(gt_seg_t1_img, caption="GT Seg T1")],
                        "train/gt_seg_t2": [wandb.Image(gt_seg_t2_img, caption="GT Seg T2")],
                        "train/gt_change": [wandb.Image(train_gt_change_color, caption="GT Change (binary color)")],
                        "global_step": current_epoch * len(train_loader) + current_step
                    })
                
                # Save change prediction for metrics before cleanup
                change_pred = outputs[2].detach()  # [B, 2, H, W]
                change_gt = (change > 0).long().detach()  # Binary ground truth
                
                # Check for NaN loss before backward pass
                if torch.isnan(train_loss) or torch.isinf(train_loss):
                    logger.warning(f"NaN/Inf loss detected at epoch {current_epoch}, step {current_step}. Skipping this batch.")
                    # logger.warning(f"Model parameter stats:")
                    # for name, param in cd_model.named_parameters():
                    #     if param.grad is not None:
                    #         logger.warning(f"  {name}: grad_norm={param.grad.norm():.6f}")
                    #     if torch.isnan(param).any():
                    #         logger.warning(f"  {name}: contains NaN parameters!")
                    optimer.zero_grad()
                    continue
                
                # Gradient accumulation without mixed precision for debugging
                train_loss.backward()
                if current_step == 0 and current_epoch == 0:
                    torch.cuda.synchronize(); import time; t0=time.time()
                    # do backward() here
                    torch.cuda.synchronize(); print("backward time:", time.time()-t0, "s")

                
                if (current_step + 1) % accumulation_steps == 0 or (current_step + 1) == len(train_loader):
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(cd_model.parameters(), max_norm=0.5)  # More aggressive clipping
                    
                    optimer.step()
                    optimer.zero_grad()
                    # Clear gradients from memory
                    # torch.cuda.empty_cache()
                    
                # Clean up memory after each batch (avoid double deletion)
                del seg_t1, seg_t2, change, outputs
                if 'pred_seg_t1' in locals():
                    del pred_seg_t1, pred_seg_t2, pred_change
                # torch.cuda.empty_cache()
                
                log_dict['loss'] = train_loss.item()
                log_dict['loss_seg_t1'] = loss_dict['seg_t1']
                log_dict['loss_seg_t2'] = loss_dict['seg_t2']
                log_dict['loss_change'] = loss_dict['change']
                epoch_loss += train_loss.item()

                # For metric, use argmax over 2-class change head
                G_pred = torch.argmax(change_pred, dim=1)
                binary_pred = G_pred.int()
                
                # Ground truth already binary (saved above)
                gt_np = change_gt.cpu().numpy().astype(np.uint8)
                pred_np = binary_pred.cpu().numpy()



                current_score = metric.update_cm(pr=pred_np, gt=gt_np)
                log_dict['running_acc'] = current_score.item()
                wandb.log({'train_loss': train_loss.item(), 'train_running_acc': current_score.item()})

                # Logging with GPU monitoring
                if current_step % opt['train']['train_print_iter'] == 0:
                    gpu_memory_info = ""
                    if torch.cuda.is_available():
                        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
                        gpu_memory_info = f", GPU Memory: {gpu_memory_allocated:.2f}GB/{gpu_memory_cached:.2f}GB"
                    
                    message = '[Training CD]. epoch: [%d/%d]. Itter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f%s\n' % (
                        current_epoch, opt['train']['n_epoch'], current_step, len(train_loader), train_loss.item(),
                        current_score.item(), gpu_memory_info)
                    logger.info(message)
                
                # Final cleanup of saved tensors
                del change_pred, change_gt, G_pred, binary_pred
                # torch.cuda.empty_cache()

            ### Epoch Summary ###
            scores = metric.get_scores()
            epoch_acc = scores['mf1']
            # Compute average epoch loss
            avg_epoch_loss = (epoch_loss / len(train_loader)) if len(train_loader) > 0 else 0.0
            
            # Log training epoch summary
            wandb.log({
                'train/epoch_mF1': epoch_acc,
                'train/epoch_loss': avg_epoch_loss,
                'train/epoch_mIoU': scores.get('mIoU', 0),
                'train/epoch_OA': scores.get('OA', 0),
                # flat keys as requested
                'train_epoch_mf1': epoch_acc,
                'train_epoch_loss': avg_epoch_loss,
                'epoch': current_epoch
            })
            
            ### VALIDATION LOOP ###
            logger.info('Starting validation...')
            val_metric = ConfuseMatrixMeter(n_class=2)
            cd_model.eval()
            val_loss_total = 0.0
            val_steps = 0
            shape_mismatch_logged = False  # Flag to log shape mismatch only once per epoch
            
            with torch.no_grad():
                _max_val = getattr(args, 'max_val_batches', 0) or 0
                _val_total = min(len(val_loader), _max_val) if _max_val > 0 else len(val_loader)
                _val_iter = islice(val_loader, _max_val) if _max_val > 0 else val_loader
                
                for val_step, val_data in enumerate(tqdm(_val_iter, total=_val_total, desc=f"Val {current_epoch}")):
                    val_img1 = val_data['A'].to(device)
                    val_img2 = val_data['B'].to(device)
                    
                    # Handle validation labels same as training
                    if 'L1' in val_data and 'L2' in val_data:
                        val_seg_t1 = val_data['L1'].to(device)
                        val_seg_t2 = val_data['L2'].to(device)
                    else:
                        val_seg_t1 = val_seg_t2 = val_data.get('L', torch.zeros_like(val_img1[:, :1])).to(device)
                    # Prefer provided change; otherwise derive from segs
                    if 'change' in val_data and val_data['change'] is not None:
                        val_change = val_data['change'].to(device)
                    else:
                        # Derive binary change mask from seg_t1 and seg_t2
                        val_change = normalize_change_target(val_seg_t1, val_seg_t2, None)
                    
                    # Forward pass
                    val_outputs = cd_model(val_img1, val_img2)
                    
                    if opt['model']['loss'] == 'multi_class_cd':
                        val_seg_logits_t1, val_seg_logits_t2, val_change_pred = val_outputs
                        # Pack targets into dictionary format expected by MultiClassCDLoss
                        val_targets = {
                            "seg_t1": val_seg_t1,
                            "seg_t2": val_seg_t2, 
                            "change": val_change
                        }
                        val_loss, val_loss_dict = loss_fun(val_outputs, val_targets)
                    elif opt['model']['loss'] == 'extended_triplet':
                        val_seg_logits_t1, val_seg_logits_t2, val_change_pred = val_outputs
                        
                        # Extract class-1 logits (change) for binary loss - needs [B,1,H,W] shape
                        if val_change_pred.shape[1] > 1:  # Model outputs [B,2,H,W]
                            u = val_change_pred[:, 1:2]  # Take only the positive class: [B,1,H,W]
                        else:
                            u = val_change_pred  # Already [B,1,H,W]
                            
                        # Ensure val_change (gt) has shape [B,1,H,W] for BCE loss
                        if val_change is not None:
                            if val_change.dim() == 3:  # [B,H,W]
                                val_change = val_change.unsqueeze(1)  # [B,1,H,W]
                            val_change = val_change.float()
                            
                        val_targets = {
                            "seg_t1": val_seg_t1,
                            "seg_t2": val_seg_t2,
                            "change": val_change
                        }
                        # Replace the change_pred part of val_outputs for loss calculation
                        val_outputs_adjusted = (val_seg_logits_t1, val_seg_logits_t2, u)
                        val_loss, val_loss_dict = loss_fun(val_outputs_adjusted, val_targets)
                    
                    val_loss_total += val_loss.item()
                    val_steps += 1
                    
                    # Update validation metrics
                    # Use argmax over 2-class change head directly
                    val_G_pred = torch.argmax(val_change_pred.detach(), dim=1)
                    val_binary_pred = val_G_pred.int()
                    
                    # Prepare ground truth change mask for both metrics and visualization
                    # Get the binary ground truth change mask
                    if val_change is not None:
                        # Ensure it's the right shape for visualization
                        if val_change.dim() == 4 and val_change.size(1) == 1:  # [B,1,H,W]
                            val_change_vis = val_change.squeeze(1)  # Convert to [B,H,W]
                        else:
                            val_change_vis = val_change
                        # Ensure it's binary and properly formatted
                        val_change_vis = val_change_vis.detach().cpu()  
                        print(f"\nval_change_vis shape: {val_change_vis.shape}, unique values: {torch.unique(val_change_vis[0])}")
                    else:
                        # Derive binary change mask from seg_t1 and seg_t2
                        val_change_vis = normalize_change_target(val_seg_t1, val_seg_t2, None)
                        val_change_vis = val_change_vis.detach().cpu()
                        print(f"\nDerived val_change_vis shape: {val_change_vis.shape}, unique values: {torch.unique(val_change_vis[0])}")
                    
                    # Ensure both arrays have the same shape for metric calculation
                    val_gt_np = val_change_vis.cpu().numpy().astype(np.uint8)
                    val_pred_np = val_binary_pred.cpu().numpy()
                    
                    # Handle potential shape mismatches
                    if val_gt_np.shape != val_pred_np.shape:
                        # If ground truth has extra dimensions, squeeze them
                        if val_gt_np.ndim > val_pred_np.ndim:
                            val_gt_np = val_gt_np.squeeze()
                        # If prediction has extra dimensions, squeeze them
                        elif val_pred_np.ndim > val_gt_np.ndim:
                            val_pred_np = val_pred_np.squeeze()
                        
                        # If still mismatched, resize to match using PyTorch interpolation
                        if val_gt_np.shape != val_pred_np.shape:
                            if not shape_mismatch_logged:
                                logger.info(f"Validation shape mismatch (expected): gt={val_gt_np.shape}, pred={val_pred_np.shape} - handling automatically")
                                shape_mismatch_logged = True
                            
                            # Handle different tensor formats
                            if val_gt_np.ndim == 4 and val_gt_np.shape[-1] == 3:  # NHWC format (channels last)
                                # Take first channel and remove channel dimension
                                val_gt_np = val_gt_np[..., 0]  # Shape: (N, H, W)
                            elif val_gt_np.ndim == 3 and val_pred_np.ndim == 3:
                                # Both are 3D, try to match shapes by interpolation
                                val_gt_tensor = torch.from_numpy(val_gt_np).float().unsqueeze(1)  # Add channel dim: (N, 1, H, W)
                                val_gt_resized = F.interpolate(val_gt_tensor, size=val_pred_np.shape[-2:], mode='nearest')
                                val_gt_np = val_gt_resized.squeeze(1).numpy().astype(np.uint8)  # Remove channel dim
                            
                            # Final shape check
                            if val_gt_np.shape != val_pred_np.shape:
                                logger.warning(f"Still mismatched after processing: gt={val_gt_np.shape}, pred={val_pred_np.shape}")
                                # As last resort, flatten both and take minimum length
                                min_size = min(val_gt_np.size, val_pred_np.size)
                                val_gt_np = val_gt_np.flatten()[:min_size].reshape(-1)
                                val_pred_np = val_pred_np.flatten()[:min_size].reshape(-1)
                    
                    # Update confusion matrix and get running mF1
                    val_running_acc = val_metric.update_cm(pr=val_pred_np, gt=val_gt_np)
                    
                    # Per-step validation logging
                    wandb.log({'val_loss': val_loss.item(), 'val_running_acc': val_running_acc.item()})
            
                    # Log validation visualizations for first batch of each epoch
                    if val_step == 0 and current_epoch % 1 == 0:
                        # Log input images for val (first batch only)
                        val_img1_np = val_img1[0].detach().cpu()
                        val_img2_np = val_img2[0].detach().cpu()
                        def norm_img(img):
                            img = img
                            if img.min() < 0:
                                img = (img + 1.0) / 2.0
                            img = (img * 255.0).clamp(0, 255).byte()
                            return img.permute(1,2,0).numpy() if img.ndim == 3 else img.numpy()
                        val_img1_np = val_data['A'][0].detach().cpu()
                        val_img2_np = val_data['B'][0].detach().cpu()
                        wandb.log({
                            "val/input_T1": [wandb.Image(norm_img(val_img1_np), caption="Val Input T1")],
                            "val/input_T2": [wandb.Image(norm_img(val_img2_np), caption="Val Input T2")],
                        }, commit=False)


                        def norm_img(img):
                            img = img
                            if img.min() < 0:
                                img = (img + 1.0) / 2.0
                            img = (img * 255.0).clamp(0, 255).byte()
                            return img.permute(1,2,0).numpy() if img.ndim == 3 else img.numpy()
                        wandb.log({
                            "val/input_T1": [wandb.Image(norm_img(val_img1_np), caption="Val Input T1")],
                            "val/input_T2": [wandb.Image(norm_img(val_img2_np), caption="Val Input T2")],
                        }, commit=False)


                        with torch.no_grad():
                            val_pred_seg_t1 = torch.argmax(val_seg_logits_t1, dim=1)
                            val_pred_seg_t2 = torch.argmax(val_seg_logits_t2, dim=1)
                            val_pred_change = torch.argmax(val_change_pred, dim=1)
                        
                        # Debug: Check validation prediction values
                        print(f"\n=== VALIDATION PREDICTIONS DEBUG (Epoch {current_epoch}) ===")
                        print(f"val_pred_seg_t1 shape: {val_pred_seg_t1.shape}, unique values: {torch.unique(val_pred_seg_t1[0])}")
                        print(f"val_pred_seg_t2 shape: {val_pred_seg_t2.shape}, unique values: {torch.unique(val_pred_seg_t2[0])}")
                        print(f"val_pred_change shape: {val_pred_change.shape}, unique values: {torch.unique(val_pred_change[0])}")
                        print(f"val_seg_logits_t1 shape: {val_seg_logits_t1.shape}, min: {val_seg_logits_t1.min():.4f}, max: {val_seg_logits_t1.max():.4f}")
                        print(f"val_seg_logits_t2 shape: {val_seg_logits_t2.shape}, min: {val_seg_logits_t2.min():.4f}, max: {val_seg_logits_t2.max():.4f}")
                        print(f"val_change_pred shape: {val_change_pred.shape}, min: {val_change_pred.min():.4f}, max: {val_change_pred.max():.4f}")
                        
                        # Handle ground truth masks same as training
                        val_seg_t1_np = val_seg_t1[0].detach().cpu().numpy()
                        val_seg_t2_np = val_seg_t2[0].detach().cpu().numpy()
                        
                        if val_seg_t1_np.ndim == 3 and val_seg_t1_np.shape[2] == 3:
                            max_val = val_seg_t1_np.max()
                            if max_val > 0:
                                val_gt_seg_t1_img = ((val_seg_t1_np / max_val) * 255).astype(np.uint8)
                            else:
                                val_gt_seg_t1_img = val_seg_t1_np.astype(np.uint8)
                        else:
                            val_gt_seg_t1_img = create_color_mask(val_seg_t1[0], num_classes=opt['model']['n_classes'])
                        
                        if val_seg_t2_np.ndim == 3 and val_seg_t2_np.shape[2] == 3:
                            max_val = val_seg_t2_np.max()
                            if max_val > 0:
                                val_gt_seg_t2_img = ((val_seg_t2_np / max_val) * 255).astype(np.uint8)
                            else:
                                val_gt_seg_t2_img = val_seg_t2_np.astype(np.uint8)
                        else:
                            val_gt_seg_t2_img = create_color_mask(val_seg_t2[0], num_classes=opt['model']['n_classes'])
                        
                        # Create binary ground truth change visualization
                        # Note: val_change_vis is already prepared earlier for metrics
                        
                        # Create color visualization with enhanced contrast for binary mask
                        # First, debug what we have
                        print(f"\nval_change_vis[0] unique values BEFORE: {torch.unique(val_change_vis[0])}")
                        
                        # Force to pure binary if needed
                        if val_change_vis[0].dtype == torch.float32:
                            # Convert to binary 0/1 if it's floating point
                            val_change_vis_binary = (val_change_vis[0] > 0.5).int()
                        else:
                            val_change_vis_binary = val_change_vis[0]
                            
                        print(f"val_change_vis_binary unique values AFTER: {torch.unique(val_change_vis_binary)}")
                        
                        # Create custom binary colormap for better visibility
                        # Black (0) for no change, bright red (1) for change
                        binary_mask_np = val_change_vis_binary.cpu().numpy()
                        h, w = binary_mask_np.shape
                        val_gt_change_color = np.zeros((h, w, 3), dtype=np.uint8)
                        val_gt_change_color[binary_mask_np == 1] = [255, 0, 0]  # Bright red for changes
                        
                        # Also log probability maps for validation debugging
                        val_seg_t1_probs = torch.softmax(val_seg_logits_t1[0], dim=0)
                        val_seg_t2_probs = torch.softmax(val_seg_logits_t2[0], dim=0)
                        val_change_probs = torch.softmax(val_change_pred[0], dim=0)
                        
                        # Create probability visualizations (show max probability across classes)
                        val_seg_t1_max_prob = torch.max(val_seg_t1_probs, dim=0)[0].detach().cpu().numpy()
                        val_seg_t2_max_prob = torch.max(val_seg_t2_probs, dim=0)[0].detach().cpu().numpy()
                        val_change_prob = val_change_probs[1].detach().cpu().numpy()
                        
                        # Prepare validation input images for logging
                        val_img1_np = val_img1[0].detach().cpu()
                        val_img2_np = val_img2[0].detach().cpu()
                        
                        def norm_img(img):
                            img = img
                            if img.min() < 0:
                                img = (img + 1.0) / 2.0
                            img = (img * 255.0).clamp(0, 255).byte()
                            return img.permute(1,2,0).numpy() if img.ndim == 3 else img.numpy()
                            
                        wandb.log({
                            # Input images
                            "val/input_T1": [wandb.Image(norm_img(val_img1_np), caption="Val Input T1")],
                            "val/input_T2": [wandb.Image(norm_img(val_img2_np), caption="Val Input T2")],
                            # Predictions
                            "val/pred_seg_t1": [wandb.Image(create_color_mask(val_pred_seg_t1[0], num_classes=opt['model']['n_classes']), caption="Val Pred Seg T1 (multi-class)")],
                            "val/pred_seg_t2": [wandb.Image(create_color_mask(val_pred_seg_t2[0], num_classes=opt['model']['n_classes']), caption="Val Pred Seg T2 (multi-class)")],
                            "val/pred_change": [wandb.Image(create_color_mask(val_pred_change[0], num_classes=2), caption="Val Pred Change (binary)")],
                            "val/pred_seg_t1_prob": [wandb.Image(val_seg_t1_max_prob, caption="Val Pred Seg T1 Max Probability")],
                            "val/pred_seg_t2_prob": [wandb.Image(val_seg_t2_max_prob, caption="Val Pred Seg T2 Max Probability")],
                            "val/pred_change_prob": [wandb.Image(val_change_prob, caption="Val Pred Change Class-1 Probability")],
                            # Ground truth
                            "val/gt_seg_t1": [wandb.Image(val_gt_seg_t1_img, caption="Val GT Seg T1")],
                            "val/gt_seg_t2": [wandb.Image(val_gt_seg_t2_img, caption="Val GT Seg T2")],
                            "val/gt_change": [wandb.Image(val_gt_change_color, caption="Val GT Change (binary color)")],
                            "global_step": current_epoch * len(train_loader) + len(train_loader)
                        })
                    
                    # Clean up validation tensors
                    del val_outputs, val_change_pred, val_G_pred, val_binary_pred
                    # torch.cuda.empty_cache()
            
            # Log validation epoch summary
            val_scores = val_metric.get_scores()
            val_epoch_acc = val_scores['mf1']
            avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0.0
            
            wandb.log({
                'val/epoch_loss': avg_val_loss,
                'val/epoch_mF1': val_epoch_acc,
                'val/epoch_mIoU': val_scores.get('mIoU', 0),
                'val/epoch_OA': val_scores.get('OA', 0),
                'epoch': current_epoch
            })
            
            logger.info(f'Validation - Epoch: {current_epoch}, Loss: {avg_val_loss:.5f}, mF1: {val_epoch_acc:.5f}')
            
            # Reset model to training mode
            cd_model.train()
            
            # Load the best model for testing
            gen_path = os.path.join(opt['path_cd']['checkpoint'], 'best_net.pth')
            if os.path.exists(gen_path):
                cd_model.load_state_dict(torch.load(gen_path), strict=True)
                logger.info(f'Loaded best model from {gen_path}')
            else:
                logger.warning(f'Best model not found at {gen_path}, using current model')
            cd_model.to(device)
            metric.clear()
            cd_model.eval()
            
            # Create test result directory
            test_result_path = '{}/test'.format(opt['path_cd']['result'])
            os.makedirs(test_result_path, exist_ok=True)
            
            with torch.no_grad():
                # Apply optional cap on test batches
                _max_test = getattr(args, 'max_test_batches', 0) or 0
                _test_total = min(len(test_loader), _max_test) if _max_test > 0 else len(test_loader)
                _test_iter = islice(test_loader, _max_test) if _max_test > 0 else test_loader
                for current_step, test_data in enumerate(tqdm(_test_iter, total=_test_total, desc="Test")):
                    test_img1 = test_data['A'].to(device)
                    test_img2 = test_data['B'].to(device)
                    # Robust label extraction - data automatically on correct device
                    if 'L1' in test_data and 'L2' in test_data:
                        seg_t1 = test_data['L1']
                        seg_t2 = test_data['L2']
                    else:
                        # Fallback for older single-label format
                        seg_t1 = seg_t2 = test_data.get('L')

                    # Obtain change mask if provided; otherwise derive binary mask from seg labels
                    change = test_data['change'] if 'change' in test_data else None

                    outputs = cd_model(test_img1, test_img2)
                    # Extract all heads
                    seg_logits_t1, seg_logits_t2, change_pred = outputs
                    # Only use change head for metric and visuals (2-class)
                    # Convert prediction to binary change mask directly
                    G_pred = torch.argmax(change_pred.detach(), dim=1)
                    # Normalize GT to binary [B,H,W]
                    test_change_bin = normalize_change_target(seg_t1, seg_t2, change)

                    # Prepare numpy arrays for metrics
                    pred_np = G_pred.int().cpu().numpy()
                    gt_np = test_change_bin.cpu().numpy().astype(np.uint8)

                    # Optional: log first batch of test predictions (segmentations + probs)
                    if current_step == 0:
                        # Log input images for test (first batch only)
                        test_img1_np = test_img1[0].detach().cpu()
                        test_img2_np = test_img2[0].detach().cpu()
                        
                        def norm_img(img):
                            img = img
                            if img.min() < 0:
                                img = (img + 1.0) / 2.0
                            img = (img * 255.0).clamp(0, 255).byte()
                            return img.permute(1,2,0).numpy() if img.ndim == 3 else img.numpy()
                        wandb.log({
                            "test/input_T1": [wandb.Image(norm_img(test_img1_np), caption="Test Input T1")],
                            "test/input_T2": [wandb.Image(norm_img(test_img2_np), caption="Test Input T2")],
                        }, commit=False)


                        # Change probabilities (class-1 probability)
                        change_probs = torch.softmax(change_pred[0], dim=0)
                        change_prob = change_probs[1].detach().cpu().numpy()

                        # Segmentation predictions and per-pixel confidence (max prob)
                        pred_seg_t1 = torch.argmax(seg_logits_t1, dim=1)
                        pred_seg_t2 = torch.argmax(seg_logits_t2, dim=1)
                        seg_t1_probs = torch.softmax(seg_logits_t1[0], dim=0)  # [C,H,W]
                        seg_t2_probs = torch.softmax(seg_logits_t2[0], dim=0)
                        seg_t1_max_prob = torch.max(seg_t1_probs, dim=0).values.detach().cpu().numpy()  # [H,W]
                        seg_t2_max_prob = torch.max(seg_t2_probs, dim=0).values.detach().cpu().numpy()
                        test_gt_change_bw = ((test_change_bin[0] > 0).float().cpu().numpy() * 255).astype(np.uint8)
                        wandb.log({
                            # Multi-class segmentations (colorized)
                            "test/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=opt['model']['n_classes']), caption="Test Pred Seg T1 (multi-class)")],
                            "test/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=opt['model']['n_classes']), caption="Test Pred Seg T2 (multi-class)")],
                            # Confidence maps
                            "test/pred_seg_t1_prob": [wandb.Image(seg_t1_max_prob, caption="Test Pred Seg T1 Max Probability")],
                            "test/pred_seg_t2_prob": [wandb.Image(seg_t2_max_prob, caption="Test Pred Seg T2 Max Probability")],
                            "test/pred_change": [wandb.Image(create_color_mask(G_pred[0], num_classes=2), caption="Test Pred Change (binary)")],
                            "test/pred_change_prob": [wandb.Image(change_prob, caption="Test Pred Change Class-1 Probability")],
                            "test/gt_change": [wandb.Image(create_color_mask(test_change_bin[0], num_classes=2), caption="Test GT Change (binary color)")],
                        })
                    binary_pred = G_pred.int()
                    
                    # Get ground truth
                    if 'change' in test_data:
                        gt = (test_data['change'] > 0).long().to(device)
                    elif change is not None:
                        gt = change.to(device)
                    else:
                        gt = (seg_t1 != seg_t2).long().to(device)
                    
                    # Create binary ground truth for visualization
                    gt_binary = (gt > 0).int()  # Convert to binary (0 or 1)

                    # Visuals
                    out_dict = OrderedDict()
                    out_dict['pred_cm'] = binary_pred  # Use binary prediction for visualization
                    out_dict['gt_cm'] = gt_binary  # Use binary ground truth for visualization
                    visuals = out_dict

                    img_mode = 'single'
                    if img_mode == 'single':
                        # Converting to uint8
                        visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                        visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                        img_A = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_B = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        # Handle tensor dimensions properly for visualization
                        gt_tensor = visuals['gt_cm']
                        pred_tensor = visuals['pred_cm']
                        
                        # Ensure tensors are in correct format (B, H, W) before adding channel dimension
                        if gt_tensor.dim() > 3:
                            gt_tensor = gt_tensor.squeeze()  # Remove extra dimensions
                        if pred_tensor.dim() > 3:
                            pred_tensor = pred_tensor.squeeze()  # Remove extra dimensions
                            
                        # Add channel dimension and repeat for RGB
                        if gt_tensor.dim() == 3:  # (B, H, W)
                            gt_tensor = gt_tensor.unsqueeze(1)  # (B, 1, H, W)
                        elif gt_tensor.dim() == 2:  # (H, W)
                            gt_tensor = gt_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                            
                        if pred_tensor.dim() == 3:  # (B, H, W)
                            pred_tensor = pred_tensor.unsqueeze(1)  # (B, 1, H, W)
                        elif pred_tensor.dim() == 2:  # (H, W)
                            pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                        
                        gt_cm = Metrics.tensor2img(gt_tensor.repeat(1, 3, 1, 1), out_type=np.uint8,
                                                   min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(pred_tensor.repeat(1, 3, 1, 1),
                                                     out_type=np.uint8, min_max=(0, 1))  # uint8

                        # Save imgs
                        Metrics.save_img(
                            img_A, '{}/img_A_{}.png'.format(test_result_path, current_step))
                        Metrics.save_img(
                            img_B, '{}/img_B_{}.png'.format(test_result_path, current_step))
                        Metrics.save_img(
                            pred_cm, '{}/img_pred_cm{}.png'.format(test_result_path, current_step))
                        Metrics.save_img(
                            gt_cm, '{}/img_gt_cm{}.png'.format(test_result_path, current_step))

                        # Log input images to wandb for test/validation
                        def norm_img(img):
                            img = img
                            if img.min() < 0:
                                img = (img + 1.0) / 2.0
                            img = (img * 255.0).clip(0, 255).astype(np.uint8)
                            if img.ndim == 3:
                                return img.transpose(1,2,0)
                            return img
                        if 'A' in test_data and 'B' in test_data:
                            img_A_np = test_data['A'][0].detach().cpu().numpy()
                            img_B_np = test_data['B'][0].detach().cpu().numpy()
                            wandb.log({
                                "test/input_T1": [wandb.Image(norm_img(img_A_np), caption="Test Input T1")],
                                "test/input_T2": [wandb.Image(norm_img(img_B_np), caption="Test Input T2")],
                            }, commit=False)

                    else:
                        # grid img
                        visuals['pred_cm'] = visuals['pred_cm'] * 2.0 - 1.0
                        visuals['gt_cm'] = visuals['gt_cm'] * 2.0 - 1.0
                        grid_img = torch.cat((test_data['A'],
                                              test_data['B'],
                                              visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
                                              visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                             dim=0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_{}.png'.format(test_result_path, current_step))

                        # Log input images to wandb for test/validation (grid case)
                        if 'A' in test_data and 'B' in test_data:
                            img_A_np = test_data['A'][0].detach().cpu().numpy()
                            img_B_np = test_data['B'][0].detach().cpu().numpy()
                            wandb.log({
                                "test/input_T1": [wandb.Image(norm_img(img_A_np), caption="Test Input T1")],
                                "test/input_T2": [wandb.Image(norm_img(img_B_np), caption="Test Input T2")],
                            }, commit=False)


                ### log epoch status ###
                scores = metric.get_scores()
                epoch_acc = scores['mf1']
                log_dict['epoch_acc'] = epoch_acc.item()
                for k, v in scores.items():
                    log_dict[k] = v
                logs = log_dict
                message = '[Test CD summary]: Test mF1=%.5f \n' % \
                          (logs['epoch_acc'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    message += '\n'
                logger.info(message)
                logger.info('End of testing...')

            # Note: Validation logging is now handled in the validation loop above
