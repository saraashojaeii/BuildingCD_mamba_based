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
from core.utils import *
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

if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/saraashojaeii/git/BuildingCD_mamba_based/config/second_cdmamba/second_cdmamba.json',
                        help='JSON file for configuration')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'], help='Run either train(training + validation) or testing',)
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
    parser.add_argument('--change_threshold', type=float, default=0.2, help='Probability threshold for change class (class-1) binarization')

    # Parse config
    args = parser.parse_args()
    opt = Logger.parse(args)

    #Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # Create a unique timestamped experiment subfolder for logs/results/checkpoints
    exp_timestamp = datetime.now().strftime('%m%d_%H')
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
        exp_folder = f"{suffix}_{exp_timestamp}"
    else:
        exp_folder = f"{exp_timestamp}"
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
    
    metric = ConfuseMatrixMeter(n_class=2)  # For binary change detection
    metric_seg = ConfuseMatrixMeter(n_class=opt['model']['n_classes'])  # For 6-class segmentation
    log_dict = OrderedDict()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # if you really want this

    #################
    # Training loop #
    #################
    if opt['phase'] == 'train':
        best_mF1 = 0.0
        epoch_losses = []
        for current_epoch in range(0, opt['train']['n_epoch']):
            print("......Begin Training......")
            metric.clear()  # Clear binary change metrics
            metric_seg.clear()  # Clear segmentation metrics
            cd_model.train()
            train_result_path = '{}/train/{}'.format(opt['path_cd']['result'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)

            #################
            #    Training   #
            #################
            message = 'lr: %0.7f\n \n' % optimer.param_groups[0]['lr']
            logger.info(message)

            epoch_loss = 0

            # Reduce gradient accumulation for memory savings
            accumulation_steps = 2  # Effective batch size = 1 * 2 = 2
            
            # Set memory fraction to avoid fragmentation (more conservative)
            torch.cuda.set_per_process_memory_fraction(0.8)
            _max_train = getattr(args, 'max_train_batches', 0) or 0
            _train_total = min(len(train_loader), _max_train) if _max_train > 0 else len(train_loader)
            _train_iter = islice(train_loader, _max_train) if _max_train > 0 else train_loader
            for current_step, train_data in enumerate(tqdm(_train_iter, total=_train_total, desc=f"Train {current_epoch}/{opt['train']['n_epoch']}")):
                
                # Move data to GPU manually
                train_im1 = train_data['A'].to(device)
                train_im2 = train_data['B'].to(device)

                # Use gradient checkpointing to save memory
                outputs = cd_model(train_im1, train_im2)
                if current_step == 0:
                    import hashlib
                    def _fp(t: torch.Tensor):
                        t_cpu = t.detach().to('cpu')
                        return hashlib.sha1(t_cpu.numpy().tobytes()).hexdigest()[:12]
                    try:
                        logger.info(f"[TRAIN fp] A[0]={_fp(train_data['A'][0])}, B[0]={_fp(train_data['B'][0])}")
                    except Exception:
                        pass
                
                # Clear input tensors from memory immediately after forward pass
                del train_im1, train_im2

                # Fetch ground truth labels from batch and move to device
                seg_t1 = train_data.get('L1', None)
                seg_t2 = train_data.get('L2', None)
                change = train_data.get('L', None)
                if (seg_t1 is None) or (seg_t2 is None):
                    # Fallback for datasets without L1/L2: reuse 'L' for both or zeros if absent
                    if change is not None:
                        seg_t1 = change
                        seg_t2 = change
                    else:
                        # Create dummy zero labels matching change_pred spatial size
                        if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
                            b, _, h, w = outputs[2].shape
                        else:
                            b, _, h, w = outputs.shape
                        seg_t1 = torch.zeros((b, h, w), dtype=torch.long)
                        seg_t2 = torch.zeros((b, h, w), dtype=torch.long)
                # Ensure dtype/device
                if isinstance(seg_t1, torch.Tensor):
                    seg_t1 = seg_t1.to(device).long()
                if isinstance(seg_t2, torch.Tensor):
                    seg_t2 = seg_t2.to(device).long()
                if isinstance(change, torch.Tensor):
                    change = change.to(device).long()

                if opt['model']['loss'] == 'extended_triplet':
                    # Extended multi-task loss branch
                    seg_logits_t1, seg_logits_t2, change_pred = outputs
                    # TripletChangeSegLoss expects a single-channel change logit
                    u = change_pred if change_pred.shape[1] == 1 else change_pred[:, 1:2]
                    change_bin = normalize_change_target(seg_t1, seg_t2, change)
                    preds = (seg_logits_t1, seg_logits_t2, u)
                    labels = {'seg_t1': seg_t1, 'seg_t2': seg_t2, 'change': change_bin}
                    if current_step == 0:
                        # Show post-normalization change target dtype
                        logger.info(f"[TRAIN dtype-check] change_bin: shape={tuple(change_bin.shape)}, dtype={change_bin.dtype}, device={change_bin.device}")
                        try:
                            _derived = normalize_change_target(seg_t1, seg_t2, None)
                            mism = (_derived != change_bin).float().mean().item()
                            logger.info(f"[TRAIN consistency] derived_vs_change_bin_mismatch={mism:.6f}")
                        except Exception as e:
                            logger.warning(f"[TRAIN consistency] could not compare derived vs change_bin: {e}")
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
                    if current_step == 0:
                        logger.info(f"[TRAIN dtype-check] change_bin: shape={tuple(change_bin.shape)}, dtype={change_bin.dtype}, device={change_bin.device}")
                        try:
                            _derived = normalize_change_target(seg_t1, seg_t2, None)
                            mism = (_derived != change_bin).float().mean().item()
                            logger.info(f"[TRAIN consistency] derived_vs_change_bin_mismatch={mism:.6f}")
                        except Exception as e:
                            logger.warning(f"[TRAIN consistency] could not compare derived vs change_bin: {e}")
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
                    change_p1 = torch.softmax(change_pred, dim=1)[:, 1, :, :]
                    pred_change = (change_p1 > args.change_threshold).long()
                
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
                    
                    # Prepare ground truth change mask for visualization
                    # Use the already loaded change mask directly
                    if change is not None:
                        # Ensure it's the right shape for visualization
                        if change.dim() == 4 and change.size(1) == 1:  # [B,1,H,W]
                            train_change_vis = change[0].squeeze(0)  # Convert to [H,W]
                        elif change.dim() == 3:  # [B,H,W]
                            train_change_vis = change[0]
                        elif change.dim() == 2:  # [H,W]
                            train_change_vis = change
                        else:
                            train_change_vis = change
                        
                        # Convert to CPU and ensure binary
                        train_change_vis = train_change_vis.detach().cpu()
                        print(f"\ntrain_change_vis shape: {train_change_vis.shape}, dtype: {train_change_vis.dtype}")
                        print(f"train_change_vis unique values: {torch.unique(train_change_vis)}")
                    else:
                        # This shouldn't happen now since we derive it above
                        train_change_vis = torch.zeros_like(seg_t1[0])
                    
                    # Force to pure binary if needed
                    if train_change_vis.dtype == torch.float32:
                        # Convert to binary 0/1 if it's floating point
                        train_change_vis_binary = (train_change_vis > 0.5).int()
                    else:
                        # For int types, ensure only 0/1 values  
                        train_change_vis_binary = (train_change_vis > 0).int()
                        
                    print(f"train_change_vis_binary unique values: {torch.unique(train_change_vis_binary)}")
                    
                    # Create custom binary colormap for better visibility
                    # Black (0) for no change, bright red (1) for change
                    binary_mask_np = train_change_vis_binary.cpu().numpy()
                    h, w = binary_mask_np.shape
                    train_gt_change_color = np.zeros((h, w, 3), dtype=np.uint8)
                    train_gt_change_color[binary_mask_np == 1] = [255, 0, 0]  # Bright red for changes
                    
                    # Log input images to wandb (first batch of each epoch)
                    train_img1_np = train_data['A'][0].detach().cpu()
                    train_img2_np = train_data['B'][0].detach().cpu()
                    
                    def norm_img(img):
                        if img.min() < 0:
                            img = (img + 1.0) / 2.0
                        img = (img * 255.0).clamp(0, 255).byte()
                        return img.permute(1,2,0).numpy() if img.ndim == 3 else img.numpy()
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
                    # Ensure GT change logged equals the normalized target used in loss
                    gt_change_for_log = normalize_change_target(seg_t1, seg_t2, change)

                    wandb.log({
                            "train/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=opt['model']['n_classes']), caption="Pred Seg T1 (multi-class)")],
                            "train/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=opt['model']['n_classes']), caption="Pred Seg T2 (multi-class)")],
                            "train/pred_change": [wandb.Image(create_color_mask(pred_change[0], num_classes=2), caption="Pred Change (binary)")],
                            "train/pred_seg_t1_prob": [wandb.Image(seg_t1_max_prob, caption="Pred Seg T1 Max Probability")],
                            "train/pred_seg_t2_prob": [wandb.Image(seg_t2_max_prob, caption="Pred Seg T2 Max Probability")],
                            "train/pred_change_prob": [wandb.Image(change_prob, caption="Pred Change Class-1 Probability")],
                            "train/gt_seg_t1": [wandb.Image(gt_seg_t1_img, caption="GT Seg T1")],
                            "train/gt_seg_t2": [wandb.Image(gt_seg_t2_img, caption="GT Seg T2")],
                            "train/gt_change": [wandb.Image(create_color_mask(gt_change_for_log[0], num_classes=2), caption="GT Change (binary color)")],
                            "train/input_T1": [wandb.Image(norm_img(train_img1_np), caption="Train Input T1")],
                            "train/input_T2": [wandb.Image(norm_img(train_img2_np), caption="Train Input T2")],
                            "global_step": current_epoch * len(train_loader) + current_step
                        })

                    for c in range(opt['model']['n_classes']):
                        wandb.log({f"train/seg_t2_prob_c{c}": [wandb.Image(seg_t2_probs[c].detach().cpu().numpy())]}, commit=False)
                
                # Save change prediction for metrics before cleanup
                change_pred_for_metric = change_pred.detach()  # [B, 2, H, W]
                change_gt = (change_bin > 0).long().detach()  # Binary ground truth
                
                # Check for NaN loss before backward pass
                if torch.isnan(train_loss) or torch.isinf(train_loss):
                    logger.warning(f"NaN/Inf loss detected at epoch {current_epoch}, step {current_step}. Skipping this batch.")
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
                    
                # Clean up memory after each batch (avoid double deletion)
                del seg_t1, seg_t2, change, outputs
                if 'pred_seg_t1' in locals():
                    del pred_seg_t1, pred_seg_t2, pred_change
                
                log_dict['loss'] = train_loss.item()
                log_dict['loss_seg_t1'] = loss_dict['seg_t1']
                log_dict['loss_seg_t2'] = loss_dict['seg_t2']
                log_dict['loss_change'] = loss_dict['change']
                epoch_loss += train_loss.item()

                # For metric, threshold class-1 probability over 2-class change head
                change_p1 = torch.softmax(change_pred_for_metric, dim=1)[:, 1, :, :]
                G_pred = (change_p1 > args.change_threshold).long()
                print("################################################")
                print(f"G_pred unique values: {torch.unique(G_pred[0])}")
                binary_pred = G_pred.int()
                
                # Ground truth already binary (saved above)
                gt_np = change_gt.cpu().numpy().astype(np.uint8)
                pred_np = binary_pred.cpu().numpy()

                current_score = metric.update_cm(pr=pred_np, gt=gt_np)
                log_dict['running_acc'] = current_score.item()
                
                # Update segmentation metrics (6-class)
                pred_seg_t1_np = pred_seg_t1.detach().cpu().numpy().astype(np.uint8)
                pred_seg_t2_np = pred_seg_t2.detach().cpu().numpy().astype(np.uint8)
                gt_seg_t1_np = seg_t1.detach().cpu().numpy().astype(np.uint8)
                gt_seg_t2_np = seg_t2.detach().cpu().numpy().astype(np.uint8)
                
                # Update metrics for both T1 and T2 segmentation heads
                seg_score_t1 = metric_seg.update_cm(pr=pred_seg_t1_np, gt=gt_seg_t1_np)
                seg_score_t2 = metric_seg.update_cm(pr=pred_seg_t2_np, gt=gt_seg_t2_np)
                seg_score_avg = (seg_score_t1 + seg_score_t2) / 2.0
                
                wandb.log({
                    'train_loss': train_loss.item(), 
                    'train_running_acc': current_score.item(),
                    'train_running_seg_mf1': seg_score_avg.item()
                })

                # Logging with GPU monitoring
                if current_step % opt['train']['train_print_iter'] == 0:
                    gpu_memory_info = ""
                    if torch.cuda.is_available():
                        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3
                        gpu_memory_info = f", GPU Memory: {gpu_memory_allocated:.2f}GB/{gpu_memory_cached:.2f}GB"
                    
                    message = '[Training CD]. epoch: [%d/%d]. Itter: [%d/%d], CD_loss: %.5f, change_mf1: %.5f, seg_mf1: %.5f%s\n' % (
                        current_epoch, opt['train']['n_epoch'], current_step, len(train_loader), train_loss.item(),
                        current_score.item(), seg_score_avg.item(), gpu_memory_info)
                    logger.info(message)
                
                # Final cleanup of saved tensors
                del change_pred_for_metric, change_gt, G_pred, binary_pred
                # torch.cuda.empty_cache()

            ### Epoch Summary ###
            # Get binary change detection scores
            scores = metric.get_scores()
            epoch_acc = scores['mf1']
            
            # Get segmentation scores (6-class)
            scores_seg = metric_seg.get_scores()
            epoch_seg_mf1 = scores_seg['mf1']
            epoch_seg_miou = scores_seg['miou']
            epoch_seg_acc = scores_seg['acc']
            
            # Compute average epoch loss
            avg_epoch_loss = (epoch_loss / len(train_loader)) if len(train_loader) > 0 else 0.0
            
            # Log training epoch summary with both change and segmentation metrics
            wandb.log({
                # Binary change detection metrics
                'train/epoch_mF1_change': epoch_acc,
                'train/epoch_mIoU_change': scores.get('miou', 0),
                'train/epoch_OA_change': scores.get('acc', 0),
                # Multi-class segmentation metrics
                'train/epoch_mF1_seg': epoch_seg_mf1,
                'train/epoch_mIoU_seg': epoch_seg_miou,
                'train/epoch_OA_seg': epoch_seg_acc,
                # Overall metrics
                'train/epoch_loss': avg_epoch_loss,
                'train_epoch_mf1': epoch_acc,  # Keep for backward compatibility
                'train_epoch_loss': avg_epoch_loss,
                'epoch': current_epoch
            })


            #################
            #   VALIDATION  #
            #################

            logger.info('Starting validation...')
            val_metric = ConfuseMatrixMeter(n_class=opt['model']['n_classes'])
            val_metric_change = ConfuseMatrixMeter(n_class=2)
            cd_model.eval()
            val_loss_total = 0.0
            val_steps = 0
            shape_mismatch_logged = False
            
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
                    # Debug: Compare outputs vs ground-truth dtypes/shapes/devices (first val batch only)
                    if val_step == 0:
                        def _vtinfo(t: torch.Tensor, name: str):
                            try:
                                tmin = t.min().item()
                                tmax = t.max().item()
                                rng = f", min={tmin:.4f}, max={tmax:.4f}"
                            except Exception:
                                rng = ""
                            logger.info(f"[VAL dtype-check] {name}: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}{rng}")
                        if isinstance(val_outputs, (list, tuple)):
                            for i, out in enumerate(val_outputs):
                                _vtinfo(out, f"output[{i}]")
                        else:
                            _vtinfo(val_outputs, "output")
                        _vtinfo(val_seg_t1, "gt/seg_t1")
                        _vtinfo(val_seg_t2, "gt/seg_t2")
                        if val_change is not None:
                            _vtinfo(val_change, "gt/change(raw)")
                    
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
                    
                    # Segmentation predictions for metrics (multi-class)
                    val_pred_seg_t1 = torch.argmax(val_seg_logits_t1.detach(), dim=1)
                    val_pred_seg_t2 = torch.argmax(val_seg_logits_t2.detach(), dim=1)

                    # Update validation metrics
                    # Threshold class-1 probability for binary decision
                    val_change_p1 = torch.softmax(val_change_pred.detach(), dim=1)[:, 1, :, :]
                    val_binary_pred = (val_change_p1 > args.change_threshold).int()
                    
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
                    # Update confusion matrices
                    # 1) Segmentation (multi-class): update with T1 and T2 predictions vs GT
                    val_pred_seg_t1_np = val_pred_seg_t1.cpu().numpy().astype(np.uint8)
                    val_gt_seg_t1_np = val_seg_t1.detach().cpu().numpy().astype(np.uint8)
                    val_running_mf1_seg_t1 = val_metric.update_cm(pr=val_pred_seg_t1_np, gt=val_gt_seg_t1_np)

                    val_pred_seg_t2_np = val_pred_seg_t2.cpu().numpy().astype(np.uint8)
                    val_gt_seg_t2_np = val_seg_t2.detach().cpu().numpy().astype(np.uint8)
                    val_running_mf1_seg_t2 = val_metric.update_cm(pr=val_pred_seg_t2_np, gt=val_gt_seg_t2_np)

                    # Average the two heads' step F1 for logging
                    try:
                        val_running_mf1_seg = float((val_running_mf1_seg_t1 + val_running_mf1_seg_t2) / 2.0)
                    except Exception:
                        val_running_mf1_seg = float(val_running_mf1_seg_t2)

                    # 2) Change (binary): update with binary prediction vs binary GT
                    val_running_mf1_change = val_metric_change.update_cm(pr=val_pred_np, gt=val_gt_np)

                    # Self-check: compare GT vs GT to verify perfect metrics (logged only on first val batch)
                    selfcheck_logs = {}
                    if val_step == 0:
                        try:
                            # Segmentation self-check (multi-class)
                            _val_metric_seg_self = ConfuseMatrixMeter(n_class=opt['model']['n_classes'])
                            _val_metric_seg_self.update_cm(pr=val_gt_seg_t1_np, gt=val_gt_seg_t1_np)
                            _val_metric_seg_self.update_cm(pr=val_gt_seg_t2_np, gt=val_gt_seg_t2_np)
                            _sc_seg = _val_metric_seg_self.get_scores()

                            # Change self-check (binary)
                            _val_metric_change_self = ConfuseMatrixMeter(n_class=2)
                            _val_metric_change_self.update_cm(pr=val_gt_np, gt=val_gt_np)
                            _sc_ch = _val_metric_change_self.get_scores()

                            selfcheck_logs = {
                                'val/selfcheck_mF1_seg': float(_sc_seg.get('mf1', 0.0)),
                                'val/selfcheck_mIoU_seg': float(_sc_seg.get('miou', 0.0)),
                                'val/selfcheck_OA_seg': float(_sc_seg.get('acc', 0.0)),
                                'val/selfcheck_mF1_change': float(_sc_ch.get('mf1', 0.0)),
                                'val/selfcheck_mIoU_change': float(_sc_ch.get('miou', 0.0)),
                                'val/selfcheck_OA_change': float(_sc_ch.get('acc', 0.0)),
                            }
                        except Exception as e:
                            logger.warning(f"Validation self-check metrics failed: {e}")

                    # Per-step validation logging
                    _val_logs = {
                        'val_loss': float(val_loss.item()),
                        'val/running_mF1_seg': float(val_running_mf1_seg),
                        'val/running_mF1_change': float(val_running_mf1_change),
                    }
                    if selfcheck_logs:
                        _val_logs.update(selfcheck_logs)
                    wandb.log(_val_logs)
            
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
                        wandb.log({
                            "val/input_T1": [wandb.Image(norm_img(val_img1_np), caption="Val Input T1")],
                            "val/input_T2": [wandb.Image(norm_img(val_img2_np), caption="Val Input T2")],
                        }, commit=False)
                        # Reuse already-computed predictions
                        val_pred_change = val_binary_pred
                        
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
            
            # Validation epoch summary metrics
            val_scores_change = val_metric_change.get_scores()
            val_epoch_mf1_change = val_scores_change['mf1']
            val_epoch_miou_change = val_scores_change['miou']
            val_epoch_acc_change = val_scores_change['acc']
            avg_val_loss = val_loss_total / val_steps if val_steps > 0 else 0.0
            
            wandb.log({
                'val/epoch_loss': avg_val_loss,
                'val/epoch_mF1_change': val_epoch_mf1_change,
                'val/epoch_mIoU_change': val_epoch_miou_change,
                'val/epoch_OA_change': val_epoch_acc_change,
                'epoch': current_epoch
            })
            
            # Validation epoch summary metrics
            val_scores = val_metric.get_scores()
            val_epoch_mf1 = val_scores['mf1']
            val_epoch_miou = val_scores['miou']
            val_epoch_acc = val_scores['acc']
            val_epoch_sek = val_scores['SCD_Sek']
            val_epoch_fscd = val_scores['Fscd']
            val_epoch_iou_mean = val_scores['SCD_IoU_mean']
            
            wandb.log({
                'val/epoch_mF1': val_epoch_mf1,
                'val/epoch_mIoU': val_epoch_miou,
                'val/epoch_OA': val_epoch_acc,
                'val/epoch_sek': val_epoch_sek,
                'val/epoch_fscd': val_epoch_fscd,
                'val/epoch_iou_mean': val_epoch_iou_mean
            })
            
            logger.info(f'Validation - Epoch: {current_epoch}, Loss: {avg_val_loss:.5f}, mF1: {val_epoch_mf1:.5f}, mIoU: {val_epoch_miou:.5f}, OA: {val_epoch_acc:.5f}, Sek: {val_epoch_sek:.5f}, Fscd: {val_epoch_fscd:.5f}, IoU_mean: {val_epoch_iou_mean:.5f}')
            
            # Save best model based on validation mF1
            if val_epoch_mf1 > best_mF1:
                best_mF1 = val_epoch_mf1
                # Save the model state dict
                best_model_path = os.path.join(opt['path_cd']['checkpoint'], 'best_net.pth')
                
                # Handle DataParallel if used
                if isinstance(cd_model, nn.DataParallel):
                    model_state = cd_model.module.state_dict()
                else:
                    model_state = cd_model.state_dict()
                
                torch.save(model_state, best_model_path)
                logger.info(f'New best model saved with mF1: {best_mF1:.5f} at {best_model_path}')
                
                # Also save using the save_network function for compatibility
                save_network(opt, current_epoch, cd_model, optimer, is_best_model=True)
                
                # Log to wandb
                wandb.log({
                    'best_val_mF1': best_mF1,
                    'best_model_epoch': current_epoch
                })
            else:
                logger.info(f'Current mF1: {val_epoch_mf1:.5f} did not improve from best: {best_mF1:.5f}')
            
            # Save regular checkpoint every epoch (regardless of performance)
            save_network(opt, current_epoch, cd_model, optimer, is_best_model=False)
            
            # Update learning rate scheduler if needed
            # scheduler.step()  # Uncomment if using a scheduler
            
            # Load the best model for testing
            gen_path = os.path.join(opt['path_cd']['checkpoint'], 'best_net.pth')
            if os.path.exists(gen_path):
                cd_model.load_state_dict(torch.load(gen_path, map_location=device), strict=True)
                logger.info(f'Loaded best model from {gen_path}')
            else:
                logger.warning(f'Best model not found at {gen_path}, using current model')
            cd_model.to(device)
            metric.clear()
            cd_model.eval()
            
            # Create test result directory
            test_result_path = '{}/test'.format(opt['path_cd']['result'])
            os.makedirs(test_result_path, exist_ok=True)

            #################
            #    TESTING    #
            #################
            # Metrics for testing: change (binary) uses existing `metric`; add segmentation (multi-class)
            test_metric_seg = ConfuseMatrixMeter(n_class=opt['model']['n_classes'])
            test_seg_updates = 0
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
                    if current_step == 0:
                        import hashlib
                        def _fp(t: torch.Tensor):
                            t_cpu = t.detach().to('cpu')
                            return hashlib.sha1(t_cpu.numpy().tobytes()).hexdigest()[:12]
                        try:
                            logger.info(f"[TEST fp] A[0]={_fp(test_data['A'][0])}, B[0]={_fp(test_data['B'][0])}")
                        except Exception:
                            pass
                    # Extract all heads
                    seg_logits_t1, seg_logits_t2, change_pred = outputs
                    # Only use change head for metric and visuals (2-class)
                    # Convert prediction to binary change mask via thresholded probability
                    change_p1 = torch.softmax(change_pred.detach(), dim=1)[:, 1, :, :]
                    G_pred = (change_p1 > args.change_threshold).long()
                    # Normalize GT to binary [B,H,W]
                    test_change_bin = normalize_change_target(seg_t1, seg_t2, change)

                    # Prepare numpy arrays for metrics and update confusion matrix
                    pred_np = G_pred.int().cpu().numpy()
                    gt_np = test_change_bin.cpu().numpy().astype(np.uint8)
                    metric.update_cm(pr=pred_np, gt=gt_np)

                    # Update segmentation confusion matrix if GT available
                    if 'L1' in test_data and 'L2' in test_data:
                        pred_seg_t1 = torch.argmax(seg_logits_t1.detach(), dim=1)
                        pred_seg_t2 = torch.argmax(seg_logits_t2.detach(), dim=1)

                        pred_seg_t1_np = pred_seg_t1.cpu().numpy().astype(np.uint8)
                        pred_seg_t2_np = pred_seg_t2.cpu().numpy().astype(np.uint8)
                        gt_seg_t1_np = seg_t1.detach().cpu().numpy().astype(np.uint8)
                        gt_seg_t2_np = seg_t2.detach().cpu().numpy().astype(np.uint8)

                        # Basic shape alignment like in validation (squeeze potential extra dims)
                        if gt_seg_t1_np.ndim > pred_seg_t1_np.ndim:
                            gt_seg_t1_np = np.squeeze(gt_seg_t1_np)
                        if gt_seg_t2_np.ndim > pred_seg_t2_np.ndim:
                            gt_seg_t2_np = np.squeeze(gt_seg_t2_np)

                        test_metric_seg.update_cm(pr=pred_seg_t1_np, gt=gt_seg_t1_np)
                        test_metric_seg.update_cm(pr=pred_seg_t2_np, gt=gt_seg_t2_np)
                        test_seg_updates += 2

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

                        # Change probabilities (class-1 probability)
                        change_probs = torch.softmax(change_pred[0], dim=0)
                        change_prob = change_probs[1].detach().cpu().numpy()

                        # Segmentation predictions and per-pixel confidence (max prob)
                        pred_seg_t1 = torch.argmax(seg_logits_t1, dim=1)
                        pred_seg_t2 = torch.argmax(seg_logits_t2, dim=1)
                        pred_seg_t1_ali = pred_seg_t1.detach().cpu().numpy()
                        pred_seg_t2_ali = pred_seg_t2.detach().cpu().numpy()
                        

                        seg_t1_probs = torch.softmax(seg_logits_t1[0], dim=0)  # [C,H,W]
                        seg_t2_probs = torch.softmax(seg_logits_t2[0], dim=0)
                        seg_t1_max_prob = torch.max(seg_t1_probs, dim=0).values.detach().cpu().numpy()  # [H,W]
                        seg_t2_max_prob = torch.max(seg_t2_probs, dim=0).values.detach().cpu().numpy()
                        
                        wandb.log({
                            # Input images
                            "test/input_T1": [wandb.Image(norm_img(test_img1_np), caption="Test Input T1")],
                            "test/input_T2": [wandb.Image(norm_img(test_img2_np), caption="Test Input T2")],
                            # Multi-class segmentations (colorized)
                            "test/pred_seg_t1_ali": [wandb.Image(pred_seg_t1_ali[0]*40, caption="Test Pred Seg T1 (multi-class)")],
                            "test/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=opt['model']['n_classes']), caption="Test Pred Seg T1 (multi-class)")],
                            "test/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=opt['model']['n_classes']), caption="Test Pred Seg T2 (multi-class)")],
                            # Confidence maps
                            "test/pred_seg_t1_prob": [wandb.Image(seg_t1_max_prob, caption="Test Pred Seg T1 Max Probability")],
                            "test/pred_seg_t2_prob": [wandb.Image(seg_t2_max_prob, caption="Test Pred Seg T2 Max Probability")],
                            "test/pred_change": [wandb.Image(create_color_mask(G_pred[0], num_classes=2), caption="Test Pred Change (binary)")],
                            "test/pred_change_prob": [wandb.Image(change_prob, caption="Test Pred Change Class-1 Probability")],
                            "test/gt_change": [wandb.Image(create_color_mask(test_change_bin[0], num_classes=2), caption="Test GT Change (binary color)")],
                        })

                    # Visuals for saving PNGs
                    binary_pred = G_pred.int()
                    visuals = OrderedDict()
                    visuals['pred_cm'] = binary_pred  # Use binary prediction for visualization
                    visuals['gt_cm'] = test_change_bin.int()  # Use normalized binary GT for visualization

                    # Convert to uint8 images and save
                    img_A = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))
                    img_B = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))

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
                    
                    gt_cm = Metrics.tensor2img(gt_tensor.repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))
                    pred_cm = Metrics.tensor2img(pred_tensor.repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))

                    # Save imgs
                    Metrics.save_img(img_A, '{}/img_A_{}.png'.format(test_result_path, current_step))
                    Metrics.save_img(img_B, '{}/img_B_{}.png'.format(test_result_path, current_step))
                    Metrics.save_img(pred_cm, '{}/img_pred_cm{}.png'.format(test_result_path, current_step))
                    Metrics.save_img(gt_cm, '{}/img_gt_cm{}.png'.format(test_result_path, current_step))

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
                # WandB: log test epoch metrics (change and segmentation)
                wandb.log({
                    'test/epoch_mF1_change': float(scores.get('mf1', 0.0)),
                    'test/epoch_mIoU_change': float(scores.get('miou', 0.0)),
                    'test/epoch_OA_change': float(scores.get('acc', 0.0)),
                    'epoch': current_epoch
                })

                if test_seg_updates > 0:
                    test_scores_seg = test_metric_seg.get_scores()
                    wandb.log({
                        'test/epoch_mF1': float(test_scores_seg.get('mf1', 0.0)),
                        'test/epoch_mIoU': float(test_scores_seg.get('miou', 0.0)),
                        'test/epoch_OA': float(test_scores_seg.get('acc', 0.0)),
                        'test/epoch_sek': float(test_scores_seg.get('SCD_Sek', 0.0)),
                        'test/epoch_fscd': float(test_scores_seg.get('Fscd', 0.0)),
                        'test/epoch_iou_mean': float(test_scores_seg.get('SCD_IoU_mean', 0.0)),
                        'epoch': current_epoch
                    })
                logger.info('End of testing...')
