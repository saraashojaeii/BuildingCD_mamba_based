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

                # Debug: Print input tensor info
                if current_step == 0:
                    print("\n" + "="*60)
                    print(f"EPOCH {current_epoch}, BATCH {current_step} - INPUT DEBUG INFO")
                    print("="*60)
                    print(f"Input T1 shape: {train_im1.shape}, dtype: {train_im1.dtype}")
                    print(f"Input T1 range: [{train_im1.min():.4f}, {train_im1.max():.4f}]")
                    print(f"Input T1 mean: {train_im1.mean():.4f}, std: {train_im1.std():.4f}")
                    print(f"Input T2 shape: {train_im2.shape}, dtype: {train_im2.dtype}")
                    print(f"Input T2 range: [{train_im2.min():.4f}, {train_im2.max():.4f}]")
                    print(f"Input T2 mean: {train_im2.mean():.4f}, std: {train_im2.std():.4f}")
                    print("-"*60)

                # Use gradient checkpointing to save memory
                outputs = cd_model(train_im1, train_im2)
                
                # Process outputs based on model architecture (extract components first)
                if isinstance(outputs, dict):
                    seg_logits_t1 = outputs.get('seg_t1', None)
                    seg_logits_t2 = outputs.get('seg_t2', None)
                    change_pred = outputs.get('change', None)
                elif isinstance(outputs, (list, tuple)):
                    if len(outputs) == 3:
                        seg_logits_t1, seg_logits_t2, change_pred = outputs
                    else:
                        change_pred = outputs[0] if len(outputs) > 0 else outputs
                        seg_logits_t1, seg_logits_t2 = None, None
                else:
                    change_pred = outputs
                    seg_logits_t1, seg_logits_t2 = None, None
                
                # Debug: Analyze and print output structure
                if current_step == 0:
                    print("\nOUTPUT DEBUG INFO:")
                    print("-"*60)
                    print(f"Output type: {type(outputs)}")
                    
                    if isinstance(outputs, dict):
                        print("Output is a dictionary with keys:", outputs.keys())
                    elif isinstance(outputs, (list, tuple)):
                        print(f"Output is a {type(outputs).__name__} with {len(outputs)} elements")
                    else:
                        print(f"Output is a tensor with shape: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
                    
                    # Print details for each output
                    if seg_logits_t1 is not None:
                        print(f"Seg T1 logits - shape: {seg_logits_t1.shape}, range: [{seg_logits_t1.min():.4f}, {seg_logits_t1.max():.4f}]")
                        pred_seg_t1_debug = torch.argmax(seg_logits_t1, dim=1)
                        print(f"Seg T1 predictions - unique classes: {torch.unique(pred_seg_t1_debug).tolist()}")
                    
                    if seg_logits_t2 is not None:
                        print(f"Seg T2 logits - shape: {seg_logits_t2.shape}, range: [{seg_logits_t2.min():.4f}, {seg_logits_t2.max():.4f}]")
                        pred_seg_t2_debug = torch.argmax(seg_logits_t2, dim=1)
                        print(f"Seg T2 predictions - unique classes: {torch.unique(pred_seg_t2_debug).tolist()}")
                    
                    if change_pred is not None:
                        print(f"Change logits - shape: {change_pred.shape}, range: [{change_pred.min():.4f}, {change_pred.max():.4f}]")
                        change_probs = torch.softmax(change_pred, dim=1)
                        print(f"Change probs (class 0) - min: {change_probs[:,0,:,:].min():.4f}, max: {change_probs[:,0,:,:].max():.4f}")
                        print(f"Change probs (class 1) - min: {change_probs[:,1,:,:].min():.4f}, max: {change_probs[:,1,:,:].max():.4f}")
                        binary_change = (change_probs[:, 1, :, :] > args.change_threshold).long()
                        unique_vals = torch.unique(binary_change)
                        print(f"Binary change mask - unique values: {unique_vals.tolist()}")
                        if len(unique_vals) > 0:
                            change_ratio = (binary_change == 1).sum().float() / binary_change.numel()
                            print(f"Change pixel ratio: {change_ratio:.4f} ({(binary_change == 1).sum()} / {binary_change.numel()})")
                    print("="*60 + "\n")
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
                
                # Debug: Print ground truth info
                if current_step == 0:
                    print("\nGROUND TRUTH DEBUG INFO:")
                    print("-"*60)
                    if seg_t1 is not None:
                        print(f"GT Seg T1 - shape: {seg_t1.shape}, dtype: {seg_t1.dtype}")
                        print(f"GT Seg T1 - unique classes: {torch.unique(seg_t1).tolist()}")
                    else:
                        print("GT Seg T1: None")
                    
                    if seg_t2 is not None:
                        print(f"GT Seg T2 - shape: {seg_t2.shape}, dtype: {seg_t2.dtype}")
                        print(f"GT Seg T2 - unique classes: {torch.unique(seg_t2).tolist()}")
                    else:
                        print("GT Seg T2: None")
                    
                    if change is not None:
                        print(f"GT Change - shape: {change.shape}, dtype: {change.dtype}")
                        print(f"GT Change - unique values: {torch.unique(change).tolist()}")
                        if change.numel() > 0:
                            change_ratio = (change == 1).sum().float() / change.numel()
                            print(f"GT Change pixel ratio: {change_ratio:.4f} ({(change == 1).sum()} / {change.numel()})")
                    else:
                        print("GT Change: None")
                    print("-"*60)
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
                    # Note: seg_logits_t1, seg_logits_t2, change_pred already extracted above
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
                    # Note: seg_logits_t1, seg_logits_t2, change_pred already extracted above
                    # Create dummy segmentation logits if not available
                    if seg_logits_t1 is None or seg_logits_t2 is None:
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
                
                # Debug: Print loss info
                if current_step == 0:
                    print("\nLOSS DEBUG INFO:")
                    print("-"*60)
                    print(f"Total train loss: {train_loss.item():.6f}")
                    if 'loss_dict' in locals():
                        print(f"Loss components - seg_t1: {loss_dict['seg_t1']:.6f}, seg_t2: {loss_dict['seg_t2']:.6f}, change: {loss_dict['change']:.6f}")
                    print(f"Loss requires_grad: {train_loss.requires_grad}")
                    print("-"*60)
                
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
            epoch_losses.append(avg_epoch_loss)  # Track loss history
            
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
            
            # Log training progress summary
            logger.info(f'Training - Epoch: {current_epoch}, Avg Loss: {avg_epoch_loss:.5f}, Change mF1: {epoch_acc:.5f}, Seg mF1: {epoch_seg_mf1:.5f}')
            if len(epoch_losses) > 1:
                loss_trend = "↓" if epoch_losses[-1] < epoch_losses[-2] else "↑"
                logger.info(f'Loss trend: {loss_trend} (Previous: {epoch_losses[-2]:.5f}, Current: {epoch_losses[-1]:.5f})')



