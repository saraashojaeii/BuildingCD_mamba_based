import torch
import os
# Set CUDA memory management before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch.optim as optim
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

# Import Accelerate for multi-GPU training
from accelerate import Accelerator
from accelerate.utils import set_seed


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/saraashojaeii/git/BuildingCD_mamba_based/config/second_cdmamba/second_cdmamba.json',
                        help='JSON file for configuration')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'], help='Run either train(training + validation) or testing',)
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('-log_eval', action='store_true')

    # Parse config
    args = parser.parse_args()
    opt = Logger.parse(args)

    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # Initialize Accelerator for multi-GPU training
    accelerator = Accelerator(
        gradient_accumulation_steps=opt['train'].get('gradient_accumulation_steps', 1),
        mixed_precision=opt['train'].get('mixed_precision', 'fp16'),  # Use fp16 for better memory efficiency
        log_with="wandb" if opt.get('wandb') and opt['wandb'].get('project') else None,
        project_dir=opt['path_cd']['log'] if opt.get('path_cd') else None
    )

    # Set seed for reproducibility
    if opt.get('manual_seed'):
        set_seed(opt['manual_seed'])

    # Logging - only on main process
    if accelerator.is_main_process:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        Logger.setup_logger(logger_name=None, root=opt['path_cd']['log'], phase='train',
                            level=logging.INFO, screen=True)
        Logger.setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test',
                            level=logging.INFO)
        logger = logging.getLogger('base')
        logger.info(Logger.dict2str(opt))
        
        # Initialize wandb only on main process
        if opt.get('wandb') and opt['wandb'].get('project'):
            accelerator.init_trackers(
                project_name=opt['wandb']['project'],
                config=opt
            )
    else:
        # Create dummy logger for non-main processes
        logger = logging.getLogger('base')
        logger.addHandler(logging.NullHandler())

    # Dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            train_set = Data.create_cd_dataset(dataset_opt, phase)
            train_loader = Data.create_cd_dataloader(
                train_set, dataset_opt, phase)
            if accelerator.is_main_process:
                logger.info('Dataset [{:s}] is created.'.format(dataset_opt['name']))
        elif phase == 'val':
            val_set = Data.create_cd_dataset(dataset_opt, phase)
            val_loader = Data.create_cd_dataloader(
                val_set, dataset_opt, phase)
            if accelerator.is_main_process:
                logger.info('Dataset [{:s}] is created.'.format(dataset_opt['name']))

    # Model
    cd_model = Model.create_CD_model(opt)
    
    if accelerator.is_main_process:
        logger.info('Initial Model Finished')

    # Create optimizer
    optim_params = []
    for k, v in cd_model.named_parameters():
        if v.requires_grad:
            optim_params.append(v)
        else:
            if accelerator.is_main_process:
                logger.warning('Params [{:s}] will not optimize.'.format(k))

    if opt['train']['optimizer']['type'] == 'adam':
        optimizer = optim.Adam(optim_params, lr=opt['train']['optimizer']['lr'])
    elif opt['train']['optimizer']['type'] == 'adamw':
        optimizer = optim.AdamW(optim_params, lr=opt['train']['optimizer']['lr'])
    else:
        raise NotImplementedError('Optimizer [{:s}] is not found'.format(opt['train']['optimizer']['type']))

    # Create scheduler
    scheduler = get_scheduler(optimizer, opt['train'])

    # Prepare everything with accelerator
    cd_model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        cd_model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        logger.info('Begin Model Training.')

    current_step = 0
    current_epoch = 0
    best_acc = 0.0
    best_epoch = 0

    # Training loop
    for epoch in range(opt['train']['n_epoch']):
        current_epoch = epoch
        
        # Training phase
        cd_model.train()
        
        # Reset metrics for this epoch
        metric = ConfuseMatrixMeter(n_class=opt['model']['n_classes'])
        
        for i, train_data in enumerate(train_loader):
            current_step += 1
            
            # Get data (accelerator handles device placement)
            train_im1 = train_data['A']
            train_im2 = train_data['B']
            # Robust label extraction with fallbacks
            seg_t1 = train_data.get('L1', train_data.get('seg_t1', train_data['L']))
            seg_t2 = train_data.get('L2', train_data.get('seg_t2', train_data['L']))
            change = train_data.get('change', train_data['L'])

            # Forward pass
            with accelerator.accumulate(cd_model):
                # Get model outputs
                outputs = cd_model(train_im1, train_im2)
                
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    pred_seg_t1, pred_seg_t2, pred_change = outputs
                else:
                    raise ValueError(f"Expected 3 outputs from model, got {len(outputs) if isinstance(outputs, tuple) else 'single output'}")

                # Calculate losses
                loss_seg_t1 = nn.CrossEntropyLoss()(pred_seg_t1, seg_t1.squeeze(1).long())
                loss_seg_t2 = nn.CrossEntropyLoss()(pred_seg_t2, seg_t2.squeeze(1).long())
                loss_change = nn.CrossEntropyLoss()(pred_change, change.squeeze(1).long())
                
                total_loss = loss_seg_t1 + loss_seg_t2 + loss_change

                # Backward pass
                accelerator.backward(total_loss)
                optimizer.step()
                optimizer.zero_grad()

            # Update metrics (only on main process for logging)
            if accelerator.is_main_process:
                # Convert predictions to class indices
                pred_seg_t1_idx = torch.argmax(pred_seg_t1, dim=1).cpu().numpy()
                pred_seg_t2_idx = torch.argmax(pred_seg_t2, dim=1).cpu().numpy()
                pred_change_idx = torch.argmax(pred_change, dim=1).cpu().numpy()
                
                seg_t1_np = seg_t1.squeeze(1).cpu().numpy()
                seg_t2_np = seg_t2.squeeze(1).cpu().numpy()
                change_np = change.squeeze(1).cpu().numpy()
                
                # Update confusion matrix
                metric.update_cm(pr=pred_change_idx, gt=change_np)

                # Log training metrics
                if current_step % opt['train'].get('train_print_iter', 100) == 0:
                    logs = {
                        'epoch': current_epoch,
                        'iter': current_step,
                        'lr': optimizer.param_groups[0]['lr'],
                        'loss_total': total_loss.item(),
                        'loss_seg_t1': loss_seg_t1.item(),
                        'loss_seg_t2': loss_seg_t2.item(),
                        'loss_change': loss_change.item()
                    }
                    
                    message = '[epoch:{:3d}, iter:{:8,d}] '.format(current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        message += '\n'
                    logger.info(message)
                    
                    # Log to wandb
                    accelerator.log(logs, step=current_step)

                # Log training visualizations periodically
                if current_step % (opt['train'].get('train_print_iter', 100) * 10) == 0:
                    accelerator.log({
                        "train/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1_idx[0]), caption="Train Pred Seg T1")],
                        "train/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2_idx[0]), caption="Train Pred Seg T2")],
                        "train/pred_change": [wandb.Image(create_color_mask(pred_change_idx[0], num_classes=opt['model']['n_classes']*opt['model']['n_classes']), caption="Train Pred Change")],
                        "train/gt_seg_t1": [wandb.Image(create_color_mask(seg_t1_np[0]), caption="Train GT Seg T1")],
                        "train/gt_seg_t2": [wandb.Image(create_color_mask(seg_t2_np[0]), caption="Train GT Seg T2")],
                        "train/gt_change": [wandb.Image(create_color_mask(change_np[0], num_classes=opt['model']['n_classes']*opt['model']['n_classes']), caption="Train GT Change")],
                    }, step=current_step)

        # Wait for all processes to finish epoch
        accelerator.wait_for_everyone()
        
        # Update scheduler
        scheduler.step()

        # Validation phase (only on main process)
        if accelerator.is_main_process and current_epoch % opt['train'].get('val_freq', 1) == 0:
            cd_model.eval()
            metric_val = ConfuseMatrixMeter(n_class=opt['model']['n_classes'])
            
            with torch.no_grad():
                for i, val_data in enumerate(val_loader):
                    val_im1 = val_data['A']
                    val_im2 = val_data['B']
                    # Robust label extraction with fallbacks
                    seg_t1 = val_data.get('L1', val_data.get('seg_t1', val_data['L']))
                    seg_t2 = val_data.get('L2', val_data.get('seg_t2', val_data['L']))
                    change = val_data.get('change', val_data['L'])

                    # Forward pass
                    outputs = cd_model(val_im1, val_im2)
                    
                    if isinstance(outputs, tuple) and len(outputs) == 3:
                        pred_seg_t1, pred_seg_t2, pred_change = outputs
                    else:
                        raise ValueError(f"Expected 3 outputs from model, got {len(outputs) if isinstance(outputs, tuple) else 'single output'}")

                    # Convert predictions to class indices
                    pred_change_idx = torch.argmax(pred_change, dim=1).cpu().numpy()
                    change_np = change.squeeze(1).cpu().numpy()
                    
                    # Update validation metrics
                    metric_val.update_cm(pr=pred_change_idx, gt=change_np)

                    # Log validation visualizations for first batch
                    if i == 0:
                        pred_seg_t1_idx = torch.argmax(pred_seg_t1, dim=1).cpu().numpy()
                        pred_seg_t2_idx = torch.argmax(pred_seg_t2, dim=1).cpu().numpy()
                        seg_t1_np = seg_t1.squeeze(1).cpu().numpy()
                        seg_t2_np = seg_t2.squeeze(1).cpu().numpy()
                        
                        accelerator.log({
                            "val/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1_idx[0]), caption="Val Pred Seg T1")],
                            "val/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2_idx[0]), caption="Val Pred Seg T2")],
                            "val/pred_change": [wandb.Image(create_color_mask(pred_change_idx[0], num_classes=opt['model']['n_classes']*opt['model']['n_classes']), caption="Val Pred Change")],
                            "val/gt_seg_t1": [wandb.Image(create_color_mask(seg_t1_np[0]), caption="Val GT Seg T1")],
                            "val/gt_seg_t2": [wandb.Image(create_color_mask(seg_t2_np[0]), caption="Val GT Seg T2")],
                            "val/gt_change": [wandb.Image(create_color_mask(change_np[0], num_classes=opt['model']['n_classes']*opt['model']['n_classes']), caption="Val GT Change")],
                        }, step=current_step)

            # Calculate validation metrics
            scores = metric_val.get_scores()
            epoch_acc = scores['mf1']
            
            # Log validation metrics
            val_logs = {
                'val_epoch': current_epoch,
                'val_mf1': epoch_acc.item()
            }
            for k, v in scores.items():
                val_logs[f'val_{k}'] = v
            
            accelerator.log(val_logs, step=current_step)
            
            message = '[Validation] Epoch: {:3d}, mF1: {:.5f}\n'.format(current_epoch, epoch_acc.item())
            for k, v in scores.items():
                message += '{:s}: {:.4e} '.format(k, v)
                message += '\n'
            logger.info(message)

            # Save best model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = current_epoch
                
                # Save model (unwrap from accelerator first)
                unwrapped_model = accelerator.unwrap_model(cd_model)
                save_network(unwrapped_model, 'best', opt['path_cd']['models'])
                logger.info(f'New best model saved at epoch {current_epoch} with mF1: {best_acc:.5f}')

        # Wait for all processes before next epoch
        accelerator.wait_for_everyone()

        # Save checkpoint periodically (only on main process)
        if accelerator.is_main_process and current_epoch % opt['train'].get('save_checkpoint_freq', 10) == 0:
            unwrapped_model = accelerator.unwrap_model(cd_model)
            save_network(unwrapped_model, current_epoch, opt['path_cd']['models'])
            logger.info(f'Checkpoint saved at epoch {current_epoch}')

    # Final cleanup
    if accelerator.is_main_process:
        logger.info(f'Training completed. Best mF1: {best_acc:.5f} at epoch {best_epoch}')
        accelerator.end_training()
