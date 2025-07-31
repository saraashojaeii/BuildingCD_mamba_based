import torch
import torch.optim as optim
import data as Data
import models as Model
import torch.nn as nn
import argparse
import logging
import core.logger as Logger
import os
import numpy as np
from misc.metric_tools import ConfuseMatrixMeter
from models.loss import *
from collections import OrderedDict
import core.metrics as Metrics
from misc.torchutils import get_scheduler, save_network
import wandb
import matplotlib
import matplotlib.pyplot as plt
from accelerate import Accelerator


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
    # Fix matplotlib deprecation warning
    cmap = _mpl.colormaps.get_cmap('tab10')
    if hasattr(cmap, 'resampled'):
        cmap = cmap.resampled(num_classes)
    rgb = _np.zeros((h, w, 3), dtype=_np.uint8)
    for cls in range(num_classes):
        rgb[arr == cls] = (_np.array(cmap(cls)[:3]) * 255).astype(_np.uint8)
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

    # Initialize Accelerator
    accelerator = Accelerator()

    # Logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(logger_name=None, root=opt['path_cd']['log'], phase='train',
                        level=logging.INFO, screen=True)
    Logger.setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test',
                        level=logging.INFO)
    logger = logging.getLogger('base')
    
    # Only log on main process
    if accelerator.is_main_process:
        logger.info(Logger.dict2str(opt))

    # Initialize wandb only on main process
    if accelerator.is_main_process and opt.get('wandb') and opt['wandb'].get('project'):
        wandb.init(project=opt['wandb']['project'], config=opt)
    else:
        wandb.init(mode="disabled")

    # Dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            print("Create [train] change-detection dataloader")
            train_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

        elif phase == 'val' and args.phase != 'test':
            print("Create [val] change-detection dataloader")
            val_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            val_loader = Data.create_cd_dataloader(val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)

        elif phase == 'test':
            print("Create [test] change-detection dataloader")
            test_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)

    logger.info('Initial Dataset Finished')

    # Create cd model
    cd_model = Model.create_CD_model(opt)

    num_classes = opt['model']['n_classes']
    logger.info(f"Number of classes for loss function: {num_classes}")

    # Create loss function
    if opt['model']['loss'] == 'multi_class_cd':
        loss_fun = MultiClassCDLoss(
            seg_weight=opt['model']['loss_weights']['seg_weight'],
            change_weight=opt['model']['loss_weights']['change_weight'],
            num_classes=num_classes
        )
        logger.info(f"Using MultiClassCDLoss with {num_classes} classes")
    else:
        loss_fun = nn.BCELoss()
        logger.info("Using BCELoss for binary change detection")

    # Create optimizer
    if opt['train']["optimizer"]["type"] == 'adam':
        optimer = optim.Adam(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'adamw':
        optimer = optim.AdamW(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'sgd':
        optimer = optim.SGD(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"],
                            momentum=0.9, weight_decay=5e-4)

    # Prepare everything with accelerator
    cd_model, optimer, train_loader = accelerator.prepare(cd_model, optimer, train_loader)
    if 'val_loader' in locals():
        val_loader = accelerator.prepare(val_loader)

    metric = ConfuseMatrixMeter(n_class=2)  # For binary change detection (change/no-change)
    log_dict = OrderedDict()
    
    #################
    # Training loop #
    #################
    if opt['phase'] == 'train':
        best_mF1 = 0.0
        epoch_losses = []
        for current_epoch in range(0, opt['train']['n_epoch']):
            if accelerator.is_main_process:
                print("......Begin Training......")
            metric.clear()
            cd_model.train()
            train_result_path = '{}/train/{}'.format(opt['path_cd']['result'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)

            #################
            #    Training   #
            #################
            message = 'lr: %0.7f\n \n' % optimer.param_groups[0]['lr']
            if accelerator.is_main_process:
                logger.info(message)

            epoch_loss = 0

            for current_step, train_data in enumerate(train_loader):
                # Data is automatically moved to correct device by accelerator
                train_im1 = train_data['A']
                train_im2 = train_data['B']
                # Robust label extraction - data automatically on correct device
                seg_t1 = train_data['L1'] if 'L1' in train_data else train_data['L']
                seg_t2 = train_data['L2'] if 'L2' in train_data else train_data['L']
                change = train_data['change'] if 'change' in train_data else train_data['L']

                outputs = cd_model(train_im1, train_im2)

                if isinstance(loss_fun, MultiClassCDLoss):
                    labels = {'seg_t1': seg_t1, 'seg_t2': seg_t2, 'change': change}
                    train_loss, loss_dict = loss_fun(outputs, labels)
                    seg_logits_t1, seg_logits_t2, change_pred = outputs
                    with torch.no_grad():
                        pred_seg_t1 = torch.argmax(seg_logits_t1, dim=1)
                        pred_seg_t2 = torch.argmax(seg_logits_t2, dim=1)
                        pred_change = torch.argmax(change_pred, dim=1)
                    
                    # Log masks to wandb (log only for the first batch of each epoch to avoid excessive logging)
                    if current_step == 0 and current_epoch % 1 == 0 and accelerator.is_main_process:
                        wandb.log({
                            "train/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=opt['model']['n_classes']), caption="Pred Seg T1 (multi-class)")],
                            "train/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=opt['model']['n_classes']), caption="Pred Seg T2 (multi-class)")],
                            "train/pred_change": [wandb.Image(create_color_mask(pred_change[0], num_classes=opt['model']['n_classes'] * opt['model']['n_classes']), caption="Pred Change (multi-class)")],
                            "train/gt_seg_t1": [wandb.Image(create_color_mask(seg_t1[0], num_classes=opt['model']['n_classes']), caption="GT Seg T1 (multi-class)")],
                            "train/gt_seg_t2": [wandb.Image(create_color_mask(seg_t2[0], num_classes=opt['model']['n_classes']), caption="GT Seg T2 (multi-class)")],
                            "train/gt_change": [wandb.Image(create_color_mask(change[0], num_classes=opt['model']['n_classes'] * opt['model']['n_classes']), caption="GT Change (multi-class)")],
                            "global_step": current_epoch * len(train_loader) + current_step
                        })

                else:
                    # Assumes binary loss on the change prediction head
                    change_pred = outputs[2] if isinstance(outputs, tuple) and len(outputs) > 2 else outputs
                    train_loss = loss_fun(change_pred, change)
                    # Create a dummy loss_dict for logging consistency
                    loss_dict = {'seg_t1': 0, 'seg_t2': 0, 'change': train_loss.item()}
                    seg_logits_t1 = seg_logits_t2 = torch.zeros_like(change_pred)  # dummy for logging
                
                # Convert logits to predicted masks for logging
                with torch.no_grad():
                    pred_seg_t1 = torch.argmax(seg_logits_t1, dim=1)
                    pred_seg_t2 = torch.argmax(seg_logits_t2, dim=1)
                    pred_change = torch.argmax(change_pred, dim=1)
                
                # Log masks to wandb (log only for the first batch of each epoch to avoid excessive logging)
                if current_step == 0 and current_epoch % 1 == 0 and accelerator.is_main_process:
                    wandb.log({
                        "train/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=num_classes), caption="Pred Seg T1 (multi-class)")],
                        "train/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=num_classes), caption="Pred Seg T2 (multi-class)")],
                        "train/pred_change": [wandb.Image(create_color_mask(pred_change[0], num_classes=num_classes * num_classes), caption="Pred Change (multi-class)")],
                        "train/gt_seg_t1": [wandb.Image(create_color_mask(seg_t1[0], num_classes=num_classes), caption="GT Seg T1")],
                        "train/gt_seg_t2": [wandb.Image(create_color_mask(seg_t2[0], num_classes=num_classes), caption="GT Seg T2")],
                        "train/gt_change": [wandb.Image(create_color_mask(change[0], num_classes=num_classes * num_classes), caption="GT Change")],
                        "global_step": current_epoch * len(train_loader) + current_step
                    })
                
                # Backward pass with accelerator
                optimer.zero_grad()
                accelerator.backward(train_loss)
                optimer.step()
                
                log_dict['loss'] = train_loss.item()
                log_dict['loss_seg_t1'] = loss_dict['seg_t1']
                log_dict['loss_seg_t2'] = loss_dict['seg_t2']
                log_dict['loss_change'] = loss_dict['change']
                epoch_loss += train_loss.item()

                # For metric, convert transition prediction to binary change map
                change_pred = outputs[2]  # [B, num_classes*num_classes, H, W]
                G_pred = torch.argmax(change_pred.detach(), dim=1)

                n_classes = opt['model']['n_classes']
                from_class = G_pred // n_classes
                to_class = G_pred % n_classes
                
                # Create binary change mask: 1 if classes are different, 0 if same
                binary_change_pred = (from_class != to_class).float()
                
                # Convert ground truth change to binary
                G_gt = torch.argmax(change.detach(), dim=1)
                from_class_gt = G_gt // n_classes
                to_class_gt = G_gt % n_classes
                binary_change_gt = (from_class_gt != to_class_gt).float()

                current_score = metric.update_cm(pr=binary_change_pred.cpu().numpy(), gt=binary_change_gt.cpu().numpy())
                
                # Log training metrics on main process
                if accelerator.is_main_process:
                    wandb.log({
                        "train/loss": train_loss.item(),
                        "train/loss_seg_t1": loss_dict['seg_t1'],
                        "train/loss_seg_t2": loss_dict['seg_t2'],
                        "train/loss_change": loss_dict['change'],
                        "train/F1": current_score['F1'],
                        "train/IoU": current_score['IoU'],
                        "global_step": current_epoch * len(train_loader) + current_step
                    })

            # Calculate epoch metrics
            scores = metric.get_scores()
            epoch_loss = epoch_loss / len(train_loader)
            epoch_losses.append(epoch_loss)
            
            if accelerator.is_main_process:
                logger.info('Epoch: {}, Loss: {:.4f}, F1: {:.4f}, IoU: {:.4f}'.format(
                    current_epoch, epoch_loss, scores['F1'], scores['IoU']))
                
                wandb.log({
                    "epoch/train_loss": epoch_loss,
                    "epoch/train_F1": scores['F1'],
                    "epoch/train_IoU": scores['IoU'],
                    "epoch": current_epoch
                })

            #################
            #   Validation  #
            #################
            if 'val_loader' in locals() and current_epoch % opt['train']['val_freq'] == 0:
                cd_model.eval()
                val_metric = ConfuseMatrixMeter(n_class=2)
                val_loss = 0.0
                
                with torch.no_grad():
                    for val_step, val_data in enumerate(val_loader):
                        val_im1 = val_data['A']
                        val_im2 = val_data['B']
                        val_seg_t1 = val_data['L1'] if 'L1' in val_data else val_data['L']
                        val_seg_t2 = val_data['L2'] if 'L2' in val_data else val_data['L']
                        val_change = val_data['change'] if 'change' in val_data else val_data['L']

                        val_outputs = cd_model(val_im1, val_im2)
                        
                        if isinstance(loss_fun, MultiClassCDLoss):
                            val_labels = {'seg_t1': val_seg_t1, 'seg_t2': val_seg_t2, 'change': val_change}
                            val_loss_batch, _ = loss_fun(val_outputs, val_labels)
                        else:
                            val_change_pred = val_outputs[2] if isinstance(val_outputs, tuple) and len(val_outputs) > 2 else val_outputs
                            val_loss_batch = loss_fun(val_change_pred, val_change)
                        
                        val_loss += val_loss_batch.item()
                        
                        # Convert to binary for metrics
                        val_change_pred = val_outputs[2]
                        val_G_pred = torch.argmax(val_change_pred.detach(), dim=1)
                        val_from_class = val_G_pred // n_classes
                        val_to_class = val_G_pred % n_classes
                        val_binary_change_pred = (val_from_class != val_to_class).float()
                        
                        val_G_gt = torch.argmax(val_change.detach(), dim=1)
                        val_from_class_gt = val_G_gt // n_classes
                        val_to_class_gt = val_G_gt % n_classes
                        val_binary_change_gt = (val_from_class_gt != val_to_class_gt).float()
                        
                        val_metric.update_cm(pr=val_binary_change_pred.cpu().numpy(), 
                                           gt=val_binary_change_gt.cpu().numpy())

                val_scores = val_metric.get_scores()
                val_loss = val_loss / len(val_loader)
                
                if accelerator.is_main_process:
                    logger.info('Validation - Epoch: {}, Loss: {:.4f}, F1: {:.4f}, IoU: {:.4f}'.format(
                        current_epoch, val_loss, val_scores['F1'], val_scores['IoU']))
                    
                    wandb.log({
                        "epoch/val_loss": val_loss,
                        "epoch/val_F1": val_scores['F1'],
                        "epoch/val_IoU": val_scores['IoU'],
                        "epoch": current_epoch
                    })

                # Save best model
                if val_scores['F1'] > best_mF1:
                    best_mF1 = val_scores['F1']
                    if accelerator.is_main_process:
                        logger.info(f'New best F1: {best_mF1:.4f}')
                        # Save model using accelerator
                        accelerator.save_model(cd_model, f"{opt['path_cd']['checkpoint']}/best_model")

            # Save checkpoint periodically
            if current_epoch % opt['train']['save_checkpoint_freq'] == 0 and accelerator.is_main_process:
                accelerator.save_model(cd_model, f"{opt['path_cd']['checkpoint']}/epoch_{current_epoch}")

    #################
    #     Testing   #
    #################
    elif opt['phase'] == 'test':
        cd_model.eval()
        test_metric = ConfuseMatrixMeter(n_class=2)
        
        with torch.no_grad():
            for test_step, test_data in enumerate(test_loader):
                test_im1 = test_data['A']
                test_im2 = test_data['B']
                test_change = test_data['change'] if 'change' in test_data else test_data['L']

                test_outputs = cd_model(test_im1, test_im2)
                test_change_pred = test_outputs[2]
                
                # Convert to binary for metrics
                test_G_pred = torch.argmax(test_change_pred.detach(), dim=1)
                test_from_class = test_G_pred // n_classes
                test_to_class = test_G_pred % n_classes
                test_binary_change_pred = (test_from_class != test_to_class).float()
                
                test_G_gt = torch.argmax(test_change.detach(), dim=1)
                test_from_class_gt = test_G_gt // n_classes
                test_to_class_gt = test_G_gt % n_classes
                test_binary_change_gt = (test_from_class_gt != test_to_class_gt).float()
                
                test_metric.update_cm(pr=test_binary_change_pred.cpu().numpy(), 
                                    gt=test_binary_change_gt.cpu().numpy())

        test_scores = test_metric.get_scores()
        if accelerator.is_main_process:
            logger.info('Test Results - F1: {:.4f}, IoU: {:.4f}'.format(test_scores['F1'], test_scores['IoU']))
            
            wandb.log({
                "test/F1": test_scores['F1'],
                "test/IoU": test_scores['IoU']
            })

    if accelerator.is_main_process:
        wandb.finish()
