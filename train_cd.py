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
    parser =argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/saraashojaeii/git/BuildingCD_mamba_based/config/second_cdmamba/second_cdmamba.json',
                        help='JSON file for configuration')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'], help='Run either train(training + validation) or testing',)
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('-log_eval', action='store_true')

    #paser config
    args = parser.parse_args()
    opt = Logger.parse(args)

    #Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    #logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(logger_name=None, root=opt['path_cd']['log'], phase='train',
                        level=logging.INFO, screen=True)
    Logger.setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test',
                        level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize wandb only on main process
    if opt.get('wandb') and opt['wandb'].get('project'):
        wandb.init(project=opt['wandb']['project'], config=opt)
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
    cd_model.to(device)
    
    # Enable gradient checkpointing if available to save memory
    if hasattr(cd_model, 'gradient_checkpointing_enable'):
        cd_model.gradient_checkpointing_enable()

    num_classes = opt['model']['n_classes']
    logger.info(f"Number of classes for loss function: {num_classes}")

    #Create criterion
    if opt['model']['loss'] == 'ce_dice':
        loss_fun = CEDiceLoss(num_classes=num_classes)
    elif opt['model']['loss'] == 'ce':
        # CrossEntropy can be used as a function or nn.Module. Using function for now.
        loss_fun = cross_entropy_loss_fn 
    elif opt['model']['loss'] == 'dice':
        loss_fun = DiceOnlyLoss(num_classes=num_classes)
    elif opt['model']['loss'] == 'ce2_dice1':
        loss_fun = CE2Dice1Loss(num_classes=num_classes)
    elif opt['model']['loss'] == 'ce1_dice2':
        loss_fun = CE1Dice2Loss(num_classes=num_classes)
    # Add other loss types if needed, e.g., for 'ce_scl'
    elif opt['model']['loss'] == 'multi_class_cd':
        loss_fun = MultiClassCDLoss(num_classes=num_classes, loss_weights=opt['model'].get('loss_weights'))
    # elif opt['model']['loss'] == 'ce_scl':
    #     loss_fun = CEDiceLoss(num_classes=num_classes) # Or a specific SCL loss class
    else:
        raise ValueError(f"Unsupported loss function type: {opt['model']['loss']}")

    # If loss_fun is an nn.Module, move it to the device
    if isinstance(loss_fun, nn.Module):
        loss_fun.to(device)

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
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Reduce gradient accumulation for memory savings
            accumulation_steps = 2  # Effective batch size = 1 * 2 = 2
            
            # Set memory fraction to avoid fragmentation (more conservative)
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            for current_step, train_data in enumerate(train_loader):
                # Aggressive memory cleanup at start of each step
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Move data to GPU manually
                train_im1 = train_data['A'].to(device)
                train_im2 = train_data['B'].to(device)
                # Robust label extraction and move to device
                seg_t1 = (train_data['L1'] if 'L1' in train_data else train_data['L']).to(device)
                seg_t2 = (train_data['L2'] if 'L2' in train_data else train_data['L']).to(device)
                change = (train_data['change'] if 'change' in train_data else train_data['L']).to(device)

                # Use gradient checkpointing to save memory
                with torch.cuda.amp.autocast():  # Mixed precision
                    outputs = cd_model(train_im1, train_im2)
                
                # Clear input tensors from memory immediately after forward pass
                del train_im1, train_im2
                torch.cuda.empty_cache()

                if isinstance(loss_fun, MultiClassCDLoss):
                    labels = {'seg_t1': seg_t1, 'seg_t2': seg_t2, 'change': change}
                    with torch.cuda.amp.autocast():
                        train_loss, loss_dict = loss_fun(outputs, labels)
                    # Scale loss for gradient accumulation
                    train_loss = train_loss / accumulation_steps
                    seg_logits_t1, seg_logits_t2, change_pred = outputs
                    with torch.no_grad():
                        pred_seg_t1 = torch.argmax(seg_logits_t1, dim=1)
                        pred_seg_t2 = torch.argmax(seg_logits_t2, dim=1)
                        pred_change = torch.argmax(change_pred, dim=1)
                    
                    # Log masks to wandb (log only for the first batch of each epoch to avoid excessive logging)
                    if current_step == 0 and current_epoch % 1 == 0:
                        
                    
                    
                        wandb.log({
                            "train/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=opt['model']['n_classes']), caption="Pred Seg T1 (multi-class)")],
                            "train/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=opt['model']['n_classes']), caption="Pred Seg T2 (multi-class)")],
                            "train/pred_change": [wandb.Image(create_color_mask(pred_change[0], num_classes=opt['model']['n_classes'] * opt['model']['n_classes']), caption="Pred Change (multi-class)")],
                        
                            # These are already colorized — just send directly
                            "train/gt_seg_t1": [wandb.Image(create_color_mask(seg_t1[0], num_classes=opt['model']['n_classes']), caption="GT Seg T1 (multi-class)")],
                            "train/gt_seg_t2": [wandb.Image(create_color_mask(seg_t2[0], num_classes=opt['model']['n_classes']), caption="GT Seg T2 (multi-class)")],
                            "train/gt_change": [wandb.Image(create_color_mask(change[0], num_classes=opt['model']['n_classes'] * opt['model']['n_classes']), caption="GT Change (multi-class)")],

                        
                            "global_step": current_epoch * len(train_loader) + current_step
                        })


                else:
                    # Assumes binary loss on the change prediction head
                    change_pred = outputs[2] if isinstance(outputs, tuple) and len(outputs) > 2 else outputs
                    with torch.cuda.amp.autocast():
                        train_loss = loss_fun(change_pred, change)
                    # Scale loss for gradient accumulation
                    train_loss = train_loss / accumulation_steps
                    # Create a dummy loss_dict for logging consistency
                    loss_dict = {'seg_t1': 0, 'seg_t2': 0, 'change': train_loss.item()}
                    seg_logits_t1 = seg_logits_t2 = torch.zeros_like(change_pred)  # dummy for logging
                
                # Convert logits to predicted masks for logging
                with torch.no_grad():
                    pred_seg_t1 = torch.argmax(seg_logits_t1, dim=1)
                    pred_seg_t2 = torch.argmax(seg_logits_t2, dim=1)
                    pred_change = torch.argmax(change_pred, dim=1)
                
                # Log masks to wandb (log only for the first batch of each epoch to avoid excessive logging)
                if current_step == 0 and current_epoch % 1 == 0:
                    wandb.log({
                        "train/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=num_classes), caption="Pred Seg T1 (multi-class)")],
                        "train/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=num_classes), caption="Pred Seg T2 (multi-class)")],
                        "train/pred_change": [wandb.Image(create_color_mask(pred_change[0], num_classes=num_classes * num_classes), caption="Pred Change (multi-class)")],
                        "train/gt_seg_t1": [wandb.Image(create_color_mask(seg_t1[0], num_classes=num_classes), caption="GT Seg T1")],
                        "train/gt_seg_t2": [wandb.Image(create_color_mask(seg_t2[0], num_classes=num_classes), caption="GT Seg T2")],
                        "train/gt_change": [wandb.Image(create_color_mask(change[0], num_classes=num_classes * num_classes), caption="GT Change")],
                        "global_step": current_epoch * len(train_loader) + current_step
                    })
                
                # Gradient accumulation with mixed precision
                scaler.scale(train_loss).backward()
                
                if (current_step + 1) % accumulation_steps == 0 or (current_step + 1) == len(train_loader):
                    scaler.step(optimer)
                    scaler.update()
                    optimer.zero_grad()
                    # Clear gradients from memory
                    torch.cuda.empty_cache()
                    
                # Clean up memory after each batch (avoid double deletion)
                del seg_t1, seg_t2, change, outputs
                if 'pred_seg_t1' in locals():
                    del pred_seg_t1, pred_seg_t2, pred_change
                torch.cuda.empty_cache()
                
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
                binary_pred = (from_class != to_class).int()
                
                # Convert ground truth to binary (0 = no change, 1 = change)
                # Assuming 'change' contains class transition IDs or is already binary
                gt_np = (change.detach().cpu().numpy() > 0).astype(np.uint8)
                pred_np = binary_pred.cpu().numpy()

                if current_step % 100 == 0:
                    print("DEBUG: Unique predictions (binary):", np.unique(pred_np))
                    print("DEBUG: Unique GT (binary):", np.unique(gt_np))
                    print("DEBUG: Metric num_classes:", metric.n_class)

                current_score = metric.update_cm(pr=pred_np, gt=gt_np)
                log_dict['running_acc'] = current_score.item()
                wandb.log({'train_loss': train_loss.item(), 'train_running_acc': current_score.item()})

                # Logging
                if current_step % opt['train']['train_print_iter'] == 0:
                    message = '[Training CD]. epoch: [%d/%d]. Itter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f\n' % (
                        current_epoch, opt['train']['n_epoch'], current_step, len(train_loader), train_loss.item(),
                        current_score.item())
                    logger.info(message)

            ### Epoch Summary ###
            scores = metric.get_scores()
            epoch_acc = scores['mf1']
            # ... (rest of the code remains the same)

            cd_model.load_state_dict(torch.load(gen_path), strict=True)
            cd_model.to(device)
            metric.clear()
            cd_model.eval()
            with torch.no_grad():
                for current_step, test_data in enumerate(test_loader):
                    test_img1 = test_data['A'].to(device)
                    test_img2 = test_data['B'].to(device)
                    # Robust label extraction - data automatically on correct device
                    if 'L1' in test_data:
                        seg_t1 = test_data['L1']
                        seg_t2 = test_data['L2']
                        change = test_data['change']
                    else:
                        # Fallback for older data format
                        seg_t1 = test_data['L']  # Assuming 'L' is the label
                        seg_t2 = test_data['L']  # You might need to adjust this
                        change = test_data['L']  # You might need to adjust this

                    outputs = cd_model(test_img1, test_img2)
                    # Only use change head for metric and visuals
                    change_pred = outputs[2]  # [B, num_classes*num_classes, H, W]
                    G_pred = torch.argmax(change_pred.detach(), dim=1)
                    # ... (rest of the code remains the same)
                    # Convert prediction to binary change mask
                    n_classes = opt['model']['n_classes']
                    from_class = G_pred // n_classes
                    to_class = G_pred % n_classes
                    binary_pred = (from_class != to_class).int()
                    
                    # Get ground truth
                    gt = test_data['change'].to(device) if 'change' in test_data else test_data['L'].to(device)
                    
                    # Convert ground truth to binary for metrics
                    gt_binary = (gt > 0).int()
                    
                    # Update confusion matrix with binary predictions
                    current_score = metric.update_cm(pr=binary_pred.cpu().numpy(), gt=gt_binary.detach().cpu().numpy())
                    log_dict['running_acc'] = current_score.item()

                    logs = log_dict
                    message = '[Testing CD]. Itter: [%d/%d], running_mf1: %.5f\n' % \
                              (current_step, len(test_loader), logs['running_acc'])
                    logger_test.info(message)

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
                        gt_cm = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8,
                                                   min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1),
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
                logger_test.info(message)
                logger.info('End of testing...')

            if current_step == 0 and current_epoch % 1 == 0:
                # Reuse create_color_mask function for validation visualizations
                wandb.log({
                    "val/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0]), caption="Val Pred Seg T1 (multi-class)")],
                    "val/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0]), caption="Val Pred Seg T2 (multi-class)")],
                    "val/pred_change": [wandb.Image(create_color_mask(pred_change[0], num_classes=opt['model']['n_classes']*opt['model']['n_classes']), caption="Val Pred Change (multi-class)")],
                    "val/gt_seg_t1": [wandb.Image(create_color_mask(seg_t1[0]), caption="Val GT Seg T1 (multi-class)")],
                    "val/gt_seg_t2": [wandb.Image(create_color_mask(seg_t2[0]), caption="Val GT Seg T2 (multi-class)")],
                    "val/gt_change": [wandb.Image(create_color_mask(change[0], num_classes=opt['model']['n_classes']*opt['model']['n_classes']), caption="Val GT Change (multi-class)")],
                    "global_step": current_epoch * len(val_loader) + current_step
                })
