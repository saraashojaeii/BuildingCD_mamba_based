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

def create_color_mask(tensor, num_classes=10):
    """
    Converts a 2D tensor or array of class labels into a color RGB image.
    """
    # Convert to numpy if it's a tensor
    if isinstance(tensor, torch.Tensor):
        mask = tensor.detach().cpu().numpy()
    else:
        mask = tensor
    
    # Create a colormap
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    # Create RGB image
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        class_pixels = (mask == class_id)
        rgb_mask[class_pixels] = (np.array(colors(class_id)[:3]) * 255).astype(np.uint8)
    
    return rgb_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        default='config/second_cdmamba/second_cdmamba.json',
                        help='JSON file for configuration')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'], help='Run either train(training + validation) or testing')
    parser.add_argument('--gpu_ids', type=str, default=None)
    parser.add_argument('-log_eval', action='store_true')

    # Parse config
    args = parser.parse_args()
    opt = Logger.parse(args)

    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # Initialize Accelerator for multi-GPU training
    accelerator = Accelerator()
    device = accelerator.device

    # Logging (only on main process)
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
            wandb.init(project=opt['wandb']['project'], config=opt)
        else:
            wandb.init(mode="disabled")
    else:
        # Create dummy logger for non-main processes
        logger = logging.getLogger('base')
        logger.addHandler(logging.NullHandler())
        wandb.init(mode="disabled")

    # Debug GPU assignment
    if accelerator.is_main_process:
        print(f"Accelerator device: {accelerator.device}")
        print(f"Process index: {accelerator.process_index}")
        print(f"Local process index: {accelerator.local_process_index}")
        print(f"Num processes: {accelerator.num_processes}")

    # Dataset creation
    train_loader = val_loader = test_loader = None
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            if accelerator.is_main_process:
                print("Create [train] change-detection dataloader")
            train_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            train_loader = Data.create_cd_dataloader(train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

        elif phase == 'val' and args.phase != 'test':
            if accelerator.is_main_process:
                print("Create [val] change-detection dataloader")
            val_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            val_loader = Data.create_cd_dataloader(val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)

        elif phase == 'test':
            if accelerator.is_main_process:
                print("Create [test] change-detection dataloader")
            test_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)

    if accelerator.is_main_process:
        logger.info('Initial Dataset Finished')

    # Create CD model
    cd_model = Model.create_CD_model(opt)
    num_classes = opt['model']['n_classes']
    
    if accelerator.is_main_process:
        logger.info(f"Number of classes for loss function: {num_classes}")

    # Create criterion
    if opt['model']['loss'] == 'ce_dice':
        loss_fun = CEDiceLoss(num_classes=num_classes)
    elif opt['model']['loss'] == 'ce':
        loss_fun = cross_entropy_loss_fn 
    elif opt['model']['loss'] == 'dice':
        loss_fun = DiceOnlyLoss(num_classes=num_classes)
    elif opt['model']['loss'] == 'ce2_dice1':
        loss_fun = CE2Dice1Loss(num_classes=num_classes)
    elif opt['model']['loss'] == 'ce1_dice2':
        loss_fun = CE1Dice2Loss(num_classes=num_classes)
    elif opt['model']['loss'] == 'multi_class_cd':
        loss_fun = MultiClassCDLoss(num_classes=num_classes, loss_weights=opt['model'].get('loss_weights'))
    else:
        raise ValueError(f"Unsupported loss function type: {opt['model']['loss']}")

    # Move loss function to device if it's a module
    if isinstance(loss_fun, nn.Module):
        loss_fun.to(device)

    # Create optimizer
    if opt['train']["optimizer"]["type"] == 'adam':
        optimizer = optim.Adam(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'adamw':
        optimizer = optim.AdamW(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'sgd':
        optimizer = optim.SGD(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"],
                            momentum=0.9, weight_decay=5e-4)

    # Prepare model, optimizer, and dataloaders with Accelerator
    if train_loader is not None and val_loader is not None:
        cd_model, optimizer, train_loader, val_loader = accelerator.prepare(
            cd_model, optimizer, train_loader, val_loader
        )
    elif train_loader is not None:
        cd_model, optimizer, train_loader = accelerator.prepare(
            cd_model, optimizer, train_loader
        )
    elif test_loader is not None:
        cd_model, test_loader = accelerator.prepare(cd_model, test_loader)
    else:
        cd_model = accelerator.prepare(cd_model)

    # Debug info after preparation
    if accelerator.is_main_process:
        print(f"Model prepared on device: {next(cd_model.parameters()).device}")
        print(f"Accelerator device: {accelerator.device}")
    
    # Wait for all processes after preparation
    accelerator.wait_for_everyone()

    # Initialize metrics
    metric = ConfuseMatrixMeter(n_class=2)
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
            if accelerator.is_main_process:
                os.makedirs(train_result_path, exist_ok=True)

            #################
            #    Training   #
            #################
            if accelerator.is_main_process:
                message = 'lr: %0.7f\n \n' % optimizer.param_groups[0]['lr']
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
                            "train/pred_change": [wandb.Image(create_color_mask(pred_change[0], num_classes=opt['model']['n_classes']*opt['model']['n_classes']), caption="Pred Change (multi-class)")],
                            "train/gt_seg_t1": [wandb.Image(create_color_mask(seg_t1[0], num_classes=opt['model']['n_classes']), caption="GT Seg T1 (multi-class)")],
                            "train/gt_seg_t2": [wandb.Image(create_color_mask(seg_t2[0], num_classes=opt['model']['n_classes']), caption="GT Seg T2 (multi-class)")],
                            "train/gt_change": [wandb.Image(create_color_mask(change[0], num_classes=opt['model']['n_classes']*opt['model']['n_classes']), caption="GT Change (multi-class)")],
                            "global_step": current_epoch * len(train_loader) + current_step
                        })

                else:
                    # Assumes binary loss on the change prediction head
                    change_pred = outputs[2] if isinstance(outputs, tuple) and len(outputs) > 2 else outputs
                    train_loss = loss_fun(change_pred, change)
                    loss_dict = {'seg_t1': 0, 'seg_t2': 0, 'change': train_loss.item()}
                    with torch.no_grad():
                        pred_seg_t1 = torch.argmax(outputs[0], dim=1) if isinstance(outputs, tuple) else None
                        pred_seg_t2 = torch.argmax(outputs[1], dim=1) if isinstance(outputs, tuple) else None
                        pred_change = torch.argmax(change_pred, dim=1)
                
                # Log masks to wandb (log only for the first batch of each epoch to avoid excessive logging)
                if current_step == 0 and current_epoch % 1 == 0 and accelerator.is_main_process:
                    if pred_seg_t1 is not None and pred_seg_t2 is not None:
                        wandb.log({
                            "train/pred_seg_t1": [wandb.Image(pred_seg_t1[0].cpu().numpy()*255, caption="Pred Seg T1")],
                            "train/pred_seg_t2": [wandb.Image(pred_seg_t2[0].cpu().numpy()*255, caption="Pred Seg T2")],
                            "train/pred_change": [wandb.Image(pred_change[0].cpu().numpy()*255, caption="Pred Change")],
                            "train/gt_seg_t1": [wandb.Image(seg_t1[0].cpu().numpy()*255, caption="GT Seg T1")],
                            "train/gt_seg_t2": [wandb.Image(seg_t2[0].cpu().numpy()*255, caption="GT Seg T2")],
                            "train/gt_change": [wandb.Image(change[0].cpu().numpy()*255, caption="GT Change")],
                            "global_step": current_epoch * len(train_loader) + current_step
                        })
                
                optimizer.zero_grad()
                accelerator.backward(train_loss)
                optimizer.step()
                
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
                gt_binary = (change > 0).int()
                gt_np = gt_binary.detach().cpu().numpy().astype(np.uint8)
                pred_np = binary_pred.cpu().numpy()

                current_score = metric.update_cm(pr=pred_np, gt=gt_np)
                log_dict['running_acc'] = current_score.item()
                
                if accelerator.is_main_process:
                    wandb.log({'train_loss': train_loss.item(), 'train_running_acc': current_score.item()})

                # Logging
                if current_step % opt['train']['train_print_iter'] == 0 and accelerator.is_main_process:
                    message = '[Training CD]. epoch: [%d/%d]. Itter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f\n' % (
                        current_epoch, opt['train']['n_epoch'], current_step, len(train_loader), train_loss.item(),
                        current_score.item())
                    logger.info(message)

            ### Epoch Summary ###
            scores = metric.get_scores()
            epoch_acc = scores['mf1']
            log_dict['epoch_acc'] = epoch_acc.item()
            epoch_losses.append(epoch_loss / len(train_loader))
            
            for k, v in scores.items():
                log_dict[k] = v

            # Log training summary to wandb (only on main process)
            if accelerator.is_main_process:
                wandb.log({
                    'epoch': current_epoch, 
                    'train_epoch_loss': epoch_loss / len(train_loader), 
                    'train_epoch_mF1': log_dict['epoch_acc']
                })

                message = '[Training CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' % (
                    current_epoch, opt['train']['n_epoch'] - 1, log_dict['epoch_acc'])
                for k, v in log_dict.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                message += '\n'
                logger.info(message)

            metric.clear()

            ##################
            ### validation ###
            ##################
            cd_model.eval()
            with torch.no_grad():
                if current_epoch % opt['train']['val_freq'] == 0:
                    val_result_path = '{}/val/{}'.format(opt['path_cd']['result'], current_epoch)
                    if accelerator.is_main_process:
                        os.makedirs(val_result_path, exist_ok=True)

                    for current_step, val_data in enumerate(val_loader):
                        val_img1 = val_data['A']
                        val_img2 = val_data['B']
                        seg_t1 = val_data['L1'] if 'L1' in val_data else val_data['L']
                        seg_t2 = val_data['L2'] if 'L2' in val_data else val_data['L']
                        change = val_data['change'] if 'change' in val_data else val_data['L']

                        outputs = cd_model(val_img1, val_img2)

                        if isinstance(loss_fun, MultiClassCDLoss):
                            labels = {'seg_t1': seg_t1, 'seg_t2': seg_t2, 'change': change}
                            val_loss, loss_dict = loss_fun(outputs, labels)
                        else:
                            # Assumes binary loss on the change prediction head
                            change_pred = outputs[2] if isinstance(outputs, tuple) and len(outputs) > 2 else outputs
                            val_loss = loss_fun(change_pred, change)
                            loss_dict = {'seg_t1': 0, 'seg_t2': 0, 'change': val_loss.item()}
                        
                        log_dict['loss'] = val_loss.item()
                        log_dict['loss_seg_t1'] = loss_dict['seg_t1']
                        log_dict['loss_seg_t2'] = loss_dict['seg_t2']
                        log_dict['loss_change'] = loss_dict['change']
                        
                        # For metric, convert transition prediction to binary change map
                        change_pred = outputs[2]  # [B, num_classes*num_classes, H, W]
                        G_pred = torch.argmax(change_pred.detach(), dim=1)

                        n_classes = opt['model']['n_classes']
                        from_class = G_pred // n_classes
                        to_class = G_pred % n_classes
                        binary_pred = (from_class != to_class).int()

                        # Convert ground truth to binary (0 = no change, 1 = change)
                        gt_binary = (change > 0).int()
                        gt_np = gt_binary.detach().cpu().numpy().astype(np.uint8)
                        pred_np = binary_pred.cpu().numpy()
                        current_score = metric.update_cm(pr=pred_np, gt=gt_np)
                        log_dict['running_acc'] = current_score.item()
                        
                        if accelerator.is_main_process:
                            wandb.log({'val_loss': val_loss.item(), 'val_running_acc': current_score.item()})

                        # log running batch status for val data
                        if current_step % opt['train']['val_print_iter'] == 0 and accelerator.is_main_process:
                            logs = log_dict
                            message = '[Validation CD]. epoch: [%d/%d]. Itter: [%d/%d], running_mf1: %.5f\n' % \
                                      (current_epoch, opt['train']['n_epoch'] - 1, current_step, len(val_loader), logs['running_acc'])
                            logger.info(message)

                    ### log epoch status ###
                    scores = metric.get_scores()
                    epoch_acc = scores['mf1']
                    log_dict['epoch_acc'] = epoch_acc.item()
                    for k, v in scores.items():
                        log_dict[k] = v
                    logs = log_dict
                    
                    if accelerator.is_main_process:
                        message = '[Validation CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' % \
                                  (current_epoch, opt['train']['n_epoch'], logs['epoch_acc'])
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                        message += '\n'
                        logger.info(message)

                        # best model (only save on main process)
                        if logs['epoch_acc'] > best_mF1:
                            is_best_model = True
                            best_mF1 = logs['epoch_acc']
                            logger.info('[Validation CD] Best model updated. Saving the models (current + best) and training states.')
                            # Save model - unwrap the model from accelerator
                            unwrapped_model = accelerator.unwrap_model(cd_model)
                            save_network(opt, current_epoch, unwrapped_model, optimizer, is_best_model)
                        else:
                            is_best_model = False
                            logger.info('[Validation CD] Saving the current cd model and training states.')
                        logger.info('--- Proceed To The Next Epoch ----\n \n')

                    metric.clear()

            # Update scheduler
            get_scheduler(optimizer=optimizer, args=opt['train']).step()
            
            # Wait for all processes to sync before next epoch
            accelerator.wait_for_everyone()
            
        if accelerator.is_main_process:
            logger.info('End of training.')
            np.save(os.path.join(opt['path_cd']['result'], 'train_losses.npy'), np.array(epoch_losses))

    else:
        # Testing phase
        if accelerator.is_main_process:
            logger.info('Begin Model Evaluation (testing).')
            test_result_path = '{}/test/'.format(opt['path_cd']['result'])
            os.makedirs(test_result_path, exist_ok=True)
            logger_test = logging.getLogger('test')  # test logger

        # Load network
        load_path = opt["path_cd"]["resume_state"]
        if accelerator.is_main_process:
            print(load_path)
        
        if load_path is not None:
            if accelerator.is_main_process:
                logger.info('Loading pretrained model for CD model [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            
            # Load model weights
            cd_model.load_state_dict(torch.load(gen_path, map_location=device), strict=True)
            
        metric.clear()
        cd_model.eval()
        
        with torch.no_grad():
            for current_step, test_data in enumerate(test_loader):
                test_img1 = test_data['A']
                test_img2 = test_data['B']
                seg_t1 = test_data['L1'] if 'L1' in test_data else test_data['L']
                seg_t2 = test_data['L2'] if 'L2' in test_data else test_data['L']
                change = test_data['change'] if 'change' in test_data else test_data['L']

                outputs = cd_model(test_img1, test_img2)
                # Only use change head for metric and visuals
                change_pred = outputs[2]  # [B, num_classes*num_classes, H, W]
                G_pred = torch.argmax(change_pred.detach(), dim=1)
                
                # Convert prediction to binary change mask
                n_classes = opt['model']['n_classes']
                from_class = G_pred // n_classes
                to_class = G_pred % n_classes
                binary_pred = (from_class != to_class).int()

                # Convert ground truth to binary (0 = no change, 1 = change)
                gt_binary = (change > 0).int()
                gt_np = gt_binary.detach().cpu().numpy().astype(np.uint8)
                pred_np = binary_pred.cpu().numpy()
                current_score = metric.update_cm(pr=pred_np, gt=gt_np)

                if accelerator.is_main_process and current_step % 50 == 0:
                    print(f"Test step {current_step}/{len(test_loader)}, Current mF1: {current_score.item():.4f}")

        ### log epoch status ###
        scores = metric.get_scores()
        epoch_acc = scores['mf1']
        log_dict['epoch_acc'] = epoch_acc.item()
        for k, v in scores.items():
            log_dict[k] = v
        logs = log_dict
        
        if accelerator.is_main_process:
            message = '[Test CD summary]: Test mF1=%.5f \n' % (logs['epoch_acc'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                message += '\n'
            logger_test.info(message)
            logger.info('End of testing...')

if __name__ == '__main__':
    main()
