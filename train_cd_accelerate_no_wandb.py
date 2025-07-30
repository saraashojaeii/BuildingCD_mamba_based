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
from accelerate import Accelerator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Try to import wandb, but continue without it if there's an import error
try:
    import wandb
    WANDB_AVAILABLE = True
    print("wandb imported successfully")
except ImportError as e:
    print(f"Warning: wandb import failed: {e}")
    print("Continuing without wandb logging...")
    WANDB_AVAILABLE = False

def create_color_mask(mask, num_classes=7):
    """Create a color mask for visualization"""
    import matplotlib.cm as cm
    colors = cm.get_cmap('tab10')
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

    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Only print on main process
    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPUs for training")
        print(f"Mixed precision: {accelerator.mixed_precision}")

    # Initialize wandb only on main process and if available
    if accelerator.is_main_process and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=opt.get('wandb_project', 'change_detection'),
                name=opt.get('wandb_run_name', 'multi_gpu_training'),
                config=opt
            )
            print("wandb initialized successfully")
        except Exception as e:
            print(f"Warning: wandb initialization failed: {e}")
            WANDB_AVAILABLE = False

    # Create datasets
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            if accelerator.is_main_process:
                print('Dataset [{:s}] is created.'.format(dataset_opt['name']))
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
            if accelerator.is_main_process:
                print('Dataset [{:s}] is created.'.format(dataset_opt['name']))
        elif phase == 'test':
            test_set = Data.create_dataset(dataset_opt, phase)
            test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
            if accelerator.is_main_process:
                print('Dataset [{:s}] is created.'.format(dataset_opt['name']))

    # Create model
    cd_model = Model.create_model(opt)
    if accelerator.is_main_process:
        print("Model created")

    # Create optimizer
    if opt['train']["optimizer"]["type"] == 'adam':
        optimizer = optim.Adam(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'adamw':
        optimizer = optim.AdamW(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'sgd':
        optimizer = optim.SGD(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"],
                            momentum=0.9, weight_decay=5e-4)

    # Prepare model, optimizer, and dataloaders with accelerator
    cd_model, optimizer, train_loader = accelerator.prepare(cd_model, optimizer, train_loader)
    
    if 'val' in opt['datasets']:
        val_loader = accelerator.prepare(val_loader)
    if 'test' in opt['datasets']:
        test_loader = accelerator.prepare(test_loader)

    if accelerator.is_main_process:
        print("Model, optimizer, and dataloaders prepared with accelerator")

    # Create loss functions
    if opt['train']['loss_type'] == 'multi_class':
        cd_criterion = nn.CrossEntropyLoss()
    else:
        cd_criterion = nn.BCEWithLogitsLoss()

    # Create scheduler
    scheduler = get_scheduler(optimizer, opt['train'])

    # Create metrics
    running_metric = ConfuseMatrixMeter(n_class=opt['model']['out_nc'])

    # Training phase
    if opt['phase'] == 'train':
        if accelerator.is_main_process:
            print('Start training from epoch 0')
        
        for current_epoch in range(0, opt['train']['n_epoch']):
            if accelerator.is_main_process:
                print(f"......Begin Training Epoch {current_epoch}......")
            
            running_metric.clear()
            cd_model.train()
            
            epoch_loss = 0.0
            num_batches = 0
            
            for current_step, train_data in enumerate(train_loader):
                # Data is automatically moved to the correct device by accelerator
                train_im1 = train_data['A']
                train_im2 = train_data['B']
                
                # Robust label extraction
                if 'seg_t1' in train_data:
                    seg_t1 = train_data['seg_t1']
                elif 'L' in train_data:
                    seg_t1 = train_data['L']
                else:
                    raise KeyError("Neither 'seg_t1' nor 'L' found in train_data")

                optimizer.zero_grad()
                
                # Forward pass
                cd_preds = cd_model(train_im1, train_im2)
                
                # Calculate loss
                if opt['train']['loss_type'] == 'multi_class':
                    cd_loss = cd_criterion(cd_preds, seg_t1.long())
                else:
                    cd_loss = cd_criterion(cd_preds, seg_t1.float())
                
                # Backward pass using accelerator
                accelerator.backward(cd_loss)
                optimizer.step()
                
                epoch_loss += cd_loss.item()
                num_batches += 1
                
                # Update metrics
                if opt['train']['loss_type'] == 'multi_class':
                    cd_pred = torch.argmax(cd_preds, dim=1)
                else:
                    cd_pred = (torch.sigmoid(cd_preds) > 0.5).float()
                
                running_metric.update_cm(pr=cd_pred.cpu().numpy(), gt=seg_t1.cpu().numpy())
                
                # Log every 100 steps on main process
                if current_step % 100 == 0 and accelerator.is_main_process:
                    print(f'Epoch [{current_epoch}/{opt["train"]["n_epoch"]}], '
                          f'Step [{current_step}/{len(train_loader)}], '
                          f'Loss: {cd_loss.item():.4f}')

            # Wait for all processes to finish the epoch
            accelerator.wait_for_everyone()
            
            # Calculate epoch metrics on main process
            if accelerator.is_main_process:
                avg_loss = epoch_loss / num_batches
                scores = running_metric.get_scores()
                
                print(f'Epoch [{current_epoch}] - Loss: {avg_loss:.4f}')
                print(f'Training Metrics: IoU: {scores["iou"]:.4f}, F1: {scores["f1"]:.4f}')
                
                # Log to wandb if available
                if WANDB_AVAILABLE:
                    try:
                        log_dict = {
                            'epoch': current_epoch,
                            'train/loss': avg_loss,
                            'train/iou': scores["iou"],
                            'train/f1': scores["f1"],
                            'train/precision': scores["precision"],
                            'train/recall': scores["recall"]
                        }
                        wandb.log(log_dict)
                    except Exception as e:
                        print(f"Warning: wandb logging failed: {e}")

            # Validation phase
            if 'val' in opt['datasets'] and current_epoch % opt['train']['val_freq'] == 0:
                if accelerator.is_main_process:
                    print('......Begin Validation......')
                
                cd_model.eval()
                val_metric = ConfuseMatrixMeter(n_class=opt['model']['out_nc'])
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for val_step, val_data in enumerate(val_loader):
                        val_im1 = val_data['A']
                        val_im2 = val_data['B']
                        
                        if 'seg_t1' in val_data:
                            val_seg_t1 = val_data['seg_t1']
                        elif 'L' in val_data:
                            val_seg_t1 = val_data['L']
                        else:
                            raise KeyError("Neither 'seg_t1' nor 'L' found in val_data")
                        
                        val_cd_preds = cd_model(val_im1, val_im2)
                        
                        if opt['train']['loss_type'] == 'multi_class':
                            val_cd_loss = cd_criterion(val_cd_preds, val_seg_t1.long())
                            val_cd_pred = torch.argmax(val_cd_preds, dim=1)
                        else:
                            val_cd_loss = cd_criterion(val_cd_preds, val_seg_t1.float())
                            val_cd_pred = (torch.sigmoid(val_cd_preds) > 0.5).float()
                        
                        val_loss += val_cd_loss.item()
                        val_batches += 1
                        val_metric.update_cm(pr=val_cd_pred.cpu().numpy(), gt=val_seg_t1.cpu().numpy())

                # Wait for all processes
                accelerator.wait_for_everyone()
                
                if accelerator.is_main_process:
                    val_scores = val_metric.get_scores()
                    avg_val_loss = val_loss / val_batches
                    
                    print(f'Validation - Loss: {avg_val_loss:.4f}')
                    print(f'Validation Metrics: IoU: {val_scores["iou"]:.4f}, F1: {val_scores["f1"]:.4f}')
                    
                    # Log validation metrics
                    if WANDB_AVAILABLE:
                        try:
                            val_log_dict = {
                                'epoch': current_epoch,
                                'val/loss': avg_val_loss,
                                'val/iou': val_scores["iou"],
                                'val/f1': val_scores["f1"],
                                'val/precision': val_scores["precision"],
                                'val/recall': val_scores["recall"]
                            }
                            wandb.log(val_log_dict)
                        except Exception as e:
                            print(f"Warning: wandb logging failed: {e}")

            # Update scheduler
            scheduler.step()
            
            # Save model checkpoint on main process
            if current_epoch % opt['train']['save_freq'] == 0 and accelerator.is_main_process:
                save_path = os.path.join(opt['path']['checkpoint'], f'epoch_{current_epoch}.pth')
                # Unwrap model before saving
                unwrapped_model = accelerator.unwrap_model(cd_model)
                save_network(unwrapped_model, save_path)
                print(f'Model saved at epoch {current_epoch}')

    # Testing phase
    elif opt['phase'] == 'test':
        if accelerator.is_main_process:
            print('......Begin Testing......')
        
        cd_model.eval()
        test_metric = ConfuseMatrixMeter(n_class=opt['model']['out_nc'])
        
        with torch.no_grad():
            for test_step, test_data in enumerate(test_loader):
                test_im1 = test_data['A']
                test_im2 = test_data['B']
                
                if 'seg_t1' in test_data:
                    test_seg_t1 = test_data['seg_t1']
                elif 'L' in test_data:
                    test_seg_t1 = test_data['L']
                else:
                    raise KeyError("Neither 'seg_t1' nor 'L' found in test_data")
                
                test_cd_preds = cd_model(test_im1, test_im2)
                
                if opt['train']['loss_type'] == 'multi_class':
                    test_cd_pred = torch.argmax(test_cd_preds, dim=1)
                else:
                    test_cd_pred = (torch.sigmoid(test_cd_preds) > 0.5).float()
                
                test_metric.update_cm(pr=test_cd_pred.cpu().numpy(), gt=test_seg_t1.cpu().numpy())

        # Wait for all processes
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            test_scores = test_metric.get_scores()
            print('Test Results:')
            print(f'IoU: {test_scores["iou"]:.4f}')
            print(f'F1: {test_scores["f1"]:.4f}')
            print(f'Precision: {test_scores["precision"]:.4f}')
            print(f'Recall: {test_scores["recall"]:.4f}')
            
            if WANDB_AVAILABLE:
                try:
                    test_log_dict = {
                        'test/iou': test_scores["iou"],
                        'test/f1': test_scores["f1"],
                        'test/precision': test_scores["precision"],
                        'test/recall': test_scores["recall"]
                    }
                    wandb.log(test_log_dict)
                except Exception as e:
                    print(f"Warning: wandb logging failed: {e}")

    # Finish wandb run
    if accelerator.is_main_process and WANDB_AVAILABLE:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: wandb finish failed: {e}")

if __name__ == '__main__':
    main()
