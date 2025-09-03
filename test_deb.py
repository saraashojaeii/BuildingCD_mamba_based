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
    parser.add_argument('--weights', type=str, required=True, help='/root/home/pvc/Building_changedetection_job/experiments')
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
    seed_suffix = f"seed{args.seed}" if getattr(args, "seed", None) is not None else ""
    parts = [p for p in [dataset_suffix, tag_suffix, seed_suffix] if p]
    suffix = "_".join(parts)
    exp_folder = f"{suffix}_{exp_timestamp}" if suffix else f"{exp_timestamp}"


    # ---------- CHANGE 4: guard and create stamped dirs ----------
    if 'path_cd' in opt and isinstance(opt['path_cd'], dict):
        for k in ['log', 'result', 'checkpoint']:
            if k in opt['path_cd'] and isinstance(opt['path_cd'][k], str):
                base_dir = opt['path_cd'][k]
                stamped = os.path.join(base_dir, exp_folder)
                opt['path_cd'][k] = stamped
                os.makedirs(stamped, exist_ok=True)
        opt['path_cd']['exp_folder'] = exp_folder
    else:
        print("[warn] opt['path_cd'] not found; skipping folder stamping")

    set_seed(args.seed if args.seed is not None else 42)

    # ---------- CHANGE 6: print resolved header ----------
    print("[run] name:", opt.get('name', 'experiment'))
    print("[run] phase:", args.phase, "| tag:", args.tag, "| seed:", args.seed)
    print("[run] change_threshold (eval only):", args.change_threshold)
    print("[run] exp_folder:", opt.get('path_cd', {}).get('exp_folder', '<none>'))


    #logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    # Logger setup for test phase only
    Logger.setup_logger(logger_name='test', root=opt['path_cd']['log'], phase='test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Optional: Seed for reproducibility
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(args.seed if args.seed is not None else 42)

    # Create only the test dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            print("Create [test] change-detection dataloader")
            test_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase, seed_worker, g)
            opt['len_test_dataloader'] = len(test_loader)

        # elif phase == 'test' and args.phase == 'test':
        elif phase == 'test':
            print("Creat [test] change-detection dataloader")
            test_set = Data.create_scd_dataset(dataset_opt=dataset_opt, phase=phase)
            test_loader = Data.create_cd_dataloader(test_set, dataset_opt, phase, seed_worker, g)
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
        beta1 = opt['train']["optimizer"].get("beta1", 0.9)  # fallback default
        beta2 = opt['train']["optimizer"].get("beta2", 0.999)
        optimizer = optim.Adam(
            cd_model.parameters(),
            lr=opt['train']["optimizer"]["lr"],
            betas=(beta1, beta2)
        )
    elif opt['train']["optimizer"]["type"] == 'adamw':
        optimizer = optim.AdamW(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"])
    elif opt['train']["optimizer"]["type"] == 'sgd':
        optimizer = optim.SGD(cd_model.parameters(), lr=opt['train']["optimizer"]["lr"],
                            momentum=0.9, weight_decay=5e-4)


    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
        
    metric = ConfuseMatrixMeter(n_class=2)  # For binary change detection
    metric_seg = ConfuseMatrixMeter(n_class=opt['model']['n_classes'])  # For 6-class segmentation
    log_dict = OrderedDict()


if torch.cuda.is_available():
    try:
        torch.cuda.set_per_process_memory_fraction(0.8)
    except Exception as e:
        logger.warning(f"set_per_process_memory_fraction failed: {e}")

    # --------- TEST ONLY MODE ---------
    from PIL import Image

    if opt['phase'] == 'test':
        cd_model.load_state_dict(torch.load(args.weights))
        cd_model.eval()
        os.makedirs(opt['path_cd']['result'], exist_ok=True)
        from tqdm import tqdm
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
                img1 = batch['A'].to(device)
                img2 = batch['B'].to(device)
                name = batch.get('name', [str(idx)])[0]
                # Forward pass
                outputs = cd_model(img1, img2)
                # Unpack outputs (multi-head)
                if isinstance(outputs, dict):
                    seg_logits_t1 = outputs.get('seg_t1', None)
                    seg_logits_t2 = outputs.get('seg_t2', None)
                    change_pred   = outputs.get('change', None)
                elif isinstance(outputs, (list, tuple)):
                    if len(outputs) == 3:
                        seg_logits_t1, seg_logits_t2, change_pred = outputs
                    else:
                        change_pred = outputs[0]
                        seg_logits_t1 = seg_logits_t2 = None
                else:
                    change_pred = outputs
                    seg_logits_t1 = seg_logits_t2 = None
                # Get predictions
                if seg_logits_t1 is not None:
                    pred_seg_t1 = torch.argmax(seg_logits_t1, dim=1)[0].cpu().numpy().astype(np.uint8)
                else:
                    pred_seg_t1 = None
                if seg_logits_t2 is not None:
                    pred_seg_t2 = torch.argmax(seg_logits_t2, dim=1)[0].cpu().numpy().astype(np.uint8)
                else:
                    pred_seg_t2 = None
                if change_pred is not None:
                    change_probs = torch.softmax(change_pred, dim=1)[0].cpu().numpy()
                    pred_change = np.argmax(change_probs, axis=0).astype(np.uint8)
                else:
                    pred_change = None
                # Save results as single-channel PNGs
                if pred_seg_t1 is not None:
                    Image.fromarray(pred_seg_t1).save(os.path.join(opt['path_cd']['result'], f'{name}_pred_seg_t1.png'))
                if pred_seg_t2 is not None:
                    Image.fromarray(pred_seg_t2).save(os.path.join(opt['path_cd']['result'], f'{name}_pred_seg_t2.png'))
                if pred_change is not None:
                    Image.fromarray(pred_change).save(os.path.join(opt['path_cd']['result'], f'{name}_pred_change.png'))
        print(f"Test results saved to {opt['path_cd']['result']}")
        exit(0)

        for current_epoch in range(n_epochs):
            print("......Begin Training......")
            metric.clear()
            metric_seg.clear()
            cd_model.train()

            train_result_path = os.path.join(opt['path_cd']['result'], 'train', str(current_epoch))
            os.makedirs(train_result_path, exist_ok=True)

            # Log LR
            logger.info(f"lr: {optimizer.param_groups[0]['lr']:.7f}\n ")

            epoch_loss = 0.0

            # Prepare limited/iterable loader if max_train_batches set
            _max_train = getattr(args, 'max_train_batches', 0) or 0
            _train_total = min(len(train_loader), _max_train) if _max_train > 0 else len(train_loader)
            _train_iter = islice(train_loader, _max_train) if _max_train > 0 else train_loader

            # Zero grad at start of accumulation window
            optimizer.zero_grad(set_to_none=True)

            for current_step, batch in enumerate(tqdm(_train_iter, total=_train_total,
                                                    desc=f"Train {current_epoch}/{n_epochs}")):
                # ------------------ Fetch & move data ------------------
                train_im1 = batch['A'].to(device, non_blocking=True)
                train_im2 = batch['B'].to(device, non_blocking=True)

                # -------------- First-batch input debug (optional) --------------
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

                # ------------------ Forward (with AMP) ------------------
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs = cd_model(train_im1, train_im2)

                    # Unpack model outputs
                    if isinstance(outputs, dict):
                        seg_logits_t1 = outputs.get('seg_t1', None)
                        seg_logits_t2 = outputs.get('seg_t2', None)
                        change_pred    = outputs.get('change', None)
                    elif isinstance(outputs, (list, tuple)):
                        if len(outputs) == 3:
                            seg_logits_t1, seg_logits_t2, change_pred = outputs
                        else:
                            change_pred = outputs[0]
                            seg_logits_t1, seg_logits_t2 = None, None
                    else:
                        change_pred = outputs
                        seg_logits_t1 = seg_logits_t2 = None

                    # -------------- First-batch output debug (optional) --------------
                    if current_step == 0:
                        print("\nOUTPUT DEBUG INFO:")
                        print("-"*60)
                        if seg_logits_t1 is not None:
                            print(f"Seg T1 logits - shape: {seg_logits_t1.shape}, range: [{seg_logits_t1.min():.4f}, {seg_logits_t1.max():.4f}]")
                            _pred = torch.argmax(seg_logits_t1, dim=1)
                            print(f"Seg T1 predictions - unique classes: {torch.unique(_pred).tolist()}")
                        if seg_logits_t2 is not None:
                            print(f"Seg T2 logits - shape: {seg_logits_t2.shape}, range: [{seg_logits_t2.min():.4f}, {seg_logits_t2.max():.4f}]")
                            _pred = torch.argmax(seg_logits_t2, dim=1)
                            print(f"Seg T2 predictions - unique classes: {torch.unique(_pred).tolist()}")
                        if change_pred is not None:
                            print(f"Change logits - shape: {change_pred.shape}, range: [{change_pred.min():.4f}, {change_pred.max():.4f}]")
                            _p = torch.softmax(change_pred, dim=1)
                            print(f"Change probs (class 0): [{_p[:,0].min():.4f}, {_p[:,0].max():.4f}]")
                            print(f"Change probs (class 1): [{_p[:,1].min():.4f}, {_p[:,1].max():.4f}]")
                        print("="*60 + "\n")

                    # We no longer need raw inputs after forward
                    del train_im1, train_im2

                    # ------------------ Prepare labels ------------------
                    seg_t1 = batch.get('L1', None)
                    seg_t2 = batch.get('L2', None)
                    change = batch.get('L', None)

                    # First-batch GT debug
                    if current_step == 0:
                        print("\nGROUND TRUTH DEBUG INFO:")
                        print("-"*60)
                        if seg_t1 is not None:
                            print(f"GT Seg T1 - shape: {seg_t1.shape}, dtype: {seg_t1.dtype}, uniq: {torch.unique(seg_t1).tolist()}")
                        else: print("GT Seg T1: None")
                        if seg_t2 is not None:
                            print(f"GT Seg T2 - shape: {seg_t2.shape}, dtype: {seg_t2.dtype}, uniq: {torch.unique(seg_t2).tolist()}")
                        else: print("GT Seg T2: None")
                        if change is not None:
                            print(f"GT Change - shape: {change.shape}, dtype: {change.dtype}, uniq: {torch.unique(change).tolist()}")
                            if change.numel() > 0:
                                _cr = (change == 1).float().mean().item()
                                print(f"GT Change pixel ratio: {_cr:.4f}")
                        else: print("GT Change: None")
                        print("-"*60)

                    # Fallback for missing L1/L2
                    if (seg_t1 is None) or (seg_t2 is None):
                        if change is not None:
                            seg_t1 = change
                            seg_t2 = change
                        else:
                            # create dummy zeros to match change_pred spatial dims
                            b, _, h, w = change_pred.shape
                            seg_t1 = torch.zeros((b, h, w), dtype=torch.long)
                            seg_t2 = torch.zeros((b, h, w), dtype=torch.long)

                    # Ensure proper dtype/device
                    if isinstance(seg_t1, torch.Tensor): seg_t1 = seg_t1.to(device).long()
                    if isinstance(seg_t2, torch.Tensor): seg_t2 = seg_t2.to(device).long()
                    if isinstance(change, torch.Tensor): change = change.to(device).long()

                    # ------------------ Compute loss (NO thresholding) ------------------
                    if opt['model']['loss'] == 'extended_triplet':
                        # Expect (seg_t1, seg_t2, change_pred)
                        assert isinstance(outputs, (tuple, list)) and len(outputs) == 3, \
                            "Expected model to return (seg_t1, seg_t2, change)"
                        seg_logits_t1, seg_logits_t2, change_pred = outputs

                        # TripletChangeSegLoss expects a 1-channel change logit
                        change_u = change_pred if change_pred.shape[1] == 1 else change_pred[:, 1:2]
                        change_bin = normalize_change_target(seg_t1, seg_t2, change)  # [B,H,W] long {0,1}

                        preds  = (seg_logits_t1, seg_logits_t2, change_u)
                        labels = {'seg_t1': seg_t1, 'seg_t2': seg_t2, 'change': change_bin}

                        if current_step == 0:
                            logger.info(f"[TRAIN dtype-check] change_bin: shape={tuple(change_bin.shape)}, dtype={change_bin.dtype}, device={change_bin.device}")
                            try:
                                _derived = normalize_change_target(seg_t1, seg_t2, None)
                                mism = (_derived != change_bin).float().mean().item()
                                logger.info(f"[TRAIN consistency] derived_vs_change_bin_mismatch={mism:.6f}")
                            except Exception as e:
                                logger.warning(f"[TRAIN consistency] compare failed: {e}")

                        raw_loss, ext_loss = loss_fun(preds, labels)  # scalar tensor
                        loss_dict = {
                            'seg_t1': ext_loss.get('seg_t1'),
                            'seg_t2': ext_loss.get('seg_t2'),
                            'change': ext_loss.get('cd')
                        }

                    else:
                        # 2-class change head
                        if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
                            seg_logits_t1, seg_logits_t2, change_pred = outputs
                        else:
                            change_pred = outputs
                            # Create dummy seg logits for consistency in logging
                            b, _, h, w = change_pred.shape
                            seg_logits_t1 = torch.zeros((b, opt['model']['n_classes'], h, w),
                                                        device=change_pred.device, dtype=change_pred.dtype)
                            seg_logits_t2 = torch.zeros_like(seg_logits_t1)

                        change_bin = normalize_change_target(seg_t1, seg_t2, change)  # [B,H,W] long {0,1}

                        if current_step == 0:
                            logger.info(f"[TRAIN dtype-check] change_bin: shape={tuple(change_bin.shape)}, dtype={change_bin.dtype}, device={change_bin.device}")

                        raw_loss = loss_fun_change(change_pred, change_bin)
                        loss_dict = {'seg_t1': 0.0, 'seg_t2': 0.0, 'change': raw_loss.item()}

                    # Scale for grad accumulation
                    train_loss = raw_loss / accumulation_steps

                # ------------------ Backward & Step (AMP-aware) ------------------
                scaler.scale(train_loss).backward()

                do_step = ((current_step + 1) % accumulation_steps == 0) or ((current_step + 1) == _train_total)
                if do_step:
                    torch.nn.utils.clip_grad_norm_(cd_model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                # ------------------ Debug loss ------------------
                if current_step == 0:
                    print("\nLOSS DEBUG INFO:")
                    print("-"*60)
                    print(f"Total train loss (scaled): {train_loss.item():.6f}")
                    print(f"Loss components - seg_t1: {loss_dict['seg_t1']}, seg_t2: {loss_dict['seg_t2']}, change: {loss_dict['change']}")
                    print(f"Loss requires_grad: {train_loss.requires_grad}")
                    print("-"*60)

                # ------------------ Predictions for metrics/vis ------------------
                with torch.no_grad():
                    pred_seg_t1 = torch.argmax(seg_logits_t1, dim=1)
                    pred_seg_t2 = torch.argmax(seg_logits_t2, dim=1)
                    change_p1   = torch.softmax(change_pred, dim=1)[:, 1, :, :]
                    pred_change_bin = (change_p1 > args.change_threshold).long()

                # Log first-batch images (guarded)
                log_images_this_step = (current_step == 0) 

                if log_images_this_step:
                    # Prepare GT visualizations
                    seg_t1_np = seg_t1[0].detach().cpu().numpy()
                    seg_t2_np = seg_t2[0].detach().cpu().numpy()
                    gt_seg_t1_img = create_color_mask(seg_t1[0], num_classes=opt['model']['n_classes']) if seg_t1_np.ndim != 3 else seg_t1_np.astype(np.uint8)
                    gt_seg_t2_img = create_color_mask(seg_t2[0], num_classes=opt['model']['n_classes']) if seg_t2_np.ndim != 3 else seg_t2_np.astype(np.uint8)
                    gt_change_for_log = normalize_change_target(seg_t1, seg_t2, change)

                    # Convert inputs from batch for logging
                    A0 = batch['A'][0].detach().cpu()
                    B0 = batch['B'][0].detach().cpu()
                    def _norm_img(img):
                        if img.min() < 0:
                            img = (img + 1.0) / 2.0
                        img = (img * 255.0).clamp(0, 255).byte()
                        return img.permute(1,2,0).numpy() if img.ndim == 3 else img.numpy()

                    # Prob heatmaps
                    seg_t1_probs = torch.softmax(seg_logits_t1[0], dim=0)
                    seg_t2_probs = torch.softmax(seg_logits_t2[0], dim=0)
                    change_probs = torch.softmax(change_pred[0], dim=0)

                    wandb.log({
                        "train/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=opt['model']['n_classes']))],
                        "train/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=opt['model']['n_classes']))],
                        "train/pred_change": [wandb.Image(create_color_mask(pred_change_bin[0], num_classes=2))],
                        "train/pred_change_prob": [wandb.Image(change_probs[1].detach().cpu().numpy())],
                        "train/gt_seg_t1": [wandb.Image(gt_seg_t1_img)],
                        "train/gt_seg_t2": [wandb.Image(gt_seg_t2_img)],
                        "train/gt_change": [wandb.Image(create_color_mask(gt_change_for_log[0], num_classes=2))],
                        "train/input_T1": [wandb.Image(_norm_img(A0))],
                        "train/input_T2": [wandb.Image(_norm_img(B0))],
                        "global_step": current_epoch * len(train_loader) + current_step
                    })

                # ------------------ Metrics ------------------
                # Change (binary)
                gt_bin = (normalize_change_target(seg_t1, seg_t2, change) > 0).long().detach()
                pred_np = pred_change_bin.detach().cpu().numpy().astype(np.uint8)
                gt_np   = gt_bin.detach().cpu().numpy().astype(np.uint8)
                current_score = metric.update_cm(pr=pred_np, gt=gt_np)

                # Segmentation (multi-class)
                pred_seg_t1_np = pred_seg_t1.detach().cpu().numpy().astype(np.uint8)
                pred_seg_t2_np = pred_seg_t2.detach().cpu().numpy().astype(np.uint8)
                gt_seg_t1_np   = seg_t1.detach().cpu().numpy().astype(np.uint8)
                gt_seg_t2_np   = seg_t2.detach().cpu().numpy().astype(np.uint8)
                seg_score_t1 = metric_seg.update_cm(pr=pred_seg_t1_np, gt=gt_seg_t1_np)
                seg_score_t2 = metric_seg.update_cm(pr=pred_seg_t2_np, gt=gt_seg_t2_np)
                seg_score_avg = (seg_score_t1 + seg_score_t2) / 2.0

                # Log batch metrics
                log_dict = {
                    'train_loss': train_loss.item(),
                    'train_running_acc': current_score.item(),
                    'train_running_seg_mf1': seg_score_avg.item()
                }
                wandb.log(log_dict)

                # Periodic console log with GPU mem
                if current_step % opt['train']['train_print_iter'] == 0:
                    gpu_info = ""
                    if torch.cuda.is_available():
                        mem_alloc = torch.cuda.memory_allocated() / 1024**3
                        mem_resv  = torch.cuda.memory_reserved() / 1024**3
                        gpu_info = f", GPU Memory: {mem_alloc:.2f}GB/{mem_resv:.2f}GB"
                    logger.info('[Training CD]. epoch: [%d/%d]. Iter: [%d/%d], CD_loss: %.5f, change_mF1: %.5f, seg_mF1: %.5f%s\n' %
                                (current_epoch, n_epochs, current_step, _train_total,
                                train_loss.item(), current_score.item(), seg_score_avg.item(), gpu_info))

                # Accumulate epoch loss
                epoch_loss += train_loss.item()

                # Cleanup per-iter (let caching handle the rest)
                del outputs, seg_logits_t1, seg_logits_t2, change_pred
                del seg_t1, seg_t2, change
                del pred_seg_t1, pred_seg_t2, pred_change_bin, change_p1
                del gt_bin

            # ------------------ Epoch summary ------------------
            scores = metric.get_scores()          # change (binary)
            epoch_acc = scores['mf1']
            scores_seg = metric_seg.get_scores()  # segmentation (multi-class)
            epoch_seg_mf1  = scores_seg['mf1']
            epoch_seg_miou = scores_seg['miou']
            epoch_seg_acc  = scores_seg['acc']

            avg_epoch_loss = (epoch_loss / max(1, _train_total))
            epoch_losses.append(avg_epoch_loss)

            wandb.log({
                'train/epoch_mF1_change': epoch_acc,
                'train/epoch_mIoU_change': scores.get('miou', 0),
                'train/epoch_OA_change': scores.get('acc', 0),
                'train/epoch_mF1_seg': epoch_seg_mf1,
                'train/epoch_mIoU_seg': epoch_seg_miou,
                'train/epoch_OA_seg': epoch_seg_acc,
                'train/epoch_loss': avg_epoch_loss,
                'train_epoch_mf1': epoch_acc,   # backward-compat key
                'train_epoch_loss': avg_epoch_loss,
                'epoch': current_epoch
            })

            logger.info(f'Training - Epoch: {current_epoch}, Avg Loss: {avg_epoch_loss:.5f}, '
                        f'Change mF1: {epoch_acc:.5f}, Seg mF1: {epoch_seg_mf1:.5f}')
            if len(epoch_losses) > 1:
                trend = "↓" if epoch_losses[-1] < epoch_losses[-2] else "↑"
                logger.info(f'Loss trend: {trend} (Prev: {epoch_losses[-2]:.5f}, Curr: {epoch_losses[-1]:.5f})')

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
                    val_seg_t1 = val_data['L1'].to(device)
                    val_seg_t2 = val_data['L2'].to(device)
                    val_change = val_data['L'].to(device)
                    
                    # Forward pass
                    val_outputs = cd_model(val_img1, val_img2)
                    
                    if opt['model']['loss'] == 'extended_triplet':
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
                    
                    
                    
                    
                    
                    # Update confusion matrices
                    # 1) Segmentation (multi-class): update with T1 and T2 predictions vs GT
                    val_pred_seg_t1_np = val_pred_seg_t1.cpu().numpy().astype(np.uint8)
                    val_gt_seg_t1_np = val_seg_t1.detach().cpu().numpy().astype(np.uint8)
                    val_running_mf1_seg_t1 = val_metric.update_cm(pr=val_pred_seg_t1_np, gt=val_gt_seg_t1_np)

                    print(f"val_pred_seg_t1_np uniques: {np.unique(val_pred_seg_t1_np)}")
                    print(f"val_gt_seg_t1_np uniques: {np.unique(val_gt_seg_t1_np)}")

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

                    # Per-step validation logging
                    _val_logs = {
                        'val_loss': float(val_loss.item()),
                        'val/running_mF1_seg': float(val_running_mf1_seg),
                        'val/running_mF1_change': float(val_running_mf1_change),
                    }
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
                        
                        # Handle ground truth masks same as training
                        val_seg_t1_np = val_seg_t1[0].detach().cpu().numpy()
                        val_seg_t2_np = val_seg_t2[0].detach().cpu().numpy()

                        val_gt_seg_t1_img = create_color_mask(val_seg_t1[0], num_classes=opt['model']['n_classes'])
                        val_gt_seg_t2_img = create_color_mask(val_seg_t2[0], num_classes=opt['model']['n_classes'])
                        
                        # Force to pure binary if needed
                        if val_change_vis[0].dtype == torch.float32:
                            val_change_vis_binary = (val_change_vis[0] > 0.5).int()
                        else:
                            val_change_vis_binary = val_change_vis[0]
                            
                        # Create custom binary colormap for better visibility
                        # Black (0) for no change, white (1) for change
                        binary_mask_np = val_change_vis_binary.cpu().numpy()
                        h, w = binary_mask_np.shape
                        val_gt_change_color = np.zeros((h, w, 3), dtype=np.uint8)
                        val_gt_change_color[binary_mask_np == 1] = [255, 255, 255]  # White for changes
                        
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
                            # first image
                            "val/pred_seg_t1_prob": [wandb.Image(val_seg_t1_max_prob, caption="Val Pred Seg T1 Max Probability")],
                            "val/pred_seg_t1": [wandb.Image(create_color_mask(val_pred_seg_t1[0], num_classes=opt['model']['n_classes']), caption="Val Pred Seg T1 (multi-class)")],
                            "val/gt_seg_t1": [wandb.Image(val_gt_seg_t1_img, caption="Val GT Seg T1")],
                            # second image
                            "val/pred_seg_t2_prob": [wandb.Image(val_seg_t2_max_prob, caption="Val Pred Seg T2 Max Probability")],
                            "val/pred_seg_t2": [wandb.Image(create_color_mask(val_pred_seg_t2[0], num_classes=opt['model']['n_classes']), caption="Val Pred Seg T2 (multi-class)")],
                            "val/gt_seg_t2": [wandb.Image(val_gt_seg_t2_img, caption="Val GT Seg T2")],
                            # change image
                            "val/pred_change_prob": [wandb.Image(val_change_prob, caption="Val Pred Change Class-1 Probability")],
                            "val/pred_change": [wandb.Image(create_color_mask(val_pred_change[0], num_classes=2), caption="Val Pred Change (binary)")],
                            "val/gt_change": [wandb.Image(val_gt_change_color, caption="Val GT Change (binary color)")],
                            "global_step": current_epoch * len(train_loader) + len(train_loader),
                        })
            
            # Validation epoch change summary metrics
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
                best_model_path = os.path.join(opt['path_cd']['checkpoint'], f'best_net_{current_epoch}.pth')
                
                # Handle DataParallel if used
                if isinstance(cd_model, nn.DataParallel):
                    model_state = cd_model.module.state_dict()
                else:
                    model_state = cd_model.state_dict()
                
                torch.save(model_state, best_model_path)
                logger.info(f'New best model saved with mF1: {best_mF1:.5f} at {best_model_path}')
                
                # Also save using the save_network function for compatibility
                save_network(opt, current_epoch, cd_model, optimizer, is_best_model=True)
                
                # Log to wandb
                wandb.log({
                    'best_val_mF1': best_mF1,
                    'best_model_epoch': current_epoch
                })
            else:
                logger.info(f'Current mF1: {val_epoch_mf1:.5f} did not improve from best: {best_mF1:.5f}')

            
            # Save regular checkpoint every epoch (regardless of performance)
            save_network(opt, current_epoch, cd_model, optimizer, is_best_model=False)

        
            #################
            #    TESTING    #
            #################
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
                    
                    seg_t1 = test_data['L1']
                    seg_t2 = test_data['L2']
                    change = test_data['L'] 

                    outputs = cd_model(test_img1, test_img2)
                    
                    seg_logits_t1, seg_logits_t2, change_pred = outputs
                    
                    # Only use change head for metric and visuals (2-class)
                    # Convert prediction to binary change mask via thresholded probability
                    u = torch.softmax(change_pred.detach(), dim=1)[:, 1, :, :]
                    G_pred = (u > args.change_threshold).long()
                    # Normalize GT to binary [B,H,W]
                    test_change_bin = normalize_change_target(seg_t1, seg_t2, change)

                    # Prepare numpy arrays for metrics and update confusion matrix
                    pred_np = G_pred.int().cpu().numpy()
                    gt_np = test_change_bin.cpu().numpy().astype(np.uint8)
                    metric.update_cm(pr=pred_np, gt=gt_np)

                    # Update segmentation confusion matrix if GT available
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
                            "test/pred_seg_t1": [wandb.Image(create_color_mask(pred_seg_t1[0], num_classes=opt['model']['n_classes']), caption="Test Pred Seg T1 (multi-class)")],
                            "test/pred_seg_t2": [wandb.Image(create_color_mask(pred_seg_t2[0], num_classes=opt['model']['n_classes']), caption="Test Pred Seg T2 (multi-class)")],
                            "test/gt_seg_t1": [wandb.Image(create_color_mask(seg_t1[0], num_classes=opt['model']['n_classes']), caption="test GT Seg T1")],
                            "test/gt_seg_t2": [wandb.Image(create_color_mask(seg_t2[0], num_classes=opt['model']['n_classes']), caption="test GT Seg T2")],
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
