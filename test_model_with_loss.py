import torch
import os
import sys
import json
from models.CDMamba_Seg_change import CDMamba_seg_cd
from models.loss import MultiClassCDLoss

def normalize_change_target(seg1, seg2, device=None):
    """Create binary change mask from semantic segmentations."""
    # Any pixel where segmentation class differs is a 'change'
    change = (seg1 != seg2).long()
    if device is not None:
        change = change.to(device)
    return change

def test_model_with_loss():
    print("Testing CDMamba_seg_cd model with MultiClassCDLoss...")
    
    # Load config values for consistency
    config_path = '/home/saraashojaeii/git/BuildingCD_mamba_based/config/second_cdmamba/cdmamba_seg_cd.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    num_classes = config['model']['n_classes']
    print(f"Using num_classes={num_classes} from config")
    
    # Create model instance
    model = CDMamba_seg_cd(
        spatial_dims=2,
        in_channels=3,
        num_classes=num_classes,
        use_change_head=True,
        init_filters=16,
        up_conv_mode="deepwise",
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Create loss function
    loss_weights = {'seg_t1': 1.0, 'seg_t2': 1.0, 'change': 1.0}
    loss_fn = MultiClassCDLoss(
        num_classes=num_classes,
        seg_loss="cedice",
        change_loss="ce",
        loss_weights=loss_weights
    )
    loss_fn = loss_fn.to(device)
    
    # Create dummy inputs and targets
    x1 = torch.randn(2, 3, 256, 256).to(device)  # B, C, H, W
    x2 = torch.randn(2, 3, 256, 256).to(device)
    
    # Create dummy segmentation targets
    seg_t1 = torch.randint(0, num_classes, (2, 256, 256), device=device)  # B, H, W
    seg_t2 = torch.randint(0, num_classes, (2, 256, 256), device=device)  # B, H, W
    
    # Forward pass
    print("\nRunning model forward pass...")
    with torch.no_grad():
        outputs = model(x1, x2)
    
    # Check the outputs
    if isinstance(outputs, tuple) and len(outputs) == 3:
        seg_logits_t1, seg_logits_t2, change_logits = outputs
        print(f"Segmentation T1 output shape: {seg_logits_t1.shape}")
        print(f"Segmentation T2 output shape: {seg_logits_t2.shape}")
        print(f"Change detection output shape: {change_logits.shape}")
    else:
        print(f"Unexpected output format: {type(outputs)}")
        return
    
    # Test the loss function
    print("\nTesting loss function...")
    try:
        # Prepare the predictions and targets
        preds = (seg_logits_t1, seg_logits_t2, change_logits)
        targets = {
            'seg_t1': seg_t1,
            'seg_t2': seg_t2
        }
        
        # Calculate loss
        loss, loss_components = loss_fn(preds, targets)
        
        print(f"Total loss: {loss.item()}")
        print(f"Loss components: {loss_components}")
        print("Loss calculation successful!")
    except Exception as e:
        print(f"Error calculating loss: {str(e)}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_model_with_loss()
