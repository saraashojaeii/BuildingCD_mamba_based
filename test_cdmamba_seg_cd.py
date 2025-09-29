import torch
import os
import sys
from models.CDMamba_Seg_change import CDMamba_seg_cd

def test_cdmamba_seg_cd():
    print("Testing CDMamba_seg_cd model...")
    
    # Create model instance
    model = CDMamba_seg_cd(
        spatial_dims=2,
        in_channels=3,
        num_classes=6,
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
    
    # Create dummy inputs
    print("\nTesting single image forward pass...")
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        output = model(x)
    print(f"Single image output shape: {output.shape}")
    
    # Test with image pair (change detection mode)
    print("\nTesting dual image (change detection) forward pass...")
    x1 = torch.randn(2, 3, 256, 256).to(device)
    x2 = torch.randn(2, 3, 256, 256).to(device)
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
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_cdmamba_seg_cd()
