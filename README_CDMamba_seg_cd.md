# CDMamba_seg_cd Model

This model implements a multi-class change detection architecture with segmentation heads for T1 and T2 images, along with a change prediction head.

## Features

- Supports both single-image segmentation and dual-image change detection
- Two loss function options:
  - `extended_triplet`: Combines segmentation losses with consistency and coupling losses
  - `multi_class_cd`: Combines losses from all three heads (seg_t1, seg_t2, change)
- Configurable number of segmentation classes

## Training

### Using Extended Triplet Loss

```bash
python train_seg_cd.py --config config/second_cdmamba/cdmamba_seg_cd.json
```

### Using Multi-Class CD Loss

```bash
python train_seg_cd.py --config config/second_cdmamba/cdmamba_seg_cd_multiclass.json
```

## Testing Models

You can test the model with different loss functions using the provided test scripts:

```bash
# Test with triplet loss
python test_triplet_loss.py

# Test with multi-class loss
python test_model_with_loss.py
```

## Configuration

Both config files include the following important parameters:

1. Model parameters:
   - `name`: "cdmamba_seg_cd" - The model identifier
   - `loss`: Either "extended_triplet" or "multi_class_cd" - The loss function to use
   - `n_classes`: 6 - Number of segmentation classes
   - `use_change_head`: true - Whether to enable the change detection head

2. Loss weights (for multi_class_cd):
   ```json
   "loss_weights": {
     "seg_t1": 1.0,
     "seg_t2": 1.0,
     "change": 1.0
   }
   ```

3. Extended triplet parameters:
   ```json
   "extended_triplet": {
     "lambda_seg": 1.0,
     "lambda_cd": 1.0,
     "lambda_unch": 0.2,
     "lambda_ch": 0.2,
     "lambda_cpl": 0.5,
     "T": 4.0,
     "margin": 0.3
   }
   ```
