import torch
import os
import random
import numpy as np

def estimate_class_counts(dataloader, num_classes: int, ignore_index: int = 255, max_batches=None):
    counts = torch.zeros(num_classes, dtype=torch.long)
    seen = 0
    for batch in dataloader:
        # adapt to your batch structure
        # assume batch = {"image": X, "label": Y} or (X, Y)
        if isinstance(batch, dict):
            y = batch["L1"]
        else:
            y = batch[1]
        y = y.long()  # [B,H,W]
        mask = (y != ignore_index) & (y >= 0) & (y < num_classes)
        for cls in range(num_classes):
            counts[cls] += (mask & (y == cls)).sum().item()
        seen += 1
        if max_batches is not None and seen >= max_batches:
            break
    return counts

def set_seed(seed: int):
    """Ensure reproducibility across runs."""
    if seed is None:
        return  # do nothing if no seed is provided
    
    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch (CPU & CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # all GPUs

    # Make CUDA deterministic (may slow things down a bit)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Control hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Random seed set to {seed}")


def normalize_change_target(seg1: torch.Tensor | None,
                            seg2: torch.Tensor | None,
                            change_gt: torch.Tensor | None) -> torch.Tensor:
    """Return binary change target of shape [B, H, W] (dtype long, {0,1}).

    Handles cases where:
    - change_gt is provided in various formats (NCHW with C=1, NHWC RGB, or [B,H,W] int/float)
    - or must be derived from seg1 and seg2 which may be RGB (NHWC), one-hot/logits (NCHW, C>1),
      or single-channel (NCHW, C=1 or [B,H,W]).
    """
    def _to_index_mask(x: torch.Tensor) -> torch.Tensor:
        # Convert arbitrary segmentation label tensor to [B,H,W] integer indices
        if x.dim() == 4:
            # NCHW
            if x.size(1) == 1:
                return x.squeeze(1).long()
            # NHWC (e.g., RGB mask)
            if x.size(-1) == 3 and x.shape[1] != 3:
                # Assume channels-last
                return x.any(dim=-1).long()  # fallback to binary presence per-pixel
            # Multi-channel: take argmax as class indices
            return torch.argmax(x, dim=1).long()
        elif x.dim() == 3:
            # Already [B,H,W]
            return x.long()
        else:
            raise ValueError(f"Unsupported seg shape: {tuple(x.shape)}")

    if change_gt is not None:
        c = change_gt
        # If NHWC RGB -> any over last channel
        if c.dim() == 4 and c.size(-1) == 3 and (c.shape[1] != 3):
            c = c.any(dim=-1).long()
        elif c.dim() == 4 and c.size(1) == 1:
            c = c.squeeze(1).long()
        elif c.dim() == 3:
            # [B,H,W] possibly float/binary
            c = (c > 0).long()
        else:
            # As a conservative fallback
            c = _to_index_mask(c)
            c = (c > 0).long()
        return c

    # Derive from seg1 and seg2
    if seg1 is None or seg2 is None:
        raise ValueError("seg1/seg2 required when change_gt is None")

    # Handle RGB NHWC
    if seg1.dim() == 4 and seg1.size(-1) == 3 and (seg1.shape[1] != 3) and \
       seg2.dim() == 4 and seg2.size(-1) == 3 and (seg2.shape[1] != 3):
        change = (seg1 != seg2).any(dim=-1).long()
        return change

    # Convert to class indices if needed
    s1 = _to_index_mask(seg1)
    s2 = _to_index_mask(seg2)
    change = (s1 != s2).long()
    return change

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
    cmap = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]]

    rgb = _np.zeros((h, w, 3), dtype=_np.uint8)
    
    # Custom color mapping to ensure class 0 is visible (not black)
    colors = []
    for i in range(num_classes):
        color = _np.array(cmap[i])
        colors.append(color.astype(_np.uint8))
    
    # Apply color mapping
    for cls in range(num_classes):
        if cls in unique_vals:
            rgb[arr == cls] = colors[cls]
    
    return rgb