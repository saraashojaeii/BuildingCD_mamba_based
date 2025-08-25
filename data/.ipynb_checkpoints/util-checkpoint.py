import torchvision

totensor = torchvision.transforms.ToTensor()

import numpy as np

def rgb_mask_to_class(mask, palette):
    """
    Convert an RGB mask (H, W, 3) or (B, H, W, 3) to class indices (H, W) or (B, H, W).
    mask: np.ndarray or torch.Tensor
    palette: list of [R, G, B]
    """
    import torch
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    palette = np.asarray(palette)
    orig_shape = mask.shape
    if mask.ndim == 4:
        B, H, W, C = mask.shape
        mask_flat = mask.reshape(-1, 3)
        idx = np.zeros((mask_flat.shape[0],), dtype=np.int64)
        for i, color in enumerate(palette):
            matches = np.all(mask_flat == color, axis=1)
            idx[matches] = i
        idx = idx.reshape(B, H, W)
    elif mask.ndim == 3 and mask.shape[2] == 3:
        H, W, C = mask.shape
        mask_flat = mask.reshape(-1, 3)
        idx = np.zeros((mask_flat.shape[0],), dtype=np.int64)
        for i, color in enumerate(palette):
            matches = np.all(mask_flat == color, axis=1)
            idx[matches] = i
        idx = idx.reshape(H, W)
    else:
        raise ValueError(f"Unsupported mask shape: {orig_shape}")
    return idx

def transform_augment_cd(img, min_max=(0, 1)):
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img