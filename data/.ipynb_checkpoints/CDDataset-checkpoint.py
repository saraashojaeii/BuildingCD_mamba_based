"""
CD Dataset
"""
import os
from PIL import Image
import numpy as np
from torch.utils import data
import data.util as Util
from torch.utils.data import Dataset
import torchvision
import torch

totensor = torchvision.transforms.ToTensor()

"""
CD Dataset 
├─image
├─image_post
├─label
└─list
"""

IMG_FOLDER_NAME = 'A'
IMG_POST_FOLDER_NAME = 'B'
LABEL_FOLDER_NAME = 'label'
LABEL1_FOLDER_NAME = 'label1'
LABEL2_FOLDER_NAME = 'label2'
LIST_FOLDER_NAME = 'list'

label_suffix = ".png"

#list内存放image_name 构建读取图片名字函数
def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

#获取各个文件夹的路径
def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)

def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL_FOLDER_NAME, img_name)

def get_label1_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL1_FOLDER_NAME, img_name)

def get_label2_path(root_dir, img_name):
    return os.path.join(root_dir, LABEL2_FOLDER_NAME, img_name)


class CDDataset(Dataset):
    def __init__(self, root_dir, resolution=256, split='train', data_len=-1, label_transform=None):

        self.root_dir = root_dir
        self.resolution = resolution
        self.data_len = data_len
        self.split = split #train / val / test
        self.label_transform = label_transform

        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')

        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.data_len])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.data_len])

        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')

        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.data_len])
        # Load label and ensure it is class indices, not RGB
        img_label = Image.open(L_path).convert('RGB')
        img_label = np.array(img_label)
        from data.util import rgb_mask_to_class
        from data.colormap import second_colormap
        if img_label.ndim == 3 and img_label.shape[2] == 3:
            img_label = rgb_mask_to_class(img_label, second_colormap)
        # Now img_label is [H, W] with integer class indices
        img_label = Image.fromarray(img_label.astype(np.uint8))

        img_A = Util.transform_augment_cd(img_A, min_max=(-1, 1))
        img_B = Util.transform_augment_cd(img_B, min_max=(-1, 1))
        # Convert label to tensor without normalization
        img_label = totensor(img_label).long()
        img_label = img_label.squeeze() # Remove channel dim and convert to long

        return {'A':img_A, 'B':img_B, 'L':img_label, 'Index':index}



class SCDDataset(Dataset):
    def __init__(self, root_dir, resolution=512, split='train', data_len=-1, label_transform=None):

        self.root_dir = root_dir
        self.resolution = resolution
        self.data_len = data_len
        self.split = split #train / val / test
        self.label_transform = label_transform

        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')

        self.img_name_list = load_img_name_list(self.list_path)

        self.dataset_len = len(self.img_name_list)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.dataset_len, self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_name = self.img_name_list[index % self.data_len]
        A_path = get_img_path(self.root_dir, img_name)
        B_path = get_img_post_path(self.root_dir, img_name)
        name = os.path.basename(A_path).split('.')[0]
    
        # --- load images ---
        img_A = Image.open(A_path).convert('RGB')
        img_B = Image.open(B_path).convert('RGB')
    
        # --- load labels (as RGB) ---
        L_path  = get_label_path(self.root_dir, img_name)   # optional, we will re-derive anyway
        L1_path = get_label1_path(self.root_dir, img_name)
        L2_path = get_label2_path(self.root_dir, img_name)
    
        try:
            rgb_L  = np.array(Image.open(L_path).convert('RGB'),  dtype=np.uint8) if os.path.exists(L_path) else None
            rgb_L1 = np.array(Image.open(L1_path).convert('RGB'), dtype=np.uint8)
            rgb_L2 = np.array(Image.open(L2_path).convert('RGB'), dtype=np.uint8)
        except Exception as e:
            print(f"Error loading label images for {img_name}: {e}")
            # fallback empty labels
            rgb_L  = None
            rgb_L1 = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
            rgb_L2 = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
    
        # --- map RGB -> class ids ---
        from data.util import rgb_mask_to_class
        from data.colormap import second_colormap
        lab1 = rgb_mask_to_class(rgb_L1, second_colormap)   # [H,W] int
        lab2 = rgb_mask_to_class(rgb_L2, second_colormap)   # [H,W] int
    
        # --- resize everything to the same size (nearest for labels) ---
        # use self.resolution or image size; here we enforce square (H=W=resolution)
        target_size = (self.resolution, self.resolution)
    
        # images: use bilinear when you later tensorize; here keep PIL
        img_A = img_A.resize(target_size, resample=Image.BILINEAR)
        img_B = img_B.resize(target_size, resample=Image.BILINEAR)
    
        # labels: nearest!
        lab1_pil = Image.fromarray(lab1.astype(np.uint8))
        lab2_pil = Image.fromarray(lab2.astype(np.uint8))
        lab1 = np.array(lab1_pil.resize(target_size, resample=Image.NEAREST), dtype=np.uint8)
        lab2 = np.array(lab2_pil.resize(target_size, resample=Image.NEAREST), dtype=np.uint8)
    
        # Optional: if an L file exists and you want to compare it:
        if rgb_L is not None:
            # convert provided L (RGB) to binary [H,W] (assumes 0/255 or palette)
            # Most robust: derive from lab1/lab2 instead of trusting file L
            pass
    
        # --- apply paired spatial augs (if you have any) ---
        # If Util.transform_augment_cd does spatial ops, you must apply identical params to labels.
        # If it only normalizes to [-1,1], it's fine to call it on images after resizing.
        img_A = Util.transform_augment_cd(img_A, min_max=(-1, 1))  # -> tensor [3,H,W] float
        img_B = Util.transform_augment_cd(img_B, min_max=(-1, 1))
    
        # --- derive change AFTER transforms/resize ---
        IGNORE = 255
        lab1_t = torch.from_numpy(lab1.astype(np.int64))          # [H,W] long
        lab2_t = torch.from_numpy(lab2.astype(np.int64))          # [H,W] long
        valid = (lab1_t != IGNORE) & (lab2_t != IGNORE)
        change_bin = ((lab1_t != lab2_t) & valid).long()          # [H,W] {0,1}
    
        # --- clamp labels to model range ---
        # Get from your config to avoid mismatches (you had 6 classes earlier)
        num_classes = int(self.label_transform.get('num_classes', 6)) if self.label_transform else 6
        max_class_id = num_classes - 1
        lab1_t = lab1_t.clamp(0, max_class_id)
        lab2_t = lab2_t.clamp(0, max_class_id)
    
        # --- class presence vectors (optional) ---
        cls1 = torch.zeros(num_classes, dtype=torch.int32)
        cls2 = torch.zeros(num_classes, dtype=torch.int32)
        for v in torch.unique(lab1_t):
            vi = int(v)
            if 0 <= vi <= max_class_id: cls1[vi] = 1
        for v in torch.unique(lab2_t):
            vi = int(v)
            if 0 <= vi <= max_class_id: cls2[vi] = 1
    
        return {
            'A': img_A,                # [3,H,W] float in [-1,1]
            'B': img_B,                # [3,H,W]
            'L': change_bin,           # [H,W] long {0,1}  <-- matches change head target
            'L1': lab1_t,              # [H,W] long in [0..K-1]
            'L2': lab2_t,              # [H,W] long in [0..K-1]
            'Index': index,
            'name': name,
            'cls1': cls1, 'cls2': cls2
        }


if __name__ == '__main__':
    # Use a platform-independent path
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_samples')
    print(f"Testing dataset with root directory: {root_dir}")
    
    try:
        # Create dataset instance
        cddata = SCDDataset(root_dir=root_dir)
        print(f"Dataset created with {len(cddata)} samples")
        
        # Test a few samples
        num_samples = min(5, len(cddata))
        for i in range(num_samples):
            try:
                sample = cddata.__getitem__(i)
                print(f"\nSample {i}:")
                print(f"Image A shape: {sample['A'].shape}")
                print(f"Image B shape: {sample['B'].shape}")
                print(f"Label shape: {sample['L'].shape}")
                print(f"Class 1 presence: {sample['cls1']}")
                print(f"Class 2 presence: {sample['cls2']}")
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
    except Exception as e:
        print(f"Error creating or testing dataset: {e}")
