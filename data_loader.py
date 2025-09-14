import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob

class OilSpillDataset(Dataset):
    """
    Dataset class for SAR-based oil spill detection
    Supports both Sentinel and PALSAR satellite data
    """
    
    def __init__(self, data_dir, satellite_type='both', transform=None, mode='train'):
        self.data_dir = data_dir
        self.satellite_type = satellite_type
        self.transform = transform
        self.mode = mode
        
        # Collect all image pairs
        self.image_pairs = self._collect_image_pairs()
        
    def _collect_image_pairs(self):
        """Collect all satellite image and mask pairs"""
        image_pairs = []
        
        if self.satellite_type in ['both', 'palsar']:
            palsar_dir = os.path.join(self.data_dir, 'palsar')
            if os.path.exists(palsar_dir):
                palsar_pairs = self._get_pairs_from_dir(palsar_dir)
                image_pairs.extend(palsar_pairs)
        
        if self.satellite_type in ['both', 'sentinel']:
            sentinel_dir = os.path.join(self.data_dir, 'sentinel')
            if os.path.exists(sentinel_dir):
                sentinel_pairs = self._get_pairs_from_dir(sentinel_dir)
                image_pairs.extend(sentinel_pairs)
        
        return image_pairs
    
    def _get_pairs_from_dir(self, satellite_dir):
        """Get image pairs from a specific satellite directory"""
        pairs = []
        
        # Check if this is a test directory with separate gt and sat folders
        if os.path.exists(os.path.join(satellite_dir, 'sat')) and os.path.exists(os.path.join(satellite_dir, 'gt')):
            # Test directory structure
            sat_dir = os.path.join(satellite_dir, 'sat')
            gt_dir = os.path.join(satellite_dir, 'gt')
            
            # Get all satellite images
            sat_files = glob.glob(os.path.join(sat_dir, '*_sat.jpg'))
            
            for sat_file in sat_files:
                # Get corresponding mask file
                base_name = os.path.basename(sat_file).replace('_sat.jpg', '')
                mask_file = os.path.join(gt_dir, f'{base_name}_mask.png')
                
                if os.path.exists(mask_file):
                    pairs.append((sat_file, mask_file))
        else:
            # Training directory structure (sat and mask in same folder)
            sat_files = glob.glob(os.path.join(satellite_dir, '*_sat.jpg'))
            
            for sat_file in sat_files:
                # Get corresponding mask file
                base_name = os.path.basename(sat_file).replace('_sat.jpg', '')
                mask_file = os.path.join(satellite_dir, f'{base_name}_mask.png')
                
                if os.path.exists(mask_file):
                    pairs.append((sat_file, mask_file))
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        sat_path, mask_path = self.image_pairs[idx]
        
        # Load satellite image
        sat_image = cv2.imread(sat_path)
        sat_image = cv2.cvtColor(sat_image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=sat_image, mask=mask)
            sat_image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to binary (0 or 1)
        if isinstance(mask, torch.Tensor):
            mask = (mask > 128).float()
        else:
            mask = (mask > 128).astype(np.float32)
        
        # Add channel dimension to match model output
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        
        return sat_image, mask

def get_transforms(mode='train', image_size=512):
    """
    Get data augmentation transforms
    """
    if mode == 'train':
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    return transform

def create_data_loaders(data_dir, batch_size=4, image_size=512, num_workers=2):
    """
    Create train and validation data loaders
    """
    # Get all image pairs for splitting
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'test')
    
    # Create datasets
    train_dataset = OilSpillDataset(
        train_dir, 
        satellite_type='both',
        transform=get_transforms('train', image_size),
        mode='train'
    )
    
    val_dataset = OilSpillDataset(
        val_dir,
        satellite_type='both', 
        transform=get_transforms('val', image_size),
        mode='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test data loading
    data_dir = "dataset"
    train_loader, val_loader = create_data_loaders(data_dir, batch_size=2)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Test loading a batch
    for images, masks in train_loader:
        print(f"Image shape: {images.shape}")
        print(f"Mask shape: {masks.shape}")
        break
