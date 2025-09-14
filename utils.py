import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """
    Combined loss function using BCE, Dice, and Focal loss
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    
    def forward(self, inputs, targets):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice + self.focal_weight * focal

def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate segmentation metrics
    """
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target_binary.view(-1).cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(target_flat, pred_flat)
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    
    # Calculate IoU
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = (intersection / (union + 1e-8)).item()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }

def save_predictions(images, predictions, targets, save_dir, epoch, batch_idx):
    """
    Save prediction visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(4, images.shape[0])):  # Save first 4 images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        gt = targets[i].cpu().numpy()
        if len(gt.shape) == 3:
            gt = gt.squeeze(0)  # Remove channel dimension for visualization
        axes[1].imshow(gt, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        pred = torch.sigmoid(predictions[i]).detach().cpu().numpy()
        if len(pred.shape) == 3:
            pred = pred.squeeze(0)  # Remove channel dimension for visualization
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_batch_{batch_idx}_sample_{i}.png'))
        plt.close()

def preprocess_image(image_path, image_size=512):
    """
    Preprocess image for inference
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original size
    original_size = image.shape[:2]
    
    # Apply transforms
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)
    
    return image_tensor, original_size

def postprocess_prediction(prediction, original_size, threshold=0.5):
    """
    Postprocess prediction mask
    """
    # Apply sigmoid and threshold
    mask = torch.sigmoid(prediction)
    mask = (mask > threshold).float()
    
    # Convert to numpy
    mask = mask.squeeze().cpu().numpy()
    
    # Resize to original size
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert to uint8
    mask = (mask * 255).astype(np.uint8)
    
    return mask

def save_mask(mask, output_path):
    """
    Save mask as PNG image
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save mask
    cv2.imwrite(output_path, mask)

def create_output_directory(base_dir="output"):
    """
    Create output directory structure
    """
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)

class EarlyStopping:
    """
    Early stopping utility
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def get_device():
    """
    Get available device (CUDA or CPU)
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device

def count_parameters(model):
    """
    Count trainable parameters in model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)

def load_model_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, loss

if __name__ == "__main__":
    # Test utility functions
    device = get_device()
    print(f"Device: {device}")
    
    # Test loss functions
    loss_fn = CombinedLoss()
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    loss = loss_fn(pred, target)
    print(f"Combined loss: {loss.item():.4f}")
    
    # Test metrics
    metrics = calculate_metrics(pred, target)
    print(f"Metrics: {metrics}")
