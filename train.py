import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
from tqdm import tqdm
import numpy as np

from data_loader import create_data_loaders
from models import create_model
from utils import (
    CombinedLoss, calculate_metrics, save_predictions, 
    EarlyStopping, get_device, count_parameters,
    save_model_checkpoint, create_output_directory
)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train model for one epoch
    """
    model.train()
    total_loss = 0.0
    total_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            metrics = calculate_metrics(outputs, masks)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'IoU': f'{metrics["iou"]:.4f}',
            'F1': f'{metrics["f1"]:.4f}'
        })
        
        # Save sample predictions
        if batch_idx == 0:
            save_predictions(images, outputs, masks, 'output/visualizations', epoch, batch_idx)
    
    # Calculate average metrics
    avg_loss = total_loss / len(train_loader)
    avg_metrics = {key: value / len(train_loader) for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics

def validate_epoch(model, val_loader, criterion, device, epoch):
    """
    Validate model for one epoch
    """
    model.eval()
    total_loss = 0.0
    total_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, masks)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'IoU': f'{metrics["iou"]:.4f}',
                'F1': f'{metrics["f1"]:.4f}'
            })
    
    # Calculate average metrics
    avg_loss = total_loss / len(val_loader)
    avg_metrics = {key: value / len(val_loader) for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics

def train_model(args):
    """
    Main training function
    """
    # Set device
    device = get_device()
    
    # Create output directories
    create_output_directory(args.output_dir)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_type=args.model_type,
        num_classes=1,
        backbone=args.backbone,
        pretrained=args.pretrained
    )
    
    model = model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function and optimizer
    criterion = CombinedLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    best_val_loss = float('inf')
    best_val_iou = 0.0
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('IoU/Train', train_metrics['iou'], epoch)
        writer.add_scalar('IoU/Validation', val_metrics['iou'], epoch)
        writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
        writer.add_scalar('F1/Validation', val_metrics['f1'], epoch)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch+1}/{args.epochs} - {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, IoU: {train_metrics["iou"]:.4f}, F1: {train_metrics["f1"]:.4f}')
        print(f'Val Loss: {val_loss:.4f}, IoU: {val_metrics["iou"]:.4f}, F1: {val_metrics["f1"]:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_iou = val_metrics['iou']
            save_model_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
            )
            print(f'New best model saved! Val Loss: {val_loss:.4f}, Val IoU: {val_metrics["iou"]:.4f}')
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_model_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # Training completed
    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time/3600:.2f} hours')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print(f'Best validation IoU: {best_val_iou:.4f}')
    
    # Save final model
    save_model_checkpoint(
        model, optimizer, epoch, val_loss,
        os.path.join(args.output_dir, 'checkpoints', 'final_model.pth')
    )
    
    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train SAR Oil Spill Detection Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Path to output directory')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['hybrid', 'deeplabv3_plus', 'segnet'],
                        help='Type of model to train')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help='Backbone for DeepLabV3+')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers')
    
    # Loss function arguments
    parser.add_argument('--bce_weight', type=float, default=1.0,
                        help='Weight for BCE loss')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Weight for Dice loss')
    parser.add_argument('--focal_weight', type=float, default=1.0,
                        help='Weight for Focal loss')
    
    # Training control arguments
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Print configuration
    print("Training Configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print()
    
    # Start training
    train_model(args)

if __name__ == "__main__":
    main()