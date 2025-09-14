#!/usr/bin/env python3
"""
Quick training script for SAR Oil Spill Detection
This script provides a simplified interface for training the model
"""

import subprocess
import sys
import os

def main():
    print("🌊 SAR Oil Spill Detection - Quick Training")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("❌ Dataset directory not found!")
        print("Please ensure your dataset is in the 'dataset' directory")
        return False
    
    # Check if dataset has required structure
    train_dir = os.path.join("dataset", "train")
    if not os.path.exists(train_dir):
        print("❌ Training data not found!")
        print("Please ensure your dataset has a 'train' subdirectory")
        return False
    
    print("✅ Dataset found")
    
    # Training parameters
    epochs = 50
    batch_size = 4
    learning_rate = 1e-4
    
    print(f"\n🚀 Starting training with:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Model: Hybrid DeepLabV3+ & SegNet")
    
    # Build training command
    cmd = [
        "python", "train.py",
        "--data_dir", "dataset",
        "--output_dir", "output",
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--model_type", "hybrid",
        "--backbone", "resnet50",
        "--pretrained"
    ]
    
    print(f"\n📝 Training command: {' '.join(cmd)}")
    print("\n🎯 Starting training...")
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        print("\n📊 Results:")
        print("  - Model checkpoints saved in: output/checkpoints/")
        print("  - Training logs saved in: output/logs/")
        print("  - Visualizations saved in: output/visualizations/")
        
        print("\n🎯 Next steps:")
        print("1. Use the trained model for predictions:")
        print("   python predict.py --input your_image.jpg --checkpoint output/checkpoints/best_model.pth")
        print("2. Check results in the 'results' directory")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
