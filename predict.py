import os
import torch
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

from models import create_model
from utils import (
    preprocess_image, postprocess_prediction, save_mask,
    get_device, create_output_directory
)

def predict_single_image(model, image_path, output_path, device, threshold=0.5):
    """
    Predict oil spill mask for a single image
    """
    # Preprocess image
    image_tensor, original_size = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Postprocess prediction
    mask = postprocess_prediction(prediction, original_size, threshold)
    
    # Save mask
    save_mask(mask, output_path)
    
    return mask

def predict_batch_images(model, input_dir, output_dir, device, threshold=0.5):
    """
    Predict oil spill masks for a batch of images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        # Get output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_mask.png")
        
        try:
            # Predict mask
            mask = predict_single_image(model, image_path, output_path, device, threshold)
            print(f"Processed: {os.path.basename(image_path)} -> {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

def create_visualization(image_path, mask_path, output_path):
    """
    Create visualization showing original image and predicted mask
    """
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original SAR Image')
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Oil Spill Detection Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = image.copy()
    overlay[mask > 128] = [255, 0, 0]  # Red overlay for oil spills
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red = Oil Spill)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def predict_with_visualization(model, input_path, output_dir, device, threshold=0.5):
    """
    Predict and create visualizations
    """
    # Create output directories
    mask_dir = os.path.join(output_dir, 'masks')
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    if os.path.isfile(input_path):
        # Single image
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
        viz_path = os.path.join(viz_dir, f"{base_name}_visualization.png")
        
        # Predict mask
        mask = predict_single_image(model, input_path, mask_path, device, threshold)
        
        # Create visualization
        create_visualization(input_path, mask_path, viz_path)
        
        print(f"Prediction completed:")
        print(f"  Mask saved: {mask_path}")
        print(f"  Visualization saved: {viz_path}")
        
    elif os.path.isdir(input_path):
        # Directory of images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
        
        if not image_files:
            print(f"No image files found in {input_path}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        for image_path in tqdm(image_files, desc="Processing images"):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(mask_dir, f"{base_name}_mask.png")
            viz_path = os.path.join(viz_dir, f"{base_name}_visualization.png")
            
            try:
                # Predict mask
                mask = predict_single_image(model, image_path, mask_path, device, threshold)
                
                # Create visualization
                create_visualization(image_path, mask_path, viz_path)
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        print(f"All predictions completed!")
        print(f"  Masks saved in: {mask_dir}")
        print(f"  Visualizations saved in: {viz_dir}")
    
    else:
        print(f"Invalid input path: {input_path}")

def load_trained_model(checkpoint_path, model_type='hybrid', backbone='resnet50', device='cpu'):
    """
    Load trained model from checkpoint
    """
    # Create model
    model = create_model(
        model_type=model_type,
        num_classes=1,
        backbone=backbone,
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Predict Oil Spill Detection')
    
    # Input/Output arguments
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='results',
                        help='Path to output directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='hybrid',
                        choices=['hybrid', 'deeplabv3_plus', 'segnet'],
                        help='Type of model')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help='Backbone for DeepLabV3+')
    
    # Prediction arguments
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Create visualizations')
    
    args = parser.parse_args()
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    print("Loading trained model...")
    model = load_trained_model(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        backbone=args.backbone,
        device=device
    )
    print("Model loaded successfully!")
    
    # Create output directory
    create_output_directory(args.output)
    
    # Make predictions
    print("Starting prediction...")
    if args.visualize:
        predict_with_visualization(
            model=model,
            input_path=args.input,
            output_dir=args.output,
            device=device,
            threshold=args.threshold
        )
    else:
        if os.path.isfile(args.input):
            # Single image
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(args.output, f"{base_name}_mask.png")
            predict_single_image(model, args.input, output_path, device, args.threshold)
            print(f"Mask saved: {output_path}")
        else:
            # Directory of images
            predict_batch_images(
                model=model,
                input_dir=args.input,
                output_dir=args.output,
                device=device,
                threshold=args.threshold
            )
    
    print("Prediction completed!")

if __name__ == "__main__":
    main()