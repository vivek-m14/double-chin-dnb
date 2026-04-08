#!/usr/bin/env python3
"""
Inference script for Double Chin Removal Model
Uses blend maps to generate flow-based warping for double chin removal
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import yaml

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.unet import BaseUNetHalf
# from utils.utils import load_checkpoint
# from blend.blend_map import apply_flow_formula


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume the checkpoint is directly the state dict
        state_dict = checkpoint

    # Strip 'module.' prefix from DDP-saved checkpoints
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith('module.') else k] = v

    model.load_state_dict(new_state_dict)
    return model

class DoubleChinRemover:
    """Inference class for Double Chin Removal Model"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 img_size: int = 1024):
        """
        Initialize the inference model
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
            img_size: Input image size for the model
        """
        self.model_path = model_path
        self.img_size = img_size
        
        # Set device
        if device == 'auto':
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = torch.device(device)
        
        
        # Initialize model
        self.model = self._load_model()
        
    def _load_model(self) -> BaseUNetHalf:
        """Load the trained model"""
        # Initialize model (2 input channels for blend maps, 2 output channels for flow)
        model = BaseUNetHalf(
            n_channels=3,  # RGB input
            n_classes=3,
            deep_supervision=False,
            init_weights=False,
            last_layer_activation="sigmoid"  # Flow maps use tanh for [-1, 1] range
        )
        
        # Load trained weights
        model = load_checkpoint(model, self.model_path)
        model.to(self.device)
        model.eval()

        return model

    def to(self, device: str):
        self.device = device
        self.model.to(device)

        return self

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image: Input image as numpy array [H, W, C] in RGB format
            
        Returns:
            Preprocessed image tensor [1, C, H, W] ready for model input
        """
        # Convert to float32 and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Resize image to model input size
        if image.shape[:2] != (self.img_size, self.img_size):
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def postprocess(self, 
                   original_image: torch.Tensor,
                   blend_map: torch.Tensor, 
                ) -> np.ndarray:
        """
        Postprocess model output to generate final result
        
        Args:
            original_image: Original input image tensor [H, W, C]
            blend_map: Model output blend map tensor [H, W, 3]
            
        Returns:
            Processed image as numpy array [H, W, C] in RGB format
        """
        # Apply flow to original image
        img_h, img_w = original_image.shape[:2]
        # blend_map = cv2.resize(blend_map, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4)
        blend_map = cv2.resize(blend_map, (img_w, img_h))

        # Apply blend formula to get retouched image
        retouched_np = self.apply_blend_formula(original_image/255, blend_map)
            
        # Convert back to numpy and denormalize
        retouched_img = np.clip(retouched_np * 255, 0, 255).astype(np.uint8)
            
        return retouched_img

    def apply_blend_formula(self, image, blend_map):
        """
        Apply the simplified Linear Light blend formula
        """
        if isinstance(image, torch.Tensor):
            return torch.clamp(image + 2 * blend_map - 1, 0.0, 1.0)

        return np.clip(image + 2 * blend_map - 1, 0.0, 1.0)
    
def run_inference(double_chin_remover, original_image) -> np.ndarray:
    """
    Run complete inference pipeline
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        save_flow_map: Whether to save flow map visualization
        
    Returns:
        Processed image as numpy array
    """
    
    input_tensor = double_chin_remover.preprocess(original_image)
    
    # Step 3: Model inference
    with torch.no_grad():
        blend_map = double_chin_remover.model(input_tensor)
        blend_map = blend_map.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
    # Step 4: Postprocess
    retouched_img = double_chin_remover.postprocess(original_image, blend_map)
    
    return retouched_img, blend_map

def save_img(image, output_path):
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image[..., ::-1])

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Double Chin Removal Inference')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', required=True, help='Output image path')
    parser.add_argument('--model', '-m', 
                       default='weights-blend-maps/blend_maps_v2_5k/double_chin_bmap_best.pth',
                       help='Model weights path')
    parser.add_argument('--device', '-d', default='auto', 
                       choices=['cpu', 'cuda', 'auto'], help='Device to use')
    parser.add_argument('--img_size', type=int, default=1024, help='Input image size')
    parser.add_argument('--save_flow', action='store_true', help='Save flow map visualization')
    
    args = parser.parse_args()

    # Load input image (BGR -> RGB)
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        print(f"Error: Could not read image at {args.input}")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Initialize model
    remover = DoubleChinRemover(
        model_path=args.model,
        device=args.device,
        img_size=args.img_size
    )

    # Run inference
    try:
        retouched_img, blend_map = run_inference(remover, image_rgb)
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        save_img(retouched_img, args.output)
        print(f"Saved retouched image to {args.output}")

        if args.save_flow:
            blend_path = str(Path(args.output).with_suffix('')) + '_blend_map.png'
            save_img((blend_map * 255).astype(np.uint8), blend_path)
            print(f"Saved blend map to {blend_path}")

        print("Inference completed successfully!")

    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()