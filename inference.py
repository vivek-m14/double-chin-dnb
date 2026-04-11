#!/usr/bin/env python3
"""
Inference script for Double Chin Removal Model
Uses blend maps with Linear Light formula for double chin removal.
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

from models.unet import BaseUNetHalf, BaseUNetHalfLite, BaseUNetHalfLiteROI
from utils.utils_blend import load_checkpoint


def _peek_checkpoint_config(checkpoint_path: str) -> dict:
    """
    Extract model_config dict from a checkpoint file.
    The full checkpoint is deserialized into CPU memory.
    Returns an empty dict for legacy checkpoints that don't contain model_config.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return ckpt.get('model_config', {})
    except Exception:
        return {}


class DoubleChinRemover:
    """Inference class for Double Chin Removal Model"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 img_size: int = 1024,
                 model_variant: str = None,
                 last_layer_activation: str = None,
                 blend_scale: float = None):
        """
        Initialize the inference model.
        
        Model architecture params (model_variant, last_layer_activation, blend_scale)
        are auto-detected from checkpoint metadata when set to None.
        Explicit values override checkpoint metadata.
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run inference on ('cpu', 'cuda', 'mps', or 'auto')
            img_size: Input image size for the model
            model_variant: 'default' (BaseUNetHalf) or 'lite' (BaseUNetHalfLite).
                           Auto-detected from checkpoint if None.
            last_layer_activation: 'sigmoid', 'tanh', or 'residual_tanh'.
                                   Auto-detected from checkpoint if None.
            blend_scale: Scale for residual_tanh activation.
                         Auto-detected from checkpoint if None.
        """
        self.model_path = model_path
        self.img_size = img_size
        
        # Auto-detect model config from checkpoint, then apply explicit overrides
        ckpt_config = _peek_checkpoint_config(model_path)
        self.model_variant = model_variant or ckpt_config.get('model_variant', 'default')
        self.last_layer_activation = last_layer_activation or ckpt_config.get('last_layer_activation', 'sigmoid')
        self.blend_scale = blend_scale if blend_scale is not None else ckpt_config.get('blend_scale', 0.5)
        # ROI crop — auto-detected from checkpoint (only used by Lite variant)
        self.roi_crop_enabled = ckpt_config.get('roi_crop_enabled', False)
        self.roi_crop_fraction = ckpt_config.get('roi_crop_fraction', 0.5)
        
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize model
        self.model = self._load_model()
        
        roi_str = f" | roi_crop: bottom {self.roi_crop_fraction:.0%}" if self.roi_crop_enabled else ""
        print(f"Model: {self.model_variant} | activation: {self.last_layer_activation} "
              f"| blend_scale: {self.blend_scale} | device: {self.device}{roi_str}")
        
    def _load_model(self):
        """Load the trained model"""
        ModelClass = BaseUNetHalfLite if self.model_variant == 'lite' else BaseUNetHalf
        model_kwargs = dict(
            n_channels=3,  # RGB input
            n_classes=3,
            deep_supervision=False,
            init_weights=False,
            last_layer_activation=self.last_layer_activation,
            blend_scale=self.blend_scale,
        )
        if ModelClass is BaseUNetHalfLite and self.roi_crop_enabled:
            ModelClass = BaseUNetHalfLiteROI
            model_kwargs['roi_crop_fraction'] = self.roi_crop_fraction
        elif self.roi_crop_enabled:
            import warnings
            warnings.warn("roi_crop_enabled ignored: only supported with model_variant='lite'")
        model = ModelClass(**model_kwargs)
        
        # Load trained weights (handles DDP prefix stripping)
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
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, image[..., ::-1])

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Double Chin Removal Inference')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', required=True, help='Output image path')
    parser.add_argument('--model', '-m', required=True, help='Model checkpoint path')
    parser.add_argument('--config', '-c', default=None,
                       help='YAML config file (model params auto-detected from checkpoint if omitted)')
    parser.add_argument('--device', '-d', default='auto', 
                       choices=['cpu', 'cuda', 'mps', 'auto'], help='Device to use')
    parser.add_argument('--img_size', type=int, default=1024, help='Input image size')
    parser.add_argument('--model_variant', default=None,
                       choices=['default', 'lite'],
                       help='Model variant (auto-detected from checkpoint if omitted)')
    parser.add_argument('--last_layer_activation', default=None,
                       choices=['sigmoid', 'tanh', 'residual_tanh'],
                       help='Output activation (auto-detected from checkpoint if omitted)')
    parser.add_argument('--blend_scale', type=float, default=None,
                       help='Blend scale for residual_tanh (auto-detected from checkpoint if omitted)')
    parser.add_argument('--save_blend', action='store_true', help='Save blend map visualization')
    
    args = parser.parse_args()

    # Load overrides from YAML config if provided
    cfg_variant = None
    cfg_activation = None
    cfg_blend_scale = None
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        cfg_variant = cfg.get('model_variant')
        cfg_activation = cfg.get('last_layer_activation')
        cfg_blend_scale = cfg.get('blend_scale')

    # Priority: CLI args > YAML config > checkpoint metadata > defaults
    model_variant = args.model_variant or cfg_variant
    last_layer_activation = args.last_layer_activation or cfg_activation
    blend_scale = args.blend_scale if args.blend_scale is not None else cfg_blend_scale

    # Load input image (BGR -> RGB)
    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        print(f"Error: Could not read image at {args.input}")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Initialize model (None values trigger auto-detection from checkpoint)
    remover = DoubleChinRemover(
        model_path=args.model,
        device=args.device,
        img_size=args.img_size,
        model_variant=model_variant,
        last_layer_activation=last_layer_activation,
        blend_scale=blend_scale,
    )

    # Run inference
    try:
        retouched_img, blend_map = run_inference(remover, image_rgb)
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        save_img(retouched_img, args.output)
        print(f"Saved retouched image to {args.output}")

        if args.save_blend:
            blend_path = str(Path(args.output).with_suffix('')) + '_blend_map.png'
            save_img((blend_map * 255).astype(np.uint8), blend_path)
            print(f"Saved blend map to {blend_path}")

        print("Inference completed successfully!")

    except Exception as e:
        print(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()