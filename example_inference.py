#!/usr/bin/env python3
"""
Example usage of the Double Chin Removal Inference script.

Model architecture params (model_variant, last_layer_activation, blend_scale)
are auto-detected from checkpoint metadata — no manual config needed.
"""

import os
import cv2
from inference import DoubleChinRemover, run_inference, save_img


def main():
    """Example of how to use the inference class"""

    # Configuration
    model_path = "checkpoint_best_psnr.pth"          # Path to trained checkpoint
    input_image = "path/to/your/input/image.jpg"     # Replace with your input image path
    output_image = "output/result.jpg"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please make sure the model weights are available.")
        return

    # Check if input image exists
    if not os.path.exists(input_image):
        print(f"Input image not found at: {input_image}")
        print("Please provide a valid input image path.")
        return

    try:
        # Initialize inference (auto-detects model config from checkpoint)
        print("Initializing inference model...")
        remover = DoubleChinRemover(
            model_path=model_path,
            device='auto',  # Will use CUDA if available, then MPS, then CPU
            img_size=1024,
        )

        # Load and convert image (BGR -> RGB)
        image_bgr = cv2.imread(input_image)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Run inference
        print("Running inference...")
        retouched_img, blend_map = run_inference(remover, image_rgb)

        # Save results
        save_img(retouched_img, output_image)
        print(f"Inference completed successfully!")
        print(f"Input:  {input_image}")
        print(f"Output: {output_image}")

    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    main() 