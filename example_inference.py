#!/usr/bin/env python3
"""
Example usage of the Double Chin Removal Inference script
"""

import os
from inference import DoubleChinInference


def main():
    """Example of how to use the inference class"""
    
    # Configuration
    model_path = "weights-blend-maps/blend_maps_v2_5k/double_chin_bmap_best.pth"
    input_image = "path/to/your/input/image.jpg"  # Replace with your input image path
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
        # Initialize inference
        print("Initializing inference model...")
        inference = DoubleChinInference(
            model_path=model_path,
            device='auto',  # Will use CUDA if available, otherwise CPU
            img_size=1024
        )
        
        # Run inference
        print("Running inference...")
        result = inference.run_inference(
            image_path=input_image,
            output_path=output_image,
            save_flow_map=True  # Also save the flow map visualization
        )
        
        print(f"Inference completed successfully!")
        print(f"Input: {input_image}")
        print(f"Output: {output_image}")
        print(f"Flow map: {output_image.replace('.jpg', '_flow.jpg')}")
        
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    main() 