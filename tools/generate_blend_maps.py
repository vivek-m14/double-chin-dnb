"""
Script to generate blend maps from original and retouched image pairs.
"""
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import argparse
import sys

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.blend.blend_map import compute_target_blend_map, reconstruct_image


def generate_gt_blend_maps(original_dir, retouched_dir, output_dir, resize_dim=(1024, 1024)):
    """
    Generate ground truth blend maps from original and retouched image pairs.
    
    Args:
        original_dir (str): Directory containing original images
        retouched_dir (str): Directory containing retouched images
        output_dir (str): Directory to save generated blend maps
        resize_dim (tuple): Dimensions to resize images to (width, height)
        
    Returns:
        list: List of PSNR values for reconstructed images
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    psnr_arr = []
    
    # Get all image files
    file_names = [f for f in os.listdir(original_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Count files that need processing
    files_to_process = []
    for file_name in file_names:
        output_path = os.path.join(output_dir, file_name)
        retouched_path = os.path.join(retouched_dir, file_name)
        
        if not os.path.exists(output_path) and os.path.exists(retouched_path):
            files_to_process.append(file_name)
    
    print(f"Found {len(files_to_process)} files to process out of {len(file_names)} total files")
    
    if len(files_to_process) == 0:
        print("All blend maps already generated. Nothing to do.")
        return []
    
    # Process each file
    for file_name in tqdm(files_to_process, desc="Generating blend maps"):
        original_path = os.path.join(original_dir, file_name)
        retouched_path = os.path.join(retouched_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        # Load and preprocess images
        original = cv2.imread(original_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, resize_dim)
        original = original.astype(np.float32) / 255.0
        
        retouched = cv2.imread(retouched_path)
        retouched = cv2.cvtColor(retouched, cv2.COLOR_BGR2RGB)
        retouched = cv2.resize(retouched, resize_dim)
        retouched = retouched.astype(np.float32) / 255.0
        
        # Compute blend map
        blend_map = compute_target_blend_map(original, retouched)
        
        # Reconstruct retouched image for validation
        original_tensor = torch.from_numpy(original).permute(2, 0, 1)
        reconstructed = reconstruct_image(original_tensor, blend_map)
        
        # Convert tensors back to numpy
        blend_map_np = blend_map.permute(1, 2, 0).numpy()
        reconstructed_np = reconstructed.permute(1, 2, 0).numpy()
        
        # Calculate PSNR between retouched and reconstructed
        mse = np.mean((retouched - reconstructed_np) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        psnr_arr.append(psnr)
        
        # Save blend map
        blend_map_img = (blend_map_np * 255).astype(np.uint8)
        cv2.imwrite(output_path, cv2.cvtColor(blend_map_img, cv2.COLOR_RGB2BGR))
        
        # Optionally save reconstructed image for verification (commented out)
        # reconstructed_path = os.path.join(output_dir, f"reconstructed_{file_name}")
        # reconstructed_img = (reconstructed_np * 255).astype(np.uint8)
        # cv2.imwrite(reconstructed_path, cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR))
    
    # Print PSNR statistics
    if psnr_arr:
        print(f"Average PSNR: {np.mean(psnr_arr):.2f} dB")
        print(f"Min PSNR: {np.min(psnr_arr):.2f} dB")
        print(f"Max PSNR: {np.max(psnr_arr):.2f} dB")
    
    return psnr_arr


def main():
    parser = argparse.ArgumentParser(description='Generate blend maps from original and retouched image pairs')
    parser.add_argument('--original_dir', type=str, default='dataset/src/',
                        help='Directory containing original images')
    parser.add_argument('--retouched_dir', type=str, default='dataset/gt/',
                        help='Directory containing retouched images')
    parser.add_argument('--output_dir', type=str, default='dataset/blend_map/',
                        help='Directory to save generated blend maps')
    parser.add_argument('--resize', type=int, default=1024,
                        help='Size to resize images to (both width and height)')
    
    args = parser.parse_args()
    
    print(f"Original images directory: {args.original_dir}")
    print(f"Retouched images directory: {args.retouched_dir}")
    print(f"Output blend maps directory: {args.output_dir}")
    print(f"Resize dimensions: ({args.resize}, {args.resize})")
    
    # Generate blend maps
    generate_gt_blend_maps(
        args.original_dir,
        args.retouched_dir,
        args.output_dir,
        resize_dim=(args.resize, args.resize)
    )
    
    print("Blend map generation completed!")


if __name__ == "__main__":
    main()
