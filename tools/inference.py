import os
import sys
import time
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.unet import UNet
from src.blend.blend_map import apply_blend_formula


def load_model(model_path, device='cuda'):
    """Load the trained model with 3-channel output."""
    model = UNet(n_channels=3, n_classes=3) 
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint) 
    model.to(device)
    model.eval()
    return model


def process_image_at_original_size(image_path, model, model_input_size=(1024, 1024), device='cuda'):
    # 1. Load the original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_h, original_w = original_image.shape[:2]
    
    # 2. Create a normalized version for processing
    original_norm = original_image.astype(np.float32) / 255.0
    
    # 3. Create a resized copy for model input
    resized_image = cv2.resize(original_image, model_input_size)
    resized_norm = resized_image.astype(np.float32) / 255.0
    tensor_image = torch.from_numpy(resized_norm).permute(2, 0, 1).unsqueeze(0)
    
    # 4. Get the 3-channel blend map from the model
    st = time.time()
    with torch.no_grad():
        tensor_image = tensor_image.to(device)
        blend_map = model(tensor_image)
    print(f"Model inference time: {time.time() - st:.3f} seconds")
    
    # 5. Resize the 3-channel blend map back to original size
    blend_map_np = blend_map.squeeze().cpu().numpy()
    blend_map_np = np.transpose(blend_map_np, (1, 2, 0))
    blend_map_original = cv2.resize(blend_map_np, (original_w, original_h))
    blend_map_original_tensor = torch.from_numpy(blend_map_original).permute(2, 0, 1).unsqueeze(0)
    original_tensor = torch.from_numpy(original_norm).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    
    # 6. Apply the blend formula to the original-sized image
    st = time.time()
    retouched_tensor = apply_blend_formula(original_tensor, blend_map_original_tensor)
    print(f"Blend application time: {time.time() - st:.3f} seconds")
    
    # 7. Convert tensor to numpy and remove batch dimension
    retouched_np = retouched_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    return original_image, blend_map_original, retouched_np


def visualize_results(original_image, blend_map, retouched_image, save_path=None):
    """Visualize the original, blend map, and retouched images."""
    if torch.is_tensor(retouched_image):
        if retouched_image.dim() == 4:  # [B,C,H,W]
            retouched_image = retouched_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        elif retouched_image.dim() == 3:  # [C,H,W]
            retouched_image = retouched_image.permute(1, 2, 0).cpu().numpy()
    
    # Ensure it's a proper 3D array with shape [H, W, C]
    if isinstance(retouched_image, np.ndarray):
        if retouched_image.ndim == 4:
            retouched_image = retouched_image.squeeze(0)
    
    retouched_image = np.clip(retouched_image, 0, 1)
    
    plt.figure(figsize=(30, 15))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Blend map
    plt.subplot(1, 3, 2)
    plt.imshow(blend_map)
    plt.title('Blend Map (3-channel)')
    plt.axis('off')
    
    # Retouched image
    plt.subplot(1, 3, 3)
    plt.imshow(retouched_image)
    plt.title('Retouched Image')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Results visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Configuration 
    model_path = 'weights/skin_retouching_best.pth' 
    samples_dir = '../samples/'
    output_dir = '../results/'  
    model_input_size = (1024, 1024)  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    
    # Get list of images
    imgs = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    imgs = [os.path.join(samples_dir, img) for img in imgs]
    
    # Process each image
    for image_path in tqdm(imgs, desc="Processing images"):
        try:
            print(f"Processing image: {image_path}")
            
            # Get base filename
            base_name = os.path.basename(image_path).split('.')[0]
            
            # Process image
            original_image, blend_map, retouched_image = process_image_at_original_size(
                image_path, model, model_input_size, device
            )
            
            # Save original image
            original_save_path = os.path.join(output_dir, f"{base_name}_original.jpg")
            original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(original_save_path, original_image_bgr)
            print(f"Original image saved to {original_save_path}")
            
            # Save retouched image
            retouched_image_np = (retouched_image * 255).astype(np.uint8)
            retouched_image_bgr = cv2.cvtColor(retouched_image_np, cv2.COLOR_RGB2BGR)
            retouched_save_path = os.path.join(output_dir, f"{base_name}_retouched.jpg")
            cv2.imwrite(retouched_save_path, retouched_image_bgr)
            print(f"Retouched image saved to {retouched_save_path}")
            
            # Save blend map image (now a 3-channel image)
            blend_map_save_path = os.path.join(output_dir, f"{base_name}_blend_map.jpg")
            blend_map_bgr = cv2.cvtColor((blend_map * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(blend_map_save_path, blend_map_bgr)
            print(f"Blend map saved to {blend_map_save_path}")
            
            # Optionally visualize and save comparison plot
            # Uncomment the lines below to save visualization
            # plot_save_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            # visualize_results(original_image, blend_map, retouched_image, plot_save_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            
    print("Processing complete!")