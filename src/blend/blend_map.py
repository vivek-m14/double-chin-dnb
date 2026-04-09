import torch
import numpy as np
import cv2
import torch.nn.functional as F

# def compute_target_blend_map(original, retouched, noise_threshold=7.0/255.0, epsilon=1e-6):
#     """
#     Compute the target blend map from original and retouched images.
    
#     Args:
#         original: Original image tensor [B, C, H, W] or numpy array [H, W, C]
#         retouched: Retouched image tensor [B, C, H, W] or numpy array [H, W, C]
#         noise_threshold: Threshold to ignore small differences
#         epsilon: Small value to avoid division by zero
        
#     Returns:
#         blend_map: Computed blend map tensor [B, C, H, W]
#     """
    # # Convert numpy arrays to tensors if needed
    # if isinstance(original, np.ndarray):
    #     original = torch.from_numpy(original).permute(2, 0, 1)
    #     retouched = torch.from_numpy(retouched).permute(2, 0, 1)
    
    # # Make sure both tensors are on the same device
    # device = original.device
    # retouched = retouched.to(device)
    
    # # Calculate difference between retouched and squared original
    # diff = torch.abs(retouched - original * original)
    
    # # Apply noise threshold to ignore small differences
    # diff = torch.where(diff < noise_threshold, torch.zeros_like(diff), diff)
    
    # # Calculate blend map using formula
    # numerator = diff * torch.sign(retouched - original * original)
    # denominator = 2 * (original - original * original)
    
    # # Avoid division by zero
    # denominator = torch.where(
    #     torch.abs(denominator) < epsilon,
    #     torch.ones_like(denominator) * epsilon * torch.sign(denominator + epsilon),
    #     denominator
    # )
    
    # # Calculate blend map
    # blend_map = torch.where(
    #     diff == 0,
    #     torch.ones_like(numerator) * 0.5,  # Default value when no retouching
    #     numerator / denominator
    # )
    
    # # Clamp values to [0, 1] range
    # return torch.clamp(blend_map, 0.0, 1.0)
def compute_target_blend_map(original, retouched, noise_threshold=3.0/255.0):
    # Linear Light formula solving (simplified):
    
    if isinstance(original, np.ndarray):
        original = torch.from_numpy(original).permute(2, 0, 1)
        retouched = torch.from_numpy(retouched).permute(2, 0, 1)
        
    blend_map = (retouched - original + 1) / 2
    
    diff = torch.abs(retouched - original)
    mask_noise = diff < noise_threshold
    
    # Apply noise filter - set to 0.5 (neutral) where noise is detected
    blend_map = torch.where(mask_noise, torch.ones_like(blend_map) * 0.5, blend_map)
    
    # Clamp to ensure values are in [0, 1] range
    blend_map = torch.clamp(blend_map, 0.0, 1.0)
    
    return blend_map

def compute_target_blend_map_np(original, retouched, noise_threshold=3.0/255.0):
    # Linear Light formula solving (simplified):
    
    # if isinstance(original, np.ndarray):
    #     original = torch.from_numpy(original).permute(2, 0, 1)
    #     retouched = torch.from_numpy(retouched).permute(2, 0, 1)
    original = original.astype(np.float32) / 255.0
    retouched = retouched.astype(np.float32) / 255.0
        
    blend_map = (retouched - original + 1) / 2
    
    diff = np.abs(retouched - original)
    mask_noise = diff < noise_threshold
    
    # Apply noise filter - set to 0.5 (neutral) where noise is detected
    blend_map = np.where(mask_noise, np.ones_like(blend_map) * 0.5, blend_map)
    
    # Clamp to ensure values are in [0, 1] range
    blend_map = np.clip(blend_map, 0.0, 1.0)
    
    return blend_map

def process_target_blend_map(flow):
    """ 

    changing the range of flow map from [-img_size, img_size] to [-1, 1] so that it can be resized to any size
    and then applied to the image using grid_sample
    
    Args:
        flow: 
            shape: [H, W, 2]
            dtype: float32
            range: [-img_size, img_size]

    Returns:
        flow_map:
            shape: [H, W, 2]
            dtype: float32
            range: [-1, 1]
    """
    h, w = flow.shape[:2]
    # y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    flow_map_x = flow[..., 0] / (w - 1)
    flow_map_y = flow[..., 1] / (h - 1)
    
    flow_map = np.stack((flow_map_x, flow_map_y), axis=-1)
    
    return flow_map


# def apply_blend_formula(image, blend_map):
#     """
#     Apply the blend formula to generate retouched image with 3-channel blend map:
#     retouched = (1 - 2 * blend_map) * image * image + 2 * blend_map * image
    
#     Args:
#         image: Original image tensor [B, C, H, W]
#         blend_map: Blend map tensor [B, C, H, W]
    
#     Returns:
#         retouched: Retouched image tensor [B, C, H, W]
#     """
#     # Ensure tensors are on the same device
#     device = image.device
#     blend_map = blend_map.to(device)
    
#     retouched = (1 - 2 * blend_map) * image * image + 2 * blend_map * image
#     return torch.clamp(retouched, 0.0, 1.0)

def apply_blend_formula(image, blend_map):
    """
    Apply the simplified Linear Light blend formula
    """
    device = image.device
    blend_map = blend_map.to(device)
    return torch.clamp(image + 2 * blend_map - 1, 0.0, 1.0)

def apply_flow_formula(img: torch.Tensor, flow_map: torch.Tensor, percentage: float=1.0):
    """
    Args:
        img:
            shape: [H, W, C]
            dtype: float32
            range: [0, 1]
        flow_map:
            shape: [H, W, 2]
            dtype: float32
            range: [-1, 1]
        percentage:
            float
            range: [0, 1]

    Returns:
    """
    # Convert inputs to torch tensors
    
    _, h, w = flow_map.shape
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), 
                         torch.arange(w, dtype=torch.float32), 
                         indexing='ij')
    
    flow_map_x = x + flow_map[0] * (w - 1) * percentage
    flow_map_y = y + flow_map[1] * (h - 1) * percentage
    
    # Stack and normalize to [-1, 1] range expected by grid_sample
    flow_map = torch.stack([
        2.0 * flow_map_x / (w - 1) - 1.0,
        2.0 * flow_map_y / (h - 1) - 1.0
    ], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    
    # Apply flow using grid_sample
    img_wrapped = F.grid_sample(img.unsqueeze(0), flow_map, 
                              mode='bilinear',
                              padding_mode='border',
                              align_corners=True).squeeze(0)
    
    return img_wrapped


def apply_flow_formula_batch(images: torch.Tensor, flow_maps: torch.Tensor, percentage: float=1.0):
    """
    Args:
        img:
            shape: [B, C, H, W]
            dtype: float32
            range: [0, 1]
        flow_map:
            shape: [B, 2, H, W]
            dtype: float32
            range: [-1, 1]
        percentage:
            float
            range: [0, 1]

    Returns:
    """
    # Convert inputs to torch tensors
    # Process entire batch at once using grid_sample
    b, _, h, w = images.shape
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=images.device),
                         torch.arange(w, dtype=torch.float32, device=images.device),
                         indexing='ij')
    
    flow_map_x = x + flow_maps[:,0] * (w - 1) * percentage 
    flow_map_y = y + flow_maps[:,1] * (h - 1) * percentage

    flow_map = torch.stack([
        2.0 * flow_map_x / (w - 1) - 1.0,
        2.0 * flow_map_y / (h - 1) - 1.0
    ], dim=-1)  # [B, H, W, 2]

    return F.grid_sample(images, flow_map,
                        mode='bilinear',
                        padding_mode='border', 
                        align_corners=True)


# def reconstruct_image(original, blend_map):
#     """
#     Reconstruct retouched image from original image and blend map.
#     Alias for apply_blend_formula for consistency with older code.
    
#     Args:
#         original: Original image tensor [B, C, H, W]
#         blend_map: Blend map tensor [B, C, H, W]
    
#     Returns:
#         reconstructed: Reconstructed image tensor [B, C, H, W]
#     """
#     # Use apply_blend_formula for consistency and to ensure device handling
#     return apply_blend_formula(original, blend_map)

def reconstruct_image(original, blend_map):
    """
    Reconstruct the retouched image using the original image and blend map
    using the simplified Linear Light formula:
    retouched = original + 2*blend_map - 1
    
    Args:
        original: Original image tensor [C, H, W]
        blend_map: Blend map tensor [C, H, W]
    
    Returns:
        reconstructed: Reconstructed retouched image
    """
    reconstructed = original + 2.0 * blend_map - 1.0
    return torch.clamp(reconstructed, 0.0, 1.0)
