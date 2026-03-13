import os
import torch
import cv2
import numpy as np


def load_checkpoint(model, checkpoint_path):
    """
    Load model weights from checkpoint file.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        
    Returns:
        model: Model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try to handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # Remove 'module.' prefix if present (from DDP)
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            # Assume the checkpoint is directly the state dict
            model.load_state_dict(checkpoint)
            
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    return model


def save_visualization_batch(batch, outputs, save_dir, prefix="", max_samples=4, type="blend_map"):
    """
    Save visualization of model outputs.
    
    Args:
        batch: Batch of input data (dict with 'image', 'blend_map', 'gt')
        outputs: Model outputs (predicted blend maps)
        save_dir: Directory to save visualizations
        prefix: Prefix for saved filenames (e.g., epoch number)
        max_samples: Maximum number of samples to save
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get tensors from batch and move everything to CPU for visualization
    images = batch['image'].detach().cpu()
    if type == "blend_map":
        gt_tensor_maps = batch['blend_map'].detach().cpu()
    elif type == "tensor_map":
        gt_tensor_maps = batch['tensor_map'].detach().cpu()
    else:
        raise ValueError(f"Invalid type: {type}")
    gt_images = batch['gt'].detach().cpu()
    filenames = batch['filename']
    
    # Move outputs to CPU
    outputs_cpu = outputs.detach().cpu()
    
    # Compute retouched images from predictions (on CPU)
    from src.blend.blend_map import apply_flow_formula
    # retouched_images = apply_flow_formula_batch(images, outputs_cpu)
    
    # Process each sample in batch (up to max_samples)
    for i in range(min(max_samples, images.size(0))):
        # Original image
        orig_img = images[i].permute(1, 2, 0).numpy() * 255
        orig_img = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Ground truth
        gt_img = gt_images[i].permute(1, 2, 0).numpy() * 255
        gt_img = cv2.cvtColor(gt_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Ground truth blend map
        gt_blend = gt_tensor_maps[i].permute(1, 2, 0).numpy()
        # min_max_normalize
        gt_blend_x = gt_blend[:, :, 0]
        gt_blend_y = gt_blend[:, :, 1]
        
        # gt_blend = cv2.cvtColor(gt_blend.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Predicted blend map
        pred_blend = outputs_cpu[i].permute(1, 2, 0).numpy() * 255
        pred_blend_x = pred_blend[:, :, 0]
        pred_blend_y = pred_blend[:, :, 1]
        # pred_blend = cv2.cvtColor(pred_blend.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Retouched image
        retouched = apply_flow_formula(images[i], outputs_cpu[i])
        retouched = retouched.permute(1, 2, 0).numpy() * 255
        retouched = cv2.cvtColor(retouched.astype(np.uint8), cv2.COLOR_RGB2BGR)

        retouched_gt = apply_flow_formula(images[i], gt_tensor_maps[i])
        retouched_gt = retouched_gt.permute(1, 2, 0).numpy() * 255
        retouched_gt = cv2.cvtColor(retouched_gt.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Get base filename without extension
        base_name = os.path.splitext(filenames[i])[0]
        
        # Save images
        prefix = f"{i}"
        cv2.imwrite(f"{save_dir}/{prefix}_original.jpg", orig_img)
        cv2.imwrite(f"{save_dir}/{prefix}_gt.jpg", gt_img)
        cv2.imwrite(f"{save_dir}/{prefix}_retouched_pred_map.jpg", retouched)
        cv2.imwrite(f"{save_dir}/{prefix}_retouched_gt_map.jpg", retouched_gt)

        cv2.imwrite(f"{save_dir}/{prefix}_tensor_map_x_gt.png", gt_blend_x)
        cv2.imwrite(f"{save_dir}/{prefix}_tensor_map_y_gt.png", gt_blend_y)
        cv2.imwrite(f"{save_dir}/{prefix}_tensor_map_x_pred.png", pred_blend_x)
        cv2.imwrite(f"{save_dir}/{prefix}_tensor_map_y_pred.png", pred_blend_y)
        
        # Create a grid visualization (optional)
        grid = np.concatenate([
            np.concatenate([orig_img, gt_img], axis=1),
            np.concatenate([retouched, retouched_gt], axis=1)
        ], axis=0)
        cv2.imwrite(f"{save_dir}/{prefix}_grid.jpg", grid)

def compute_metrics(pred_images, gt_images):
    """
    Compute evaluation metrics between predicted and ground truth images.
    
    Args:
        pred_images: Predicted images tensor [B, C, H, W]
        gt_images: Ground truth images tensor [B, C, H, W]
        
    Returns:
        dict: Dictionary of metrics (PSNR, SSIM, etc.)
    """
    # Make sure tensors are on CPU and detached from the computation graph
    pred_np = pred_images.detach().cpu().numpy()
    gt_np = gt_images.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        pred = pred_np[i].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        gt = gt_np[i].transpose(1, 2, 0)
        
        # Calculate MSE
        mse = np.mean((pred - gt) ** 2)
        
        # Calculate PSNR
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(1.0 / mse)
        
        psnr_values.append(psnr)
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    
    return {
        'psnr': avg_psnr,
        # Add more metrics as needed
    }