import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    
    Computes MSE loss between VGG feature maps of generated and target images.
    """
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        # Use weights parameter with proper weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])
        self.slice4 = nn.Sequential(*list(vgg.children())[16:23])
        
        # Freeze parameters
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def to(self, device):
        """Override to method to ensure all submodules are moved to device"""
        self.slice1 = self.slice1.to(device)
        self.slice2 = self.slice2.to(device)
        self.slice3 = self.slice3.to(device)
        self.slice4 = self.slice4.to(device)
        return super(PerceptualLoss, self).to(device)

    def forward(self, x, y):
        # Make sure model is on the same device as inputs
        if next(self.parameters()).device != x.device:
            self.to(x.device)

        # ImageNet normalization — VGG16 expects this distribution
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        y = (y - mean) / std

        # Extract feature maps
        h_x = x
        h_y = y
        h1_x = self.slice1(h_x)
        h1_y = self.slice1(h_y)
        h2_x = self.slice2(h1_x)
        h2_y = self.slice2(h1_y)
        h3_x = self.slice3(h2_x)
        h3_y = self.slice3(h2_y)
        h4_x = self.slice4(h3_x)
        h4_y = self.slice4(h3_y)
        
        # Compute MSE loss on feature maps
        return F.mse_loss(h1_x, h1_y) + F.mse_loss(h2_x, h2_y) + F.mse_loss(h3_x, h3_y) + F.mse_loss(h4_x, h4_y)


class TotalVariationLoss(nn.Module):
    """
    Total Variation Loss for spatial smoothness.
    
    Encourages spatial smoothness in the generated image by minimizing
    the difference between adjacent pixel values.
    """
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, x):
        batch_size, c, h, w = x.size()
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        
        # Compute horizontal and vertical differences
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w-1]), 2).sum()
        
        # Normalize by size
        return (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        """
        Compute the total number of elements in a tensor, excluding batch dim.
        """
        return t.size()[1] * t.size()[2] * t.size()[3]


def masked_l1_loss(pred, target, mask):
    """
    L1 loss weighted by a binary mask, normalized by mask area.

    Focuses training on the edit region (where mask == 1).

    Args:
        pred:  (B, C, H, W) predicted blend maps
        target: (B, C, H, W) target blend maps
        mask:  (B, 1, H, W) binary mask (1 = edit region)

    Returns:
        Scalar loss averaged over masked pixels.
    """
    mask_c = mask.expand_as(pred)
    loss_map = (pred - target).abs() * mask_c
    return loss_map.sum() / mask_c.sum().clamp(min=1e-6)


def unmasked_l1_loss(pred, target, mask):
    """
    L1 loss on the non-edit region (identity preservation).

    Args:
        pred:  (B, C, H, W) predicted blend maps
        target: (B, C, H, W) target blend maps
        mask:  (B, 1, H, W) binary mask (1 = edit region)

    Returns:
        Scalar loss averaged over unmasked pixels.
    """
    inv_mask = (1.0 - mask).expand_as(pred)
    loss_map = (pred - target).abs() * inv_mask
    return loss_map.sum() / inv_mask.sum().clamp(min=1e-6)


def generate_blend_mask(target_blend_maps, threshold=0.02):
    """
    Derive a binary edit mask from the target blend map.

    Neutral blend value is 0.5.  Any channel deviating beyond *threshold*
    from 0.5 is considered part of the edit region.

    Args:
        target_blend_maps: (B, C, H, W) target blend maps in [0, 1]
        threshold: minimum per-channel deviation from 0.5 to count as edit

    Returns:
        mask: (B, 1, H, W) float tensor — 1 in edit region, 0 elsewhere.
    """
    # Any channel deviating → edit pixel
    deviation = (target_blend_maps - 0.5).abs()         # (B, C, H, W)
    max_dev, _ = deviation.max(dim=1, keepdim=True)     # (B, 1, H, W)
    mask = (max_dev > threshold).float()

    # Light 3×3 dilation to cover boundary pixels
    mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    return mask


class CombinedLoss:
    """
    Combined loss function using multiple components with weights.

    Supports **masked** and **unmasked** L1 terms that split the blend-map
    loss into an edit-region term and an identity-preservation term, following
    the pattern from the reference ABPN trainer.

    Args:
        lambda_blend (float): Weight for global blend map L1 loss
        lambda_blend_masked (float): Weight for masked (edit-region) blend L1
        lambda_blend_unmasked (float): Weight for unmasked (identity) blend L1
        lambda_image_mse (float): Weight for retouched image MSE loss
        lambda_perc (float): Weight for perceptual loss
        lambda_tv (float): Weight for total variation loss
        blend_mask_threshold (float): Deviation from 0.5 to define edit region
    """
    def __init__(
        self,
        lambda_blend=1.0,
        lambda_blend_masked=1.0,
        lambda_blend_unmasked=0.0,
        lambda_image_mse=1.0,
        lambda_perc=0.1,
        lambda_tv=0.1,
        blend_mask_threshold=0.02,
    ):
        self.criterion_l1 = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_perceptual = PerceptualLoss()
        self.criterion_tv = TotalVariationLoss()

        self.lambda_blend = lambda_blend
        self.lambda_blend_masked = lambda_blend_masked
        self.lambda_blend_unmasked = lambda_blend_unmasked
        self.lambda_image_mse = lambda_image_mse
        self.lambda_perc = lambda_perc
        self.lambda_tv = lambda_tv
        self.blend_mask_threshold = blend_mask_threshold

        self.device = None

    def to(self, device):
        """
        Move loss components to specified device
        """
        self.device = device
        self.criterion_l1 = self.criterion_l1.to(device)
        self.criterion_mse = self.criterion_mse.to(device)
        self.criterion_perceptual = self.criterion_perceptual.to(device)
        self.criterion_tv = self.criterion_tv.to(device)
        return self

    def __call__(self, pred_blend_maps, target_blend_maps, retouched_images, gt_images):
        """
        Compute combined loss.

        Args:
            pred_blend_maps: Predicted blend maps
            target_blend_maps: Target blend maps
            retouched_images: Retouched images using pred_blend_maps
            gt_images: Ground truth images

        Returns:
            total_loss: Combined loss value
            losses_dict: Dictionary with individual loss components
        """
        # If device is not set, use the device of the inputs
        if self.device is None:
            self.to(pred_blend_maps.device)

        # ── Blend map mask (derived from ground-truth) ───────────────
        mask = generate_blend_mask(target_blend_maps, self.blend_mask_threshold)

        # ── Individual losses ────────────────────────────────────────
        # Global blend map L1
        blend_map_loss = self.criterion_l1(pred_blend_maps, target_blend_maps)
        # Masked (edit-region) blend L1
        blend_masked_loss = masked_l1_loss(pred_blend_maps, target_blend_maps, mask)
        # Unmasked (identity) blend L1
        blend_unmasked_loss = unmasked_l1_loss(pred_blend_maps, target_blend_maps, mask)

        image_mse_loss = self.criterion_mse(retouched_images, gt_images)
        perc_loss = self.criterion_perceptual(retouched_images, gt_images)
        tv_loss = self.criterion_tv(pred_blend_maps)

        # ── Combined loss ────────────────────────────────────────────
        total_loss = (
            self.lambda_blend * blend_map_loss
            + self.lambda_blend_masked * blend_masked_loss
            + self.lambda_blend_unmasked * blend_unmasked_loss
            + self.lambda_image_mse * image_mse_loss
            + self.lambda_perc * perc_loss
            + self.lambda_tv * tv_loss
        )

        # Return all losses for logging
        losses_dict = {
            'blend_map_loss': blend_map_loss.item(),
            'blend_masked_loss': blend_masked_loss.item(),
            'blend_unmasked_loss': blend_unmasked_loss.item(),
            'image_mse_loss': image_mse_loss.item(),
            'perc_loss': perc_loss.item(),
            'tv_loss': tv_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, losses_dict