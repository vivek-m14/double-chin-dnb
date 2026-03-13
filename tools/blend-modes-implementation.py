import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import math

# Small value to prevent division by zero
EPSILON = 1e-6
# Minimum difference threshold to exclude noise
NOISE_THRESHOLD = 3.0/255.0

#################################
# Blend Mode Implementations
#################################

class BlendMode:
    """Base class for blend modes"""
    def __init__(self, name):
        self.name = name

    def blend(self, original, blend_map):
        """Apply blending using the blend map"""
        raise NotImplementedError("Subclasses must implement this method")
    
    def compute_blend_map(self, original, retouched):
        """Compute blend map from original and retouched images"""
        raise NotImplementedError("Subclasses must implement this method")


class CustomMode(BlendMode):
    """The original custom blending mode from the provided code"""
    def __init__(self):
        super().__init__("Custom (Original)")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: 
        retouched = (1 - 2 * blend_map) * original * original + 2 * blend_map * original
        """
        result = (1 - 2 * blend_map) * original * original + 2 * blend_map * original
        return torch.clamp(result, 0.0, 1.0)
    
    def compute_blend_map(self, original, retouched):
        """
        Compute blend map from: 
        retouched = (1 - 2 * blend_map) * original * original + 2 * blend_map * original
        
        Solve for blend_map:
        retouched = original * original + blend_map * 2 * (original - original * original)
        => (retouched - original * original) / (2 * (original - original * original)) = blend_map
        """
        # Calculate difference between retouched and original squared
        diff = torch.abs(retouched - original * original)
        
        # Apply noise threshold - set differences below threshold to zero
        diff = torch.where(diff < NOISE_THRESHOLD, torch.zeros_like(diff), diff)
        
        numerator = diff * torch.sign(retouched - original * original)
        denominator = 2 * (original - original * original)
        
        # Handle division by zero
        denominator = torch.where(
            torch.abs(denominator) < EPSILON,
            torch.ones_like(denominator) * EPSILON * torch.sign(denominator + EPSILON),
            denominator
        )
        
        # Where diff is zero (below threshold), make blend_map 0.5 (no effect)
        blend_map = torch.where(
            diff == 0,
            torch.ones_like(numerator) * 0.5,
            numerator / denominator
        )
        
        # Clamp the blend map to [0, 1] range
        blend_map = torch.clamp(blend_map, 0.0, 1.0)
        
        return blend_map


class DarkenMode(BlendMode):
    """Darken blending mode: min(Target, Blend)"""
    def __init__(self):
        super().__init__("Darken")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: retouched = min(original, blend_map)
        """
        result = torch.minimum(original, blend_map)
        return result
    
    def compute_blend_map(self, original, retouched):
        """
        For Darken mode: retouched = min(original, blend_map)
        
        To compute blend_map, we need:
        - When original <= retouched: blend_map can be any value >= original
        - When original > retouched: blend_map = retouched
        
        For simplicity, we set blend_map = retouched when original > retouched,
        and blend_map = 1.0 when original <= retouched
        """
        mask = original > retouched
        blend_map = torch.where(mask, retouched, torch.ones_like(retouched))
        return blend_map


class MultiplyMode(BlendMode):
    """Multiply blending mode: Target * Blend"""
    def __init__(self):
        super().__init__("Multiply")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: retouched = original * blend_map
        """
        result = original * blend_map
        return result
    
    def compute_blend_map(self, original, retouched):
        """
        For Multiply mode: retouched = original * blend_map
        => blend_map = retouched / original
        
        Need to handle division by zero.
        """
        # Avoid division by zero
        denominator = torch.where(
            original < EPSILON,
            torch.ones_like(original) * EPSILON,
            original
        )
        
        blend_map = retouched / denominator
        
        # Where original is nearly zero, set blend_map to 1.0 if retouched is also nearly zero,
        # or 0.0 if retouched is not nearly zero
        mask_zero = original < EPSILON
        blend_map = torch.where(
            mask_zero & (retouched < EPSILON),
            torch.ones_like(blend_map),
            blend_map
        )
        blend_map = torch.where(
            mask_zero & (retouched >= EPSILON),
            torch.zeros_like(blend_map),
            blend_map
        )
        
        return torch.clamp(blend_map, 0.0, 1.0)


class ColorBurnMode(BlendMode):
    """Color Burn blending mode: 1 - (1-Target) / Blend"""
    def __init__(self):
        super().__init__("Color Burn")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: retouched = 1 - (1-original) / blend_map
        """
        # Avoid division by zero
        denominator = torch.where(
            blend_map < EPSILON,
            torch.ones_like(blend_map) * EPSILON,
            blend_map
        )
        
        result = 1 - (1 - original) / denominator
        return torch.clamp(result, 0.0, 1.0)
    
    def compute_blend_map(self, original, retouched):
        """
        For Color Burn mode: retouched = 1 - (1-original) / blend_map
        => (1-original) / (1-retouched) = blend_map
        
        Need to handle division by zero and negative values.
        """
        # Calculate intermediate values
        numerator = 1 - original
        denominator = 1 - retouched
        
        # Avoid division by zero
        denominator = torch.where(
            torch.abs(denominator) < EPSILON,
            torch.ones_like(denominator) * EPSILON,
            denominator
        )
        
        blend_map = numerator / denominator
        
        # Special cases
        # When retouched is 1.0, blend_map should be 1.0
        blend_map = torch.where(
            retouched > 1.0 - EPSILON,
            torch.ones_like(blend_map),
            blend_map
        )
        
        # When original is 0.0 and retouched is 0.0, blend_map can be any value, default to 1.0
        blend_map = torch.where(
            (original < EPSILON) & (retouched < EPSILON),
            torch.ones_like(blend_map),
            blend_map
        )
        
        return torch.clamp(blend_map, 0.0, 1.0)


class LinearBurnMode(BlendMode):
    """Linear Burn blending mode: Target + Blend - 1"""
    def __init__(self):
        super().__init__("Linear Burn")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: retouched = original + blend_map - 1
        """
        result = original + blend_map - 1.0
        return torch.clamp(result, 0.0, 1.0)
    
    def compute_blend_map(self, original, retouched):
        """
        For Linear Burn mode: retouched = original + blend_map - 1
        => blend_map = retouched + 1 - original
        """
        blend_map = retouched + 1.0 - original
        return torch.clamp(blend_map, 0.0, 1.0)


class LightenMode(BlendMode):
    """Lighten blending mode: max(Target, Blend)"""
    def __init__(self):
        super().__init__("Lighten")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: retouched = max(original, blend_map)
        """
        result = torch.maximum(original, blend_map)
        return result
    
    def compute_blend_map(self, original, retouched):
        """
        For Lighten mode: retouched = max(original, blend_map)
        
        To compute blend_map, we need:
        - When original >= retouched: blend_map can be any value <= original
        - When original < retouched: blend_map = retouched
        
        For simplicity, we set blend_map = retouched when original < retouched,
        and blend_map = 0.0 when original >= retouched
        """
        mask = original < retouched
        blend_map = torch.where(mask, retouched, torch.zeros_like(retouched))
        return blend_map


class ScreenMode(BlendMode):
    """Screen blending mode: 1 - (1-Target) * (1-Blend)"""
    def __init__(self):
        super().__init__("Screen")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: retouched = 1 - (1-original) * (1-blend_map)
        """
        result = 1.0 - (1.0 - original) * (1.0 - blend_map)
        return result
    
    def compute_blend_map(self, original, retouched):
        """
        For Screen mode: retouched = 1 - (1-original) * (1-blend_map)
        => (1-blend_map) = (1-retouched) / (1-original)
        => blend_map = 1 - (1-retouched) / (1-original)
        
        Need to handle division by zero.
        """
        # Calculate intermediate values
        numerator = 1.0 - retouched
        denominator = 1.0 - original
        
        # Avoid division by zero
        denominator = torch.where(
            torch.abs(denominator) < EPSILON,
            torch.ones_like(denominator) * EPSILON,
            denominator
        )
        
        # Compute blend map
        blend_map = 1.0 - numerator / denominator
        
        # Special cases
        # When original is 1.0, blend_map can be any value, default to 0.0
        blend_map = torch.where(
            original > 1.0 - EPSILON,
            torch.zeros_like(blend_map),
            blend_map
        )
        
        # When retouched is 1.0, blend_map should be 1.0
        blend_map = torch.where(
            retouched > 1.0 - EPSILON,
            torch.ones_like(blend_map),
            blend_map
        )
        
        return torch.clamp(blend_map, 0.0, 1.0)


class ColorDodgeMode(BlendMode):
    """Color Dodge blending mode: Target / (1-Blend)"""
    def __init__(self):
        super().__init__("Color Dodge")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: retouched = original / (1-blend_map)
        """
        # Avoid division by zero
        denominator = 1.0 - blend_map
        denominator = torch.where(
            denominator < EPSILON,
            torch.ones_like(denominator) * EPSILON,
            denominator
        )
        
        result = original / denominator
        return torch.clamp(result, 0.0, 1.0)
    
    def compute_blend_map(self, original, retouched):
        """
        For Color Dodge mode: retouched = original / (1-blend_map)
        => (1-blend_map) = original / retouched
        => blend_map = 1 - original / retouched
        
        Need to handle division by zero.
        """
        # Special case: if retouched is close to 1.0, blend_map should be close to 1.0
        blend_map = torch.ones_like(original)
        
        # Calculate blend_map for non-extreme cases
        mask = retouched < 1.0 - EPSILON
        
        # Avoid division by zero
        denominator = torch.where(
            retouched < EPSILON,
            torch.ones_like(retouched) * EPSILON,
            retouched
        )
        
        blend_map_calc = 1.0 - original / denominator
        blend_map = torch.where(mask, blend_map_calc, blend_map)
        
        # Special case: if original is 0, blend_map can be any value, default to 0.0
        blend_map = torch.where(
            original < EPSILON,
            torch.zeros_like(blend_map),
            blend_map
        )
        
        return torch.clamp(blend_map, 0.0, 1.0)


class LinearDodgeMode(BlendMode):
    """Linear Dodge (Add) blending mode: Target + Blend"""
    def __init__(self):
        super().__init__("Linear Dodge (Add)")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: retouched = original + blend_map
        """
        result = original + blend_map
        return torch.clamp(result, 0.0, 1.0)
    
    def compute_blend_map(self, original, retouched):
        """
        For Linear Dodge mode: retouched = original + blend_map
        => blend_map = retouched - original
        """
        blend_map = retouched - original
        return torch.clamp(blend_map, 0.0, 1.0)


class OverlayMode(BlendMode):
    """Overlay blending mode"""
    def __init__(self):
        super().__init__("Overlay")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: 
        if original > 0.5: retouched = 1 - (1-2*(original-0.5)) * (1-blend_map)
        if original <= 0.5: retouched = (2*original) * blend_map
        """
        mask = original > 0.5
        result_high = 1.0 - (1.0 - 2.0 * (original - 0.5)) * (1.0 - blend_map)
        result_low = (2.0 * original) * blend_map
        result = torch.where(mask, result_high, result_low)
        return torch.clamp(result, 0.0, 1.0)
    
    def compute_blend_map(self, original, retouched):
        """
        For Overlay mode:
        if original > 0.5: retouched = 1 - (1-2*(original-0.5)) * (1-blend_map)
                         => (1-blend_map) = (1-retouched) / (1-2*(original-0.5))
                         => blend_map = 1 - (1-retouched) / (1-2*(original-0.5))
        
        if original <= 0.5: retouched = (2*original) * blend_map
                          => blend_map = retouched / (2*original)
        """
        blend_map = torch.zeros_like(original)
        
        # Case 1: original > 0.5
        mask_high = original > 0.5
        
        # For high values, compute using the screen-like formula
        numerator_high = 1.0 - retouched
        denominator_high = 1.0 - 2.0 * (original - 0.5)
        
        # Avoid division by zero
        denominator_high = torch.where(
            torch.abs(denominator_high) < EPSILON,
            torch.ones_like(denominator_high) * EPSILON * torch.sign(denominator_high + EPSILON),
            denominator_high
        )
        
        blend_map_high = 1.0 - numerator_high / denominator_high
        
        # Case 2: original <= 0.5
        mask_low = ~mask_high
        
        # For low values, compute using the multiply-like formula
        numerator_low = retouched
        denominator_low = 2.0 * original
        
        # Avoid division by zero
        denominator_low = torch.where(
            denominator_low < EPSILON,
            torch.ones_like(denominator_low) * EPSILON,
            denominator_low
        )
        
        blend_map_low = numerator_low / denominator_low
        
        # Combine results
        blend_map = torch.where(mask_high, blend_map_high, blend_map_low)
        
        # Handle special cases
        # When original is 0.5, the equation becomes ambiguous, default to 0.5
        blend_map = torch.where(
            torch.abs(original - 0.5) < EPSILON,
            torch.ones_like(blend_map) * 0.5,
            blend_map
        )
        
        return torch.clamp(blend_map, 0.0, 1.0)


class SoftLightMode(BlendMode):
    """Soft Light blending mode"""
    def __init__(self):
        super().__init__("Soft Light")
    
    def blend(self, original, blend_map):
        """
        Blend using the approximate formula: 
        if blend_map > 0.5: retouched = 1 - (1-original) * (1-(blend_map-0.5))
        if blend_map <= 0.5: retouched = original * (blend_map+0.5)
        """
        mask = blend_map > 0.5
        result_high = 1.0 - (1.0 - original) * (1.0 - (blend_map - 0.5))
        result_low = original * (blend_map + 0.5)
        result = torch.where(mask, result_high, result_low)
        return torch.clamp(result, 0.0, 1.0)
    
    def compute_blend_map(self, original, retouched):
        """
        For Soft Light mode (approximate formula):
        
        This is an iterative approach due to the complexity of solving for blend_map analytically.
        We use a simple optimization approach to find the blend_map values.
        """
        # Initialize blend_map with a value of 0.5 (neutral)
        blend_map = torch.ones_like(original) * 0.5
        
        # Learning rate for optimization
        lr = 0.1
        
        # Number of iterations for optimization
        num_iterations = 10
        
        for i in range(num_iterations):
            # Forward pass: compute the blended result
            current_result = self.blend(original, blend_map)
            
            # Compute error
            error = retouched - current_result
            
            # Update blend_map using gradient descent
            # Simple update rule: increase blend_map if result < target, decrease if result > target
            blend_map = blend_map + lr * error
            
            # Clamp blend_map to [0, 1]
            blend_map = torch.clamp(blend_map, 0.0, 1.0)
            
            # Reduce learning rate over time
            lr *= 0.9
        
        return blend_map


class HardLightMode(BlendMode):
    """Hard Light blending mode"""
    def __init__(self):
        super().__init__("Hard Light")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: 
        if blend_map > 0.5: retouched = 1 - (1-original) * (1-2*(blend_map-0.5))
        if blend_map <= 0.5: retouched = original * (2*blend_map)
        """
        mask = blend_map > 0.5
        result_high = 1.0 - (1.0 - original) * (1.0 - 2.0 * (blend_map - 0.5))
        result_low = original * (2.0 * blend_map)
        result = torch.where(mask, result_high, result_low)
        return torch.clamp(result, 0.0, 1.0)
    
    def compute_blend_map(self, original, retouched):
        """
        For Hard Light mode:
        
        Similar to Overlay but with blend_map and original switched.
        We use an iterative approach due to the complexity.
        """
        # Initialize blend_map with a value of 0.5 (neutral)
        blend_map = torch.ones_like(original) * 0.5
        
        # Learning rate for optimization
        lr = 0.1
        
        # Number of iterations for optimization
        num_iterations = 10
        
        for i in range(num_iterations):
            # Forward pass: compute the blended result
            current_result = self.blend(original, blend_map)
            
            # Compute error
            error = retouched - current_result
            
            # Update blend_map using gradient descent
            blend_map = blend_map + lr * error
            
            # Clamp blend_map to [0, 1]
            blend_map = torch.clamp(blend_map, 0.0, 1.0)
            
            # Reduce learning rate over time
            lr *= 0.9
        
        return blend_map


class VividLightMode(BlendMode):
    """Vivid Light blending mode"""
    def __init__(self):
        super().__init__("Vivid Light")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: 
        if blend_map > 0.5: retouched = 1 - (1-original) / (2*(blend_map-0.5))
        if blend_map <= 0.5: retouched = original / (1-2*blend_map)
        """
        # Avoid division by zero
        denom_high = 2.0 * (blend_map - 0.5)
        denom_high = torch.where(
            denom_high < EPSILON,
            torch.ones_like(denom_high) * EPSILON,
            denom_high
        )
        
        denom_low = 1.0 - 2.0 * blend_map
        denom_low = torch.where(
            denom_low < EPSILON,
            torch.ones_like(denom_low) * EPSILON,
            denom_low
        )
        
        # Compute both parts
        result_high = 1.0 - (1.0 - original) / denom_high
        result_low = original / denom_low
        
        # Combine using mask
        mask = blend_map > 0.5
        result = torch.where(mask, result_high, result_low)
        
        return torch.clamp(result, 0.0, 1.0)
    
    def compute_blend_map(self, original, retouched):
        """
        For Vivid Light mode:
        
        Due to the complexity of solving for blend_map analytically,
        we use an optimization approach.
        """
        # Initialize blend_map with a value of 0.5 (neutral)
        blend_map = torch.ones_like(original) * 0.5
        
        # Learning rate for optimization
        lr = 0.1
        
        # Number of iterations for optimization
        num_iterations = 15
        
        for i in range(num_iterations):
            # Forward pass: compute the blended result
            current_result = self.blend(original, blend_map)
            
            # Compute error
            error = retouched - current_result
            
            # Update blend_map using gradient descent
            blend_map = blend_map + lr * error
            
            # Clamp blend_map to [0, 1]
            blend_map = torch.clamp(blend_map, 0.0, 1.0)
            
            # Reduce learning rate over time
            lr *= 0.85
        
        return blend_map


class LinearLightMode(BlendMode):
    """Linear Light blending mode"""
    def __init__(self):
        super().__init__("Linear Light")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: 
        if blend_map > 0.5: retouched = original + 2*(blend_map-0.5)
        if blend_map <= 0.5: retouched = original + 2*blend_map - 1
        """
        mask = blend_map > 0.5
        result_high = original + 2.0 * (blend_map - 0.5)
        result_low = original + 2.0 * blend_map - 1.0
        result = torch.where(mask, result_high, result_low)
        return torch.clamp(result, 0.0, 1.0)
    
    def compute_blend_map(self, original, retouched):
        """
        For Linear Light mode:
        
        if blend_map > 0.5: retouched = original + 2*(blend_map-0.5)
                         => blend_map = (retouched - original)/2 + 0.5
        
        if blend_map <= 0.5: retouched = original + 2*blend_map - 1
                          => blend_map = (retouched - original + 1)/2
        """
        # Try both formulas and select the one that gives blend_map in [0, 1]
        blend_map_high = (retouched - original) / 2.0 + 0.5
        blend_map_low = (retouched - original + 1.0) / 2.0
        
        # Check which formula gives valid results
        mask_high_valid = (blend_map_high > 0.5) & (blend_map_high <= 1.0)
        mask_low_valid = (blend_map_low >= 0.0) & (blend_map_low <= 0.5)
        
        # Combine results
        blend_map = torch.where(
            mask_high_valid,
            blend_map_high,
            torch.where(
                mask_low_valid,
                blend_map_low,
                # Default to 0.5 if neither is valid
                torch.ones_like(retouched) * 0.5
            )
        )
        
        # Handle special case where both are valid (should be rare)
        mask_both_valid = mask_high_valid & mask_low_valid
        blend_map = torch.where(
            mask_both_valid,
            # Use the average in this case
            (blend_map_high + blend_map_low) / 2.0,
            blend_map
        )
        
        return torch.clamp(blend_map, 0.0, 1.0)


class PinLightMode(BlendMode):
    """Pin Light blending mode"""
    def __init__(self):
        super().__init__("Pin Light")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: 
        if blend_map > 0.5: retouched = max(original, 2*(blend_map-0.5))
        if blend_map <= 0.5: retouched = min(original, 2*blend_map)
        """
        mask = blend_map > 0.5
        result_high = torch.maximum(original, 2.0 * (blend_map - 0.5))
        result_low = torch.minimum(original, 2.0 * blend_map)
        result = torch.where(mask, result_high, result_low)
        return result
    
    def compute_blend_map(self, original, retouched):
        """
        For Pin Light mode:
        
        Due to the max/min operations, this is difficult to solve analytically.
        We use an iterative approach instead.
        """
        # Initialize blend_map with a value of 0.5 (neutral)
        blend_map = torch.ones_like(original) * 0.5
        
        # Learning rate for optimization
        lr = 0.1
        
        # Number of iterations for optimization
        num_iterations = 10
        
        for i in range(num_iterations):
            # Forward pass: compute the blended result
            current_result = self.blend(original, blend_map)
            
            # Compute error
            error = retouched - current_result
            
            # Update blend_map using gradient descent
            blend_map = blend_map + lr * error
            
            # Clamp blend_map to [0, 1]
            blend_map = torch.clamp(blend_map, 0.0, 1.0)
            
            # Reduce learning rate over time
            lr *= 0.9
        
        return blend_map


class DifferenceMode(BlendMode):
    """Difference blending mode: |Target - Blend|"""
    def __init__(self):
        super().__init__("Difference")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: retouched = |original - blend_map|
        """
        result = torch.abs(original - blend_map)
        return result
    
    def compute_blend_map(self, original, retouched):
        """
        For Difference mode: retouched = |original - blend_map|
        
        This gives two possible solutions:
        1. blend_map = original - retouched (if original ≥ retouched)
        2. blend_map = original + retouched (if original < retouched)
        
        We need to choose the solution that gives blend_map in [0, 1]
        """
        # Calculate both potential solutions
        blend_map1 = original - retouched  # For original ≥ retouched
        blend_map2 = original + retouched  # For original < retouched
        
        # Check which solution gives values in [0, 1]
        mask1_valid = (blend_map1 >= 0.0) & (blend_map1 <= 1.0)
        mask2_valid = (blend_map2 >= 0.0) & (blend_map2 <= 1.0)
        
        # Default to 0.5 if neither solution works
        blend_map = torch.ones_like(original) * 0.5
        
        # Use solution 1 where valid
        blend_map = torch.where(mask1_valid, blend_map1, blend_map)
        
        # Use solution 2 where valid (and solution 1 is not valid)
        blend_map = torch.where(mask2_valid & ~mask1_valid, blend_map2, blend_map)
        
        # If both solutions are valid, prefer solution 1
        
        return blend_map


class ExclusionMode(BlendMode):
    """Exclusion blending mode: 0.5 - 2*(Target-0.5)*(Blend-0.5)"""
    def __init__(self):
        super().__init__("Exclusion")
    
    def blend(self, original, blend_map):
        """
        Blend using the formula: retouched = 0.5 - 2*(original-0.5)*(blend_map-0.5)
        """
        result = 0.5 - 2.0 * (original - 0.5) * (blend_map - 0.5)
        return result
    
    def compute_blend_map(self, original, retouched):
        """
        For Exclusion mode: retouched = 0.5 - 2*(original-0.5)*(blend_map-0.5)
        => 2*(original-0.5)*(blend_map-0.5) = 0.5 - retouched
        => (blend_map-0.5) = (0.5 - retouched) / (2*(original-0.5))
        => blend_map = 0.5 + (0.5 - retouched) / (2*(original-0.5))
        
        Need to handle division by zero when original is 0.5
        """
        # Calculate intermediate values
        numerator = 0.5 - retouched
        denominator = 2.0 * (original - 0.5)
        
        # Avoid division by zero
        denominator = torch.where(
            torch.abs(denominator) < EPSILON,
            torch.ones_like(denominator) * EPSILON * torch.sign(denominator + EPSILON),
            denominator
        )
        
        blend_map = 0.5 + numerator / denominator
        
        # When original is 0.5, the blend_map doesn't affect the result
        # So we can set it to any value, default to 0.5
        blend_map = torch.where(
            torch.abs(original - 0.5) < EPSILON,
            torch.ones_like(blend_map) * 0.5,
            blend_map
        )
        
        return torch.clamp(blend_map, 0.0, 1.0)


#################################
# Helper Functions
#################################

def read_and_preprocess_images(original_path, retouched_path, resize_dim=(1024, 1024)):
    """
    Read and preprocess original and retouched images
    
    Args:
        original_path: Path to original image
        retouched_path: Path to retouched image
        resize_dim: Resize dimensions
    
    Returns:
        original: Original image as numpy array [0,1]
        retouched: Retouched image as numpy array [0,1]
        original_tensor: Original image as tensor [C, H, W]
        retouched_tensor: Retouched image as tensor [C, H, W]
    """
    # Read original image
    original = cv2.imread(original_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original = cv2.resize(original, resize_dim)
    original = original.astype(np.float32) / 255.0
    
    # Read retouched image
    retouched = cv2.imread(retouched_path)
    retouched = cv2.cvtColor(retouched, cv2.COLOR_BGR2RGB)
    retouched = cv2.resize(retouched, resize_dim)
    retouched = retouched.astype(np.float32) / 255.0
    
    # Convert to tensors
    original_tensor = torch.from_numpy(original).permute(2, 0, 1)  # [C, H, W]
    retouched_tensor = torch.from_numpy(retouched).permute(2, 0, 1)  # [C, H, W]
    
    return original, retouched, original_tensor, retouched_tensor


def calculate_metrics(retouched, reconstructed):
    """
    Calculate quality metrics (MSE, PSNR) between retouched and reconstructed images
    
    Args:
        retouched: Ground truth retouched image [0,1]
        reconstructed: Reconstructed image [0,1]
    
    Returns:
        mse: Mean Squared Error
        psnr: Peak Signal-to-Noise Ratio (dB)
    """
    mse = np.mean((retouched - reconstructed) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    return mse, psnr


def visualize_blend_modes(original, retouched, blend_mode_results, output_path=None):
    """
    Visualize original, retouched, and reconstructed images for multiple blend modes
    
    Args:
        original: Original image as numpy array [H, W, C]
        retouched: Retouched image as numpy array [H, W, C]
        blend_mode_results: Dictionary of blend mode results
        output_path: Path to save visualization
    """
    num_modes = len(blend_mode_results)
    num_rows = math.ceil(num_modes / 3) + 1  # +1 for original and retouched
    
    plt.figure(figsize=(18, 4 * num_rows))
    
    # Plot original and retouched images
    plt.subplot(num_rows, 3, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(num_rows, 3, 2)
    plt.imshow(retouched)
    plt.title('Retouched Image')
    plt.axis('off')
    
    # Plot reconstructed images and blend maps for each mode
    for i, (mode_name, result) in enumerate(blend_mode_results.items()):
        reconstructed = result['reconstructed']
        blend_map = result['blend_map']
        psnr = result['psnr']
        
        # Plot reconstructed image
        plt.subplot(num_rows, 3, i + 4)  # Start from the second row
        plt.imshow(reconstructed)
        plt.title(f'{mode_name} (PSNR: {psnr:.2f} dB)')
        plt.axis('off')
        
        # If the blend map has 3 channels, average them for visualization
        if blend_map.shape[2] == 3:
            blend_map_vis = np.mean(blend_map, axis=2)
        else:
            blend_map_vis = blend_map[:, :, 0]
        
        # Add a small subplot for the blend map in the corner of the reconstructed image
        # This could be improved with a more elaborate visualization
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
    
    plt.show()


def compare_blend_modes(original_dir, retouched_dir, output_dir, test_file=None, resize_dim=(1024, 1024)):
    """
    Compare different blend modes on original and retouched image pairs
    
    Args:
        original_dir: Directory containing original images
        retouched_dir: Directory containing retouched images
        output_dir: Directory to save results
        test_file: Specific file to test (None to use the first file)
        resize_dim: Resize dimensions for input images
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get file names in original directory
    file_names = [f for f in os.listdir(original_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if test_file:
        if test_file in file_names:
            file_names = [test_file]
        else:
            print(f"Warning: Test file {test_file} not found. Using the first file instead.")
            file_names = [file_names[0]]
    else:
        # Use the first file for testing
        file_names = [file_names[0]]
    
    # Initialize blend modes
    blend_modes = [
        CustomMode(),
        DarkenMode(),
        MultiplyMode(),
        ColorBurnMode(),
        LinearBurnMode(),
        LightenMode(),
        ScreenMode(),
        ColorDodgeMode(),
        LinearDodgeMode(),
        OverlayMode(),
        SoftLightMode(),
        HardLightMode(),
        VividLightMode(),
        LinearLightMode(),
        PinLightMode(),
        DifferenceMode(),
        ExclusionMode()
    ]
    
    for file_name in tqdm(file_names, desc="Processing images"):
        original_path = os.path.join(original_dir, file_name)
        retouched_path = os.path.join(retouched_dir, file_name)
        
        # Check if retouched image exists
        if not os.path.exists(retouched_path):
            print(f"Warning: Retouched image not found for {file_name}")
            continue
        
        # Read and preprocess images
        original, retouched, original_tensor, retouched_tensor = read_and_preprocess_images(
            original_path, retouched_path, resize_dim
        )
        
        # Process each blend mode
        results = {}
        
        for blend_mode in tqdm(blend_modes, desc=f"Testing blend modes for {file_name}"):
            # Compute blend map
            blend_map = blend_mode.compute_blend_map(original_tensor, retouched_tensor)
            
            # Reconstruct the retouched image
            reconstructed_tensor = blend_mode.blend(original_tensor, blend_map)
            
            # Convert to numpy for metrics and visualization
            blend_map_np = blend_map.permute(1, 2, 0).numpy()  # [H, W, C]
            reconstructed_np = reconstructed_tensor.permute(1, 2, 0).numpy()  # [H, W, C]
            
            # Calculate metrics
            mse, psnr = calculate_metrics(retouched, reconstructed_np)
            
            # Store results
            results[blend_mode.name] = {
                'blend_map': blend_map_np,
                'reconstructed': reconstructed_np,
                'mse': mse,
                'psnr': psnr
            }
            
            print(f"{blend_mode.name} - PSNR: {psnr:.2f} dB")
        
        # Visualize results
        output_path = os.path.join(output_dir, f"comparison_{file_name.split('.')[0]}.png")
        visualize_blend_modes(original, retouched, results, output_path)
        
        # Create a summary of PSNR results
        psnr_results = {name: res['psnr'] for name, res in results.items()}
        psnr_sorted = sorted(psnr_results.items(), key=lambda x: x[1], reverse=True)
        
        print("\nPSNR Ranking:")
        for i, (name, psnr) in enumerate(psnr_sorted):
            print(f"{i+1}. {name}: {psnr:.2f} dB")


if __name__ == "__main__":
    original_dir = "src/"
    retouched_dir = "gt/"
    output_dir = "blend_mode_comparisons/"
    resize_dim = (1024, 1024)
    
    # Optionally specify a test file
    test_file = None  # Set to None to use the first file
    
    compare_blend_modes(original_dir, retouched_dir, output_dir, test_file, resize_dim)
