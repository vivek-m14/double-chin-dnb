# Model Improvements — Next Steps

> **Date**: 2025-04-09  
> **Status**: Planning  
> **Context**: The blend-map pipeline is architecturally sound (see [analysis_notes.md](analysis_notes.md) and [adversarial_audit.md](adversarial_audit.md)). This document captures the next round of improvements, prioritized by impact.

---

## Table of Contents

1. [Core Problem Statement](#1-core-problem-statement)
2. [Improvement 1: Edit-Region-Weighted Loss](#2-improvement-1-edit-region-weighted-loss)
3. [Improvement 2: Chin Mask as 4th Input Channel](#3-improvement-2-chin-mask-as-4th-input-channel)
4. [Improvement 3: ROI Crop Pipeline](#4-improvement-3-roi-crop-pipeline)
5. [Improvement 4: Attention Gates in U-Net](#5-improvement-4-attention-gates-in-u-net)
6. [Improvement 5: Chin-Centered Crop Augmentation](#6-improvement-5-chin-centered-crop-augmentation)
7. [Improvement 6: Multi-Resolution Input](#7-improvement-6-multi-resolution-input)
8. [Priority Matrix](#8-priority-matrix)
9. [Experiment Plan](#9-experiment-plan)

---

## 1. Core Problem Statement

Data analysis (1,500 images, see [analysis_notes.md §4](analysis_notes.md)) established:

| Metric | Value |
|---|---|
| Mean edit area | **3.0%** of image pixels |
| Mean edit magnitude | **9.0 / 255** (0.035) |
| Edit type | 85% luminance (shadow/brightness), 15% color |
| Blend map neutral value | 0.5 (no change) |
| Target blend map range | 99.987% of pixels within [0.2, 0.8] |

**The fundamental imbalance**: The model sees 1024×1024 = 1,048,576 pixels per image, but only ~31,000 pixels (~3%) carry any meaningful edit signal. The remaining 97% of pixels have a target blend map value of exactly 0.5 (identity). This creates two problems:

1. **Capacity waste** — 97% of the network's representational capacity is spent learning to output 0.5 for background/forehead/hair.
2. **Metric deception** — A model that outputs constant 0.5 everywhere already achieves ~35–40 dB PSNR (the "do-nothing" baseline flagged in the [adversarial audit, C1](adversarial_audit.md)).

All improvements below address this core imbalance from different angles.

---

## 2. Improvement 1: Edit-Region-Weighted Loss

### Problem

The current `CombinedLoss` computes uniform MSE across all pixels. A 0.01 error at an untouched forehead pixel is penalized identically to a 0.01 error in the chin where the actual edit occurs. This lets the model converge toward 0.5 everywhere — minimizing background error at the expense of learning the actual retouching.

### Solution

Weight the loss spatially using the target blend map itself to identify edit regions:

```python
# Derive edit mask from target blend map
edit_weight = (target_blend_map - 0.5).abs()          # high where edits occur
edit_weight = edit_weight / (edit_weight.mean() + eps) # normalize so mean weight ≈ 1
edit_weight = edit_weight.clamp(min=1.0)               # background stays at 1.0, edit region upweighted

# Weighted MSE
blend_map_loss = (edit_weight * (pred - target) ** 2).mean()
```

Alternatively, a simpler binary approach:

```python
edit_mask = (target_blend_map - 0.5).abs() > noise_threshold  # e.g., 0.02
fg_loss = mse(pred[edit_mask], target[edit_mask])
bg_loss = mse(pred[~edit_mask], target[~edit_mask])
blend_map_loss = λ_fg * fg_loss + λ_bg * bg_loss  # e.g., λ_fg=5.0, λ_bg=1.0
```

### Impact

- Forces the model to actually learn the chin edit rather than gaming the metric with near-constant output.
- Directly addresses the "do-nothing" baseline problem (adversarial audit C1).

### Files to change

- [src/losses/losses.py](../src/losses/losses.py) — add `WeightedMSELoss` or modify `CombinedLoss`
- [blend_map.yaml](../blend_map.yaml) — add `lambda_fg_weight` config parameter

### Effort: **Low** | Inference change: **None**

---

## 3. Improvement 2: Chin Mask as 4th Input Channel

### Problem

The model receives only the 3-channel RGB image. It must implicitly learn (a) where the chin is and (b) what edit to apply. Combining spatial localization and edit prediction in one network wastes capacity — the model doesn't know *where* to focus.

### Solution

Compute a soft face-parsing mask highlighting the chin/jaw region and feed it as a 4th input channel:

```
Model input: [R, G, B, chin_mask]  →  4-channel input
chin_mask: 1.0 in chin/jaw region, 0.0 elsewhere (soft edges)
```

The mask can be generated during data preprocessing using:
- **MediaPipe FaceMesh** — landmarks 0-16 (jawline contour) + 152-199 (chin)
- **BiSeNet face parser** — semantic segmentation class for chin/jaw
- **Simple heuristic** — lower-third bounding box from face detection, with Gaussian falloff

The model architecture change is minimal: `InConv(n_channels=3, ...)` → `InConv(n_channels=4, ...)`.

### Impact

- The model immediately knows *where* to apply the edit, freeing all capacity for *what* edit to apply.
- No cropping/pasting pipeline needed at inference — just generate the mask and run the model.
- Naturally handles varying chin positions across poses.

### Files to change

- [src/models/unet.py](../src/models/unet.py) — `n_channels=4`
- [src/data/dataset.py](../src/data/dataset.py) — generate/load chin mask, concatenate as 4th channel
- [blend_map.yaml](../blend_map.yaml) — add `use_chin_mask: true`, mask generation config
- Inference pipeline — add mask generation step

### Effort: **Medium** | Inference change: **Yes** (minor — need mask generation at inference)

---

## 4. Improvement 3: ROI Crop Pipeline

### Problem

Processing the full 1024×1024 image when only the chin region (~3% of area) changes is computationally wasteful and dilutes the learning signal. The model's receptive field (200×200 px) is only marginally adequate for the chin contour (300–500 px wide).

### Solution

Crop to the chin region before the model, predict a blend map on the crop, paste back into a neutral (0.5) full-size blend map.

**Training pipeline:**

```
1. Detect face landmarks (MediaPipe / dlib)
2. Compute jaw-chin bounding box
3. Expand by 30-50% for context
4. Crop original, GT, and blend map to this region
5. Resize crop to 256×256 or 512×512
6. Train model on crops only
```

**Inference pipeline:**

```
1. Detect face → compute chin ROI
2. Crop input image to ROI
3. Run model → predict blend map on crop
4. Create full-size blend map initialized to 0.5
5. Paste predicted crop blend map into ROI (with feathered edges)
6. Apply full blend formula: retouched = clamp(image + 2*blend_map - 1, 0, 1)
```

### Advantages

- All model capacity focused on the edit region
- Smaller input → faster training and inference
- Larger effective receptive field relative to edit region
- Can use a smaller model (fewer params)

### Disadvantages

- Requires a landmark detector at both training and inference time
- Feathered-edge blending at ROI boundaries needs tuning
- Multi-face handling adds complexity
- Fails gracefully only if landmark detection works

### Files to change

- [src/data/dataset.py](../src/data/dataset.py) — add ROI crop logic to `BlendMapDataset.__getitem__`
- [src/models/unet.py](../src/models/unet.py) — potentially smaller architecture for smaller input
- [inference.py](../inference.py) — add crop/paste pipeline with feathered edges
- New utility module for face landmark-based ROI extraction

### Effort: **High** | Inference change: **Yes** (significant)

---

## 5. Improvement 4: Attention Gates in U-Net

### Problem

Standard skip connections in the U-Net blindly concatenate encoder features (including background, hair, forehead) into the decoder. With only 3% of pixels being edited, most skip-connection features are irrelevant noise for the decoder.

### Solution

Add attention gates (Oktay et al., 2018) to the skip connections. The decoder learns to suppress irrelevant spatial regions and amplify the chin area:

```python
class AttentionGate(nn.Module):
    """Attention gate for skip connections.
    Suppresses irrelevant regions, amplifies features in the edit region."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, 1, bias=True)
        self.psi = nn.Conv2d(F_int, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        # g: gating signal (decoder), x: skip connection (encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi
```

This is the same sparse-target problem attention U-Nets were invented for in medical imaging (small lesions in large scans).

### Impact

- ~25% parameter increase (~8M total vs 6.3M) but much better feature utilization.
- Decoder naturally learns to focus on the chin without explicit mask input.
- Proven technique in medical segmentation with similar sparsity characteristics.

### Files to change

- [src/models/unet.py](../src/models/unet.py) — add `AttentionGate`, modify `Up` blocks

### Effort: **Medium** | Inference change: **None**

---

## 6. Improvement 5: Chin-Centered Crop Augmentation

### Problem

Current augmentations (horizontal flip + rotation) don't change the spatial distribution of edit regions within the frame. The chin is always in the same relative position — lower-center of the image. The model can learn a spatial prior (always predict 0.5 in the top half) instead of learning visual features.

### Solution

Add random crop-and-zoom centered on the chin region during training:

```python
# During augmentation:
if random() < 0.3:
    # Random crop centered on lower-third of face
    crop_size = randint(512, 900)
    cx, cy = face_center_x + randint(-50, 50), face_center_y + randint(50, 200)
    # Crop image, blend_map, GT identically
    # Resize back to training resolution
```

This forces the edit region to appear at different locations and scales within the training frame, breaking the spatial prior.

### Impact

- Prevents the model from learning a fixed spatial prior
- Increases effective diversity of the edit region's position and scale
- Simple to implement, no inference change

### Files to change

- [src/data/dataset.py](../src/data/dataset.py) — add to `apply_augmentation()`

### Effort: **Low** | Inference change: **None**

---

## 7. Improvement 6: Multi-Resolution Input

### Problem

At 1024×1024, the model must encode both global context (face shape, lighting direction) and local detail (chin texture, shadow edges) through a single encoder path. The bottleneck at 64×64 can capture global context but discards local detail; skip connections preserve local detail but lack global context.

### Solution

Two-branch encoder that processes the image at two scales:

```
Branch 1 (global):  Full image @ 256×256 → lightweight encoder → global features
Branch 2 (local):   Chin crop @ 512×512 → main encoder → local features
Fusion:             Concatenate at bottleneck → shared decoder → blend map on crop
```

### Impact

- Best quality: global illumination awareness + local detail preservation
- Highest complexity and engineering effort

### Files to change

- [src/models/unet.py](../src/models/unet.py) — new two-branch architecture
- [src/data/dataset.py](../src/data/dataset.py) — dual-scale data loading
- Full training and inference pipeline changes

### Effort: **High** | Inference change: **Yes**

---

## 8. Priority Matrix

| # | Improvement | Impact | Effort | Inference Change | Recommended Order |
|---|---|---|---|---|---|
| 1 | **Edit-region-weighted loss** | 🔴 High | Low | None | **Do first** |
| 2 | **Chin mask as 4th channel** | 🔴 High | Medium | Minor | **Do second** |
| 5 | **Chin-centered crop augmentation** | 🟡 Medium | Low | None | **Do with #1** |
| 4 | **Attention gates** | 🟠 Med-High | Medium | None | **Do third** |
| 3 | **ROI crop pipeline** | 🔴 High | High | Significant | **Do fourth** (or skip if #2+#4 suffice) |
| 6 | **Multi-resolution input** | 🟡 Medium | High | Yes | **Future / if needed** |

### Recommended implementation phases

**Phase 1 — Quick wins (1–2 days):**
- Edit-region-weighted loss (#1)
- Chin-centered crop augmentation (#5)
- Log "do-nothing" baseline PSNR at epoch 0 for honest evaluation

**Phase 2 — Spatial awareness (3–5 days):**
- Chin mask as 4th input channel (#2)
- Attention gates in U-Net (#4)

**Phase 3 — Full ROI pipeline (1–2 weeks):**
- ROI crop pipeline (#3) — only if Phase 1+2 results are insufficient
- Multi-resolution input (#6) — only if Phase 3 results are insufficient

---

## 9. Experiment Plan

### Ablation experiments (one change at a time from current best config):

| Experiment | Config Base | Change |
|---|---|---|
| `exp4_weighted_loss` | `exp2_lite_residual_tanh.yaml` | Add edit-region weighted loss (λ_fg=5.0) |
| `exp5_weighted_loss_crop_aug` | `exp4` | + chin-centered crop augmentation |
| `exp6_chin_mask_input` | `exp5` | + chin mask as 4th input channel |
| `exp7_attention_unet` | `exp5` | + attention gates (no chin mask) |
| `exp8_attention_chin_mask` | `exp6` + `exp7` | Both chin mask + attention gates |

### Evaluation protocol

For every experiment, log:
1. Standard metrics: total loss, PSNR, SSIM (full image)
2. **Edit-region metrics**: PSNR / SSIM computed only on pixels where `|B_target - 0.5| > 0.02`
3. **Do-nothing baseline delta**: `metric - baseline_metric` at every epoch
4. Visual inspection of 20 test images — check for artifacts at edit boundaries

The edit-region metrics are the **primary** evaluation criterion. Full-image PSNR is misleading due to the 97% identity region dominating the metric.
