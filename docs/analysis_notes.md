# Double Chin Removal — Full Analysis Notes

> Generated from a comprehensive code review + data analysis session.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Training Pipeline Walkthrough](#2-training-pipeline-walkthrough)
3. [Data Format & Structure](#3-data-format--structure)
4. [Data Analysis: Blend Map Viability](#4-data-analysis-blend-map-viability)
5. [Network Architecture Analysis](#5-network-architecture-analysis)
6. [Pose Diversity Analysis](#6-pose-diversity-analysis)
7. [CoreML Deployment Benchmark](#7-coreml-deployment-benchmark)
8. [Training Improvements (15 Issues)](#8-training-improvements-15-issues)
9. [Final Recommendations](#9-final-recommendations)

---

## 1. Project Overview

A PyTorch deep learning project for removing double chins from portrait images using **blend maps** (not direct pixel manipulation).

### Pipeline

1. A U-Net (`BaseUNetHalf`) takes an RGB image (1024×1024) and predicts a 3-channel blend map ∈ [0,1] (sigmoid output).
2. The blend map is applied via a **Linear Light formula**:
   ```
   retouched = clamp(image + 2 * blend_map - 1, 0, 1)
   ```
   A neutral (no-change) blend map is all **0.5**.
3. Target blend maps for training are derived from (original, edited) image pairs using the inverse formula:
   ```
   blend_map = (edited - original + 1) / 2
   ```
   They are cached as `.npy` files under `blendmap_image/`.

### Key Files

| File | Purpose |
|---|---|
| `src/models/unet.py` | `BaseUNetHalf` (active) and `UNet` (legacy) architectures |
| `src/blend/blend_map.py` | `apply_blend_formula`, `compute_target_blend_map`, flow utils |
| `src/data/dataset.py` | `BlendMapDataset`, `TensorMapDataset`, `create_data_loaders()` |
| `src/losses/losses.py` | `CombinedLoss` (blend MSE + image MSE + VGG perceptual + TV) |
| `src/utils/utils_blend.py` | Checkpoint loading, visualization, metrics (blend-map training) |
| `tools/train_blend_map.py` | DDP training entry point — reads `blend_map.yaml` |
| `inference.py` | `DoubleChinRemover` class + `run_inference()` + CLI |
| `blend_map.yaml` | Primary training config |

### Legacy vs Active

- **Active path**: blend maps (`tools/train_blend_map.py`, `BlendMapDataset`, `blend_map.yaml`)
- **Legacy path**: flow-based (`tools/train.py`, `TensorMapDataset`, `default.yaml`) — predicts 2-channel displacement fields with tanh output

---

## 2. Training Pipeline Walkthrough

### Step-by-step

1. **Config**: `blend_map.yaml` defines all hyperparameters
2. **Launch**: `python tools/train_blend_map.py` from repo root
3. **DDP**: Uses `torch.multiprocessing.spawn` with gloo backend across all visible GPUs
4. **Dataset**: `BlendMapDataset` loads JSON manifest → reads original/edited pairs → computes/caches blend maps as `.npy`
5. **Augmentations**: Horizontal flip + rotation (applied identically to image, blend map, and GT). Blend map rotation fills with 0.5 (neutral)
6. **Model**: `BaseUNetHalf(n_channels=3, n_classes=3, last_layer_activation="sigmoid")`
7. **Loss**: `CombinedLoss` with 4 terms (see below)
8. **Optimizer**: Adam, lr=1e-4, StepLR(step_size=50, gamma=0.1)
9. **Checkpoints**: Best saved as `double_chin_bmap_best.pth`; periodic as `double_chin_bmap_epoch_N.pth`

### Current Config (`blend_map.yaml`)

```yaml
num_epochs: 200
batch_size: 4
learning_rate: 0.0001
img_size: 1024
num_workers: 8
lambda_blend_mse: 1.0
lambda_image_mse: 1.0
lambda_perc: 0.1
lambda_tv: 0.1
lr_step_size: 50
lr_gamma: 0.1
start_epoch: 106  # resuming from epoch 106
```

### Loss Function (`CombinedLoss`)

| Term | Weight | Description |
|---|---|---|
| `blend_map_loss` | 1.0 | MSE between predicted and target blend maps |
| `image_mse_loss` | 1.0 | MSE between retouched output and ground truth |
| `perc_loss` | 0.1 | VGG16 multi-layer perceptual loss (frozen weights, layers 4/9/16/23) |
| `tv_loss` | 0.1 | Total variation on the blend map (smoothness regularizer) |

---

## 3. Data Format & Structure

### JSON Manifest

The JSON file contains 9,437 entries. **Only 3 fields are used by the code**:

```json
{
  "original_image": "double_chin_data_v3/original/filename.jpg",
  "edited_image": "double_chin_data_v3/edited/filename.jpg",
  "source": "pegu_double_chin"
}
```

All other fields (`diff_mask`, `cropped_*`, `bbox`, etc.) are present in the JSON but **ignored** by `BlendMapDataset`.

### Data Root Mapping

- `data_root` (in YAML) points to: `.../Double_Chin/double_chin_images/`
- JSON paths are relative to `data_root`: `data_root + "/" + entry["original_image"]`

### Source Breakdown

| Source | Count | Status |
|---|---|---|
| `pegu_double_chin` | 3,400 | ✅ Used |
| `prd-update` | 3,880 | ✅ Used |
| `aadan_double_chin` | 2,157 | ❌ Filtered out in code |
| **Usable total** | **7,280** | |

---

## 4. Data Analysis: Blend Map Viability

> Full analysis in `data_analysis.ipynb` — **1,500 images analyzed** (20.6% of dataset).

### Core Question

Can an additive blend map faithfully represent the original→edited edits, or are the edits geometric (pixel displacement) that a blend map fundamentally cannot capture?

### Key Findings (1,500-image analysis)

| Metric | Value | Implication |
|---|---|---|
| **Reconstruction PSNR** | Mean **147.9 dB**, 100% > 60 dB | Formula captures edits with **zero** measurable error |
| **Optical Flow** | Mean **0.042 px**, only **3.9%** images with >1% displaced pixels | Edits are almost entirely non-geometric |
| **Luminance Energy** | Mean **85.0%**, 45.6% images >90% luminance-only | Edits are overwhelmingly shadow/brightness changes |
| **Edit Extent** | Mean **3.0%** area changed, **9.0/255** magnitude | Small, subtle, localized edits |
| **Blend Strength Ablation** | PSNR peaks at exactly **α = 1.0** | Formula is mathematically correct |
| **Darkening vs Brightening** | 3.24% darken vs 0.78% brighten | Predominantly shadow addition |
| **Edit Intensity (P95)** | 31.2/255 | Even at the 95th percentile, edits are subtle |

### Per-Source Breakdown (1,500 images)

| Source | Count | PSNR (dB) | Flow > 2px mean | Imgs w/ >1% flow | Luminance % | Edit mag | Changed area |
|---|---|---|---|---|---|---|---|
| `pegu_double_chin` | 700 (46.7%) | 147.8 | 0.14% | 15 | 88.9% | 8.8/255 | 3.4% |
| `prd-update` | 800 (53.3%) | 148.0 | 0.38% | 44 | 81.6% | 9.2/255 | 2.7% |

### Statistical Confidence (n=1,500)

| Metric | Mean ± 95% CI |
|---|---|
| Recon PSNR (dB) | 147.93 ± 0.03 |
| Flow >2px (%) | 0.27 ± 0.11 |
| Luminance energy (%) | 85.00 ± 0.66 |

Coverage: 1,500/7,280 = 20.6% — confidence intervals are tight.

### Dataset Quality Flags

| Flag | Count | % |
|---|---|---|
| PSNR < 40 dB (poor blend map fit) | **0** | 0.0% |
| >5% flow >2px (heavy geometric) | **1** | 0.1% |
| Luminance energy < 50% (color-dominant) | **38** | 2.5% |

### Verdict

**✅ The blend map approach is WELL-SUITED for this dataset.**

- **100%** of 1,500 images have PSNR > 60 dB — the formula is mathematically perfect
- Only **3.9%** of images show any geometric displacement, and only **1 image** (0.1%) has heavy geometric edits
- Edits are overwhelmingly shadow/brightness corrections (85% luminance energy)
- Results are statistically robust with tight 95% confidence intervals

### Recommendations

- Filter out the ~1 image with >5% geometric displacement from training
- The `prd-update` source has slightly more geometric edits (44 images vs 15) and lower luminance purity (81.6% vs 88.9%), but both sources are well-suited
- The 38 images (2.5%) with luminance energy < 50% involve color-channel-specific edits — these are still representable by a 3-channel blend map

---

## 5. Network Architecture Analysis

### `BaseUNetHalf` — Architecture Summary

| Property | Value |
|---|---|
| **Type** | U-Net, 5-level encoder-decoder with skip connections |
| **Channel progression** | 3 → 64 → 128 → 256 → 256 → 256 (bottleneck) |
| **Total params** | **6,314,499** (~6.3M) |
| **Model size** | 25.3 MB (float32) |
| **Bottleneck resolution** | 64×64 (from 1024×1024 input) |
| **Receptive field** | 200×200 px (19.5% of image) |
| **FLOPs** | ~845 GFLOPs |
| **Inference GPU memory** | ~0.9 GB |
| **Training GPU memory** | ~7.1 GB (batch=4) |
| **Params-to-data ratio** | 867 params/image (7,280 images) |

### Layer-by-Layer Breakdown

| Layer | Output Shape | # Params |
|---|---|---|
| inc (InConv) | [1, 64, 1024, 1024] | 38,976 |
| down1 (Down) | [1, 128, 512, 512] | 221,952 |
| down2 (Down) | [1, 256, 256, 256] | 886,272 |
| down3 (Down) | [1, 256, 128, 128] | 1,181,184 |
| down4 (Down) | [1, 256, 64, 64] | 1,181,184 |
| up1 (Up) | [1, 256, 128, 128] | 1,771,008 |
| up2 (Up) | [1, 128, 256, 256] | 738,048 |
| up3 (Up) | [1, 64, 512, 512] | 184,704 |
| up4 (Up) | [1, 64, 1024, 1024] | 110,976 |
| outc (1×1) | [1, 3, 1024, 1024] | 195 |

### Comparison with Full UNet

| | BaseUNetHalf | UNet (full) |
|---|---|---|
| Params | 6.3M | 13.4M |
| Channels | 64→128→256→256→256 | 64→128→256→512→512 |
| Ratio | 0.47x | 1.0x |
| Params/image | 867 | 1,840 |
| Overfit risk | Moderate | High |

### Design Scorecard

| Design Choice | Current | Best Practice | Grade |
|---|---|---|---|
| Architecture family | U-Net | U-Net | ✅ A |
| Depth (encoder levels) | 5 levels | 4-5 levels | ✅ A |
| Channel progression | 64→256 cap | 64→512 | ⚠️ B |
| Skip connections | Concatenation | Concatenation | ✅ A |
| Normalization | BatchNorm | InstanceNorm/GN | ⚠️ B- |
| Activation | ReLU | LeakyReLU/GELU | ⚠️ B |
| Output activation | Sigmoid | Sigmoid | ✅ A |
| Downsampling | MaxPool2d | Strided Conv | ⚠️ B |
| Upsampling | Bilinear | Bilinear | ✅ A |
| Weight init | Kaiming | Kaiming | ✅ A |
| **Attention gates** | ❌ None | ✅ Attention gates | ❌ C |
| **Residual connections** | ❌ None | ✅ Residual blocks | ⚠️ B- |
| **Global residual (input→output skip)** | ❌ None | ✅ Essential | ❌ C |
| Global context (SE/ASPP) | ❌ None | SE blocks | ⚠️ B- |
| Dropout | ❌ None | Optional | ⚠️ B |

### Top 3 Architecture Issues

#### 1. 🔴 No Global Residual Connection (HIGH impact)

Since `blend_map = 0.5` means "no change", the network must learn to output 0.5 for ~97% of pixels. A global residual would let it learn only the **deviation**:

```python
# Current: pred = sigmoid(network(x))          → must output 0.5 everywhere
# Better:  pred = 0.5 + 0.5 * tanh(network(x)) → only learns deviation from neutral
```

This lets the network focus only on the ~3% of pixels that actually need correction.

#### 2. 🟠 No Attention Gates (MEDIUM-HIGH impact)

Only ~3% of pixels are edited. Standard skip connections blindly pass forehead/background features to the decoder. Attention gates (Oktay et al., 2018) would let the decoder suppress irrelevant regions and focus on the chin — the same sparse-target problem attention U-Nets were invented for in medical imaging.

#### 3. 🟡 BatchNorm with Small Batches (MEDIUM impact)

With `batch_size=4` per GPU, BatchNorm statistics are noisy. InstanceNorm is the standard for image-to-image translation at small batch sizes (used in pix2pix, CycleGAN). GroupNorm(32) is also a strong alternative.

### Bottleneck Adequacy

- **Bottleneck resolution**: 64×64 (Nyquist limit = 32 cycles/image)
- **Required frequency range**: 18-36 cycles/image (from FFT analysis)
- **Skip connections** bypass the bottleneck and carry frequencies up to 512 cycles
- **Verdict**: Bottleneck is **adequate** ✅ — provides global context while skips provide local detail

### Receptive Field

- **Encoder RF**: 140×140 px (13.7% of image)
- **Full network RF**: 200×200 px (19.5% of image)
- **Chin region**: ~100-300 px wide, ~50-200 px tall
- **Verdict**: ⚠️ **Marginal** — covers chin but not full jaw contour (300-500 px). Adding an SE block at the bottleneck would give full-image global context.

### Architecture Comparison

| Architecture | Params | GFLOPs | Verdict |
|---|---|---|---|
| **BaseUNetHalf (current)** | 6.3M | 845 | ✅ Right size for 7K images |
| UNet (full channels) | 13.4M | ~1800 | ❌ Overfit risk |
| **Attention U-Net** | ~8M | ~950 | ✅ Best upgrade path |
| ResUNet | ~7M | ~900 | ✅ Better gradient flow |
| U²-Net | 44M | ~7000 | ❌ Overkill |
| pix2pix Generator | 54M | ~8000 | ❌ Way too big |
| NAFNet / Restormer | 17-26M | 2000-3000 | ❌ Would overfit on 7K images |

### Recommended Upgrade

Keep `BaseUNetHalf`, add three things → **Attention U-Net with global residual** (~8M params):

```
Current:     input → UNet → sigmoid → blend_map
Improved:    input → AttentionUNet(InstanceNorm) → tanh·0.5 + 0.5 → blend_map
```

---

## 6. Pose Diversity Analysis

> Full dataset analysis using MediaPipe `FaceLandmarker` with facial transformation matrices → Euler angle decomposition.  
> Script: `tools/pose_analysis.py` — run with `python tools/pose_analysis.py`

### Method

- **Tool**: MediaPipe `FaceLandmarker` (float16 model, `face_landmarker.task`)
- **Approach**: Extract 4×4 facial transformation matrix per image → decompose rotation sub-matrix into yaw/pitch/roll via `atan2`
- **Dataset**: 7,280 images (after filtering `aadan_double_chin`)
- **Detection rate**: 7,164 faces detected (**98.4%**), 116 no face (all from `prd-update`)

### Overall Statistics

| Angle | Mean | Std | Min | Max | P5 | P95 |
|---|---|---|---|---|---|---|
| **Yaw** | 0.9° | 11.5° | −67.4° | 65.3° | −14.9° | 17.9° |
| **Pitch** | 1.4° | 6.4° | −35.7° | 39.8° | −8.3° | 11.1° |
| **Roll** | 0.2° | 7.9° | −99.4° | 59.7° | −7.3° | 7.3° |

### Yaw Distribution (Head Turning)

| Bucket | Yaw Range | Count | % |
|---|---|---|---|
| Frontal | |yaw| ≤ 10° | 3,462 | **48.3%** |
| Slight turn | 10° < |yaw| ≤ 25° | 3,187 | **44.5%** |
| Moderate turn | 25° < |yaw| ≤ 45° | 433 | **6.0%** |
| Hard turn | |yaw| > 45° | 82 | **1.1%** |

### Pitch Distribution (Head Tilt Up/Down)

| Bucket | Pitch Range | Count | % |
|---|---|---|---|
| Level | |pitch| ≤ 10° | 6,427 | **89.7%** |
| Slightly up | pitch > 10° | 254 | **3.5%** |
| Slightly down | pitch < −10° | 475 | **6.6%** |
| Looking up | pitch > 20° | 26 | **0.4%** |

### Roll Distribution (Head Tilt Side-to-Side)

| Bucket | Roll Range | Count | % |
|---|---|---|---|
| Upright | |roll| ≤ 5° | 5,330 | **74.4%** |
| Slight left | roll < −5° | 618 | **8.6%** |
| Slight right | roll > 5° | 833 | **11.6%** |
| Heavy tilt | |roll| > 15° | 383 | **5.3%** |

### Per-Source Breakdown

| Metric | `pegu_double_chin` | `prd-update` |
|---|---|---|
| Images | 3,400 | 3,880 |
| Faces detected | 3,400 (100%) | 3,764 (97.0%) |
| No face | 0 | 116 |
| **Yaw mean ± std** | 1.5° ± 6.5° | 0.3° ± 14.6° |
| **Yaw range** | [−24.3°, 24.1°] | [−67.4°, 65.3°] |
| **Pitch mean ± std** | 0.1° ± 3.1° | 2.6° ± 8.2° |
| **Roll mean ± std** | 0.2° ± 0.5° | 0.1° ± 11.1° |
| Frontal (|yaw| ≤ 10°) | 59.7% | 38.0% |
| Moderate+ (|yaw| > 25°) | 0.5% | 12.0% |

**Key observation**: `pegu_double_chin` is very uniform — near-frontal poses with almost zero roll variation (std 0.5°). **All pose diversity comes from `prd-update`**.

### Coverage Gaps

| Gap | Current Coverage | Target | Shortfall |
|---|---|---|---|
| Hard-turn yaw (|yaw| > 45°) | 1.1% (82 images) | 5-10% | ~300-600 images needed |
| Looking-up (pitch > 10°) | 3.6% (254 images) | 5-10% | ~100-450 images needed |
| Profile views (|yaw| > 60°) | ~10 images | 2-5% | ~130-340 images needed |

### Verdict

| Dimension | Range | Rating |
|---|---|---|
| Yaw (total span) | 133° | ✅ Excellent |
| Pitch (total span) | 75° | ✅ Excellent |
| **Yaw distribution balance** | 92.8% within ±25° | ⚠️ Skewed frontal |
| **Profile coverage** | 1.1% hard turn | ❌ Weak |

**Overall: MODERATE diversity** — good frontal and slight-turn coverage, but weak on profiles and upward poses. Model will generalize well for typical selfie/portrait angles but may struggle with strong side profiles.

### Recommendations

1. **Augment with 500-1000 profile images** (|yaw| > 30°) for better generalization
2. **Apply yaw-aware augmentation** during training — horizontal flip already helps (mirrors yaw distribution)
3. **Consider pose-stratified sampling** to prevent the model from over-fitting to frontal poses
4. **Filter `pegu_double_chin`** is safe to use but doesn't contribute to pose robustness

---

## 7. CoreML Deployment Benchmark

> Benchmarked using `tools/benchmark_coreml.py` on Apple Silicon (arm).  
> Environment: Python 3.11.14, torch 2.4.0, coremltools 8.1, macOS 26.2  
> Conversion: `torch.export` (strict=False) → `run_decompositions({})` → CoreML mlprogram (float16)  
> Compute units: ALL (ANE + GPU + CPU)

### Results Summary

| Metric | 1024×1024 | 512×512 |
|---|---|---|
| **PyTorch (MPS, fp32) — Mean** | 168.3 ms (5.9 FPS) | 42.5 ms (23.5 FPS) |
| **CoreML (fp16, ALL) — Mean** | 186.7 ms (5.4 FPS) | 37.1 ms (27.0 FPS) |
| **CoreML speedup** | 0.90× (PyTorch faster) | **1.15× (CoreML faster)** |

### Detailed Latency Stats

#### 1024×1024 (100 iterations, 10 warmup)

| Stat | PyTorch MPS (fp32) | CoreML (fp16) |
|---|---|---|
| Mean | 168.3 ms | 186.7 ms |
| Median | 168.3 ms | 186.4 ms |
| Std | 0.2 ms | 6.2 ms |
| Min | 167.8 ms | 170.0 ms |
| Max | 168.7 ms | 212.4 ms |
| FPS | 5.9 | 5.4 |

#### 512×512 (100 iterations, 10 warmup)

| Stat | PyTorch MPS (fp32) | CoreML (fp16) |
|---|---|---|
| Mean | 42.5 ms | 37.1 ms |
| Median | 42.5 ms | 36.4 ms |
| Std | 0.2 ms | 5.5 ms |
| Min | 42.2 ms | 24.6 ms |
| Max | 43.1 ms | 52.3 ms |
| P95 | 42.8 ms | 45.4 ms |
| P99 | 43.0 ms | 49.6 ms |
| FPS | 23.5 | 27.0 |

### Numerical Validation (512×512)

| Metric | Value |
|---|---|
| Status | ✅ PASS |
| Max absolute error | 0.0005 |
| Mean absolute error | 0.0001 |
| Tolerance (atol) | 0.01 |

Float16 quantization introduces **negligible** error — well within perceptual threshold.

### Why CoreML Loses at 1024 but Wins at 512

**At 1024×1024 — PyTorch MPS is 1.1× faster:**
- **ANE SRAM overflow**: The Neural Engine has ~32 MB on-chip SRAM. At 1024² with 256 channels, activation tensors exceed the buffer → constant spilling to DRAM
- **ANE ↔ GPU fallback transfers**: The U-Net's bilinear upsampling may not run entirely on the ANE — parts fall back to GPU, incurring memory bus transfers for large tensors (4 transfers, one per decoder level)
- **MPS saturation**: The GPU's raw parallelism fully saturates at large convolutions

**At 512×512 — CoreML is 1.15× faster:**
- **Activations tile into ANE SRAM** → minimal DRAM spills
- **Float16 halves bandwidth** vs float32 — decisive when memory isn't the bottleneck
- **Lower per-layer dispatch overhead** on ANE vs GPU kernel launches

**Variance confirms multi-unit dispatch:**
PyTorch MPS std is 0.2 ms (pure GPU, rock-steady), while CoreML std is 5–6 ms (ANE scheduler non-deterministically routing ops across ANE/GPU/CPU).

### Deployment Recommendations

1. **For iOS at 1024×1024**: Use `--compute-units cpu_and_gpu` to skip ANE and avoid fallback transfers
2. **For iOS at 512×512**: Use `--compute-units all` — 27 FPS is near-realtime
3. **Consider training at 512×512**: 4.5× faster inference, negligible quality loss for the ~3% area edits in this dataset
4. **Save model**: `python tools/benchmark_coreml.py --save-mlmodel model.mlpackage` for integration

---

## 8. Training Improvements (15 Issues)

### 🔴 Priority 1 — Bugs / Correctness

#### 1. Non-Deterministic Train/Val Split
- **File**: `src/data/dataset.py` → `create_data_loaders()`
- **Issue**: Uses `random_split()` without seeding → different split every run → validation metrics are incomparable across runs
- **Fix**: Add `generator=torch.Generator().manual_seed(42)` to `random_split()`

#### 2. Overly Aggressive LR Decay
- **Config**: `lr_step_size=50, lr_gamma=0.1`
- **Issue**: LR drops by 10x every 50 epochs: 1e-4 → 1e-5 → 1e-6 → 1e-7. By epoch 150+, LR is essentially zero
- **Fix**: Use `lr_gamma=0.5` or switch to CosineAnnealingLR

#### 3. No Gradient Clipping
- **File**: `tools/train_blend_map.py`
- **Issue**: No `torch.nn.utils.clip_grad_norm_()`. VGG perceptual loss can produce large gradients
- **Fix**: Add `clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`

#### 4. Double `ToTensor()` Normalization Risk
- **File**: `src/data/dataset.py` → `BlendMapDataset.__getitem__()`
- **Issue**: Images are manually divided by 255 to [0,1], then `ToTensor()` is applied. If input is uint8, `ToTensor()` divides by 255 again → values in [0, 1/255]
- **Fix**: Ensure images are float32 before `ToTensor()`, or use `torch.from_numpy()` directly

### 🟠 Priority 2 — Quality

#### 5. No SSIM Loss or Metric
- **Issue**: Only MSE losses are used. SSIM better captures perceptual quality for image retouching
- **Fix**: Add `pytorch_msssim.SSIM` as a loss term or at least track it as a validation metric

#### 6. No Exponential Moving Average (EMA)
- **Issue**: No weight averaging → checkpoint quality fluctuates with batch variance
- **Fix**: Add EMA with decay 0.999, use EMA weights for inference

#### 7. Augmentations Are Partially Commented Out
- **File**: `src/data/dataset.py`
- **Issue**: Color jitter, Gaussian blur commented out. Only hflip + rotation active
- **Fix**: Re-enable with appropriate probabilities. Consider elastic deformation for chin variation

#### 8. `CombinedLoss` Is Not `nn.Module`
- **File**: `src/losses/losses.py`
- **Issue**: Plain class, not `nn.Module` → device management is manual and fragile
- **Fix**: Make it inherit `nn.Module`, register sub-losses as submodules

#### 9. Checkpoint Resume Is Incomplete
- **File**: `tools/train_blend_map.py`
- **Issue**: Only model weights are loaded on resume. Optimizer state, scheduler state, and best loss are lost → LR restarts from initial value
- **Fix**: Save/load `optimizer.state_dict()`, `scheduler.state_dict()`, and `best_val_loss`

### 🟡 Priority 3 — Performance / Cleanup

#### 10. Excessive Visualization Every Epoch
- **File**: `src/utils/utils_blend.py`
- **Issue**: Saves visualizations for every epoch → I/O bottleneck
- **Fix**: Visualize every N epochs (e.g., 5 or 10)

#### 11. No Mixed Precision (AMP)
- **Issue**: Training at full float32 → slower and uses more GPU memory
- **Fix**: Add `torch.cuda.amp.GradScaler` and `autocast` for ~2x speedup

#### 12. No `pin_memory` in DataLoader
- **Issue**: CPU→GPU transfer is slower without pinned memory
- **Fix**: Add `pin_memory=True` to DataLoader kwargs

#### 13. Stale Documentation & Example Code
- **Files**: `example_inference.py`, `inference.py`
- **Issue**: Reference old class name `DoubleChinInference` (renamed to `DoubleChinRemover`)
- **Fix**: Update all references

#### 14. Hardcoded `noise_threshold` in Dataset
- **File**: `src/data/dataset.py`
- **Issue**: `noise_threshold=3.0/255.0` is hardcoded, not configurable via YAML
- **Fix**: Add to config, pass through to `BlendMapDataset`

#### 15. Silent Error Retry in Dataset
- **File**: `src/data/dataset.py`
- **Issue**: On load failure, silently returns a random different sample → hides data problems
- **Fix**: Log a warning when falling back to a different sample

---

## 9. Final Recommendations

### Immediate Actions (Before Next Training Run)

1. **Fix train/val split** — add seed to `random_split()` (Issue #1)
2. **Fix LR schedule** — change `lr_gamma` from 0.1 to 0.5, or use cosine annealing (Issue #2)
3. **Add gradient clipping** — `clip_grad_norm_(model.parameters(), 1.0)` (Issue #3)
4. **Save full checkpoint** — include optimizer + scheduler state on resume (Issue #9)
5. **Enable AMP** — free ~2x speedup (Issue #11)

### Model Architecture Changes (Next Version)

1. **Add global residual** — `output = 0.5 + 0.5 * tanh(network(x))` — single biggest improvement
2. **Add attention gates** on skip connections — focus on chin region
3. **Switch BatchNorm → InstanceNorm** — better for small batches and image-to-image tasks

### Data Pipeline

1. **Filter geometric cases** — remove the ~6% of images with optical flow > 2px from training
2. **Enable commented augmentations** — color jitter, Gaussian blur
3. **Track SSIM** as a validation metric
4. **Augment pose diversity** — add 500-1000 profile images (|yaw| > 30°) or use pose-stratified sampling (see Section 6)

### Don't Change

- Architecture family (U-Net is correct)
- Channel cap at 256 (appropriate for 7K images)
- Blend map approach (validated by data analysis)
- Image resolution (1024×1024)
- Loss function structure (4-term combination is good, just tune weights)
