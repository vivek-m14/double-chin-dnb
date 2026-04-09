# Adversarial System Audit — Double-Chin Blend Map Pipeline

> **Scope**: Full pipeline audit from first principles. Every assumption challenged.
> **Config under test**: `experiments/exp2_lite_residual_tanh.yaml`
> **Audit date**: 2025-01-XX
> **Posture**: Adversarial — trying to break the system intellectually.

---

## Executive Summary

The pipeline is architecturally sound for a v1 product, but has **3 critical** issues that could silently produce a model that appears to train successfully while learning almost nothing useful, **5 high-risk** issues that will bite in production or at scale, and **7 subtle risks** that compound over time. The most dangerous single finding is **the "do-nothing" baseline problem** — the task's inherent subtlety means a model predicting B≈0.5 everywhere already achieves deceptively good metrics.

---

## 🔴 Critical Flaws

### C1. The "Do-Nothing" Baseline Is Dangerously Good

**Failure mechanism**: The mean edit magnitude across the dataset is 11.6/255 ≈ 0.045. If the model predicts B=0.5 (neutral) everywhere, the retouched image equals the original. The image MSE loss for "do nothing" is:

$$\text{MSE}_{\text{do-nothing}} = \mathbb{E}\left[\|I_{\text{original}} - I_{\text{GT}}\|^2\right] \approx (0.045)^2 \approx 0.002$$

The blend map MSE for "do nothing" (predicting 0.5 vs the target):

$$\text{MSE}_{\text{bmap}} = \mathbb{E}\left[\|0.5 - B_{\text{target}}\|^2\right]$$

Since 99.987% of target pixels are within [0.2, 0.8] and the sharp mode at 0.5 dominates, this MSE is also very small. The corresponding "do-nothing" PSNR is ~35–40 dB — already a number that *looks good on paper*.

**Real-world impact**: You cannot tell from loss curves alone whether the model is learning meaningful chin retouching or converging to identity. Training loss will decrease (the model can trivially minimize TV loss by predicting exactly 0.5), giving the illusion of learning. PSNR will be high from epoch 1.

**Concrete fix**:
1. **Compute and log the do-nothing baseline** at epoch 0 — pass B=0.5 through the loss function and log this as `baseline_loss` and `baseline_psnr`. Every subsequent epoch's metrics should be reported as **delta from baseline**.
2. **Add a masked evaluation metric**: compute PSNR/SSIM only on the *edited region* (where `|B_target - 0.5| > noise_threshold`). This is the only metric that tells you whether the model is learning the actual retouching.
3. **Add SSIM** — it captures structural changes that MSE/PSNR miss at this subtle signal level.

---

### C2. VGG Perceptual Loss Receives Un-Normalized Input

**Failure mechanism**: VGG16 was trained on ImageNet with input normalization:
```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The pipeline feeds raw [0, 1] images directly to VGG (`losses.py` line 43-50). This shifts the entire operating point of VGG's internal feature extractors. The first conv layer expects inputs centered near 0 with std ~0.23; it receives inputs centered near 0.5 with std ~0.29. All subsequent layer activations are in a distributional regime VGG was never trained for.

**What actually happens**: Early layers (slice1, slice2) still extract *some* useful edge/texture features because low-level filters are somewhat input-range agnostic. But deeper layers (slice3, slice4) produce essentially meaningless features. The loss function is paying `λ_perc=0.1` weight for a signal that is ~50% useful (early layers) and ~50% noise (deep layers).

**Why this is critical, not just "medium"**: The perceptual loss gradient is the *only* gradient that provides high-level structural supervision (edges of chin, skin texture coherence). Without correct normalization, the model receives corrupted structural guidance. It may converge to something that looks good by MSE but has perceptual artifacts.

**Concrete fix** in `PerceptualLoss.forward()`:
```python
def forward(self, x, y):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    y = (y - mean) / std
    # ... rest of forward
```

---

### C3. Validation Split Leaks Identity

**Failure mechanism**: The train/val split is done by random index permutation:
```python
perm = torch.randperm(n, generator=split_gen).tolist()
train_indices = perm[:train_size]
val_indices   = perm[train_size:]
```

This splits at the *image level*, not the *subject level*. The JSON manifest contains paths like `double_chin_data_v3/original/{subject_id}/filename.jpg`. If a subject has multiple images (which is common in portrait datasets — different poses, lighting), some images of the same person will land in train and others in val.

**Real-world impact**: The model memorizes person-specific blend maps. Val metrics are artificially high because the model has seen (slightly different images of) the same face during training. When deployed on truly unseen subjects, performance degrades.

**How to verify**: Parse the `img_path` to extract subject IDs (the directory name), count unique subjects in train vs val, and check for overlap.

**Concrete fix**: Implement group-aware splitting — group images by subject directory, then split groups:
```python
from collections import defaultdict
subject_groups = defaultdict(list)
for i, item in enumerate(data_map):
    subject = item['original_image'].split('/')[-2]
    subject_groups[subject].append(i)
# Shuffle groups, then assign entire groups to train or val
```

---

## 🟠 High-Risk Issues

### H1. No Gradient Clipping — Perceptual Loss Can Explode

The VGG perceptual loss computes MSE across 4 feature layers and sums them without normalization. If one batch contains an outlier image (corrupted JPEG, extreme exposure), VGG features can produce very large activations, causing gradient spikes that destabilize Adam's momentum estimates.

The pipeline has **no gradient clipping** (`torch.nn.utils.clip_grad_norm_` is never called).

**Impact**: Occasional training instability, especially in early epochs before the model has learned to predict near-neutral blend maps. A single bad gradient step can shift Adam's running variance estimate, causing learning rate to be implicitly too high for subsequent steps.

**Fix**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`.

---

### H2. JPEG Compression Artifacts Corrupt Ground Truth

Original and retouched images are stored as JPEG (`.jpg`). JPEG compression introduces:
- Block artifacts at 8×8 boundaries
- Chroma subsampling artifacts
- Quality-level-dependent quantization noise

If the original and retouched images were saved at different JPEG quality levels (e.g., original at q=95, retouched at q=90 after re-encoding), the per-pixel difference includes a **systematic JPEG noise component** that has nothing to do with retouching.

The noise_threshold of 3/255 ≈ 0.012 catches small random noise, but **systematic JPEG re-encoding differences can be 5–15/255** in high-frequency regions (hair, fabric textures). These create spurious non-neutral blend map values in areas outside the chin.

**Impact**: The model learns to "retouch" JPEG artifacts — predicting non-neutral blend maps in textured regions where no retouching was applied. At inference time on lossless/differently-compressed inputs, this creates visible artifacts.

**How to verify**: Compute the blend map for a pair, then visualize it as a heatmap. If non-neutral regions appear in hair, clothing, or background (not just the chin/jawline), JPEG artifacts are leaking through.

**Fix**: Either (a) use PNG for training data, (b) increase noise_threshold to 8/255 for JPEG data, or (c) add a spatial mask that restricts supervision to the face/chin region (using the `diff_mask` or `bbox` fields already in the JSON but currently ignored).

---

### H3. Blend Map Cache Invalidation Is Silent

The cached `.npy` blend maps are computed from images resized to `self.resize_dim` (1024×1024 in the current config). The cache file name is derived from the GT image path — it does not encode the resize dimension, noise_threshold, or blend formula version.

If you:
1. Change `img_size` from 1024 to 512 → stale cache is loaded and resized, producing different values than a fresh computation at 512
2. Change `noise_threshold` → stale cache uses the old threshold
3. Switch from Linear Light to a different blend formula → stale cache uses the old formula

**Impact**: Silently training on incorrect supervision. The model converges on a blend map that was computed under different assumptions.

**Fix**: Include a hash of the computation parameters in the cache filename:
```python
cache_key = f"{noise_threshold}_{resize_dim[0]}x{resize_dim[1]}"
bmap_path = os.path.splitext(bmap_path)[0] + f'_{cache_key}.npy'
```

---

### H4. `CombinedLoss` Is Not `nn.Module` — Fragile Device/State Handling

`CombinedLoss` is a plain Python class, not `nn.Module`. Consequences:
- Its internal modules (`PerceptualLoss`, `TotalVariationLoss`, `MSELoss`) are not tracked by PyTorch's module system
- `model.state_dict()` won't include loss parameters (VGG weights are frozen so this is tolerable, but fragile)
- The manual `.to(device)` override is incomplete — if the class gains learnable parameters (e.g., learnable loss weights), they won't be found by any optimizer
- `DataParallel` or FSDP integration would fail silently

**Fix**: Inherit from `nn.Module`, register sub-losses as attributes, remove the manual `.to()` override.

---

### H5. Model Output Range [0.2, 0.8] vs Ground Truth Range [0.0, 1.0]

`ResidualTanh(scale=0.3)` outputs blend maps in [0.2, 0.8]. The empirical analysis showed 99.987% of target pixels fall within this range. But the 0.013% that don't are **systematically the most important pixels** — they represent the strongest edits, the pixels where retouching is most visible.

The model architecturally *cannot represent* these extreme edits. MSE on the blend map will have an irreducible floor from these clipped pixels. More importantly, the image-space loss will try to push the model toward extreme values it cannot reach, creating a persistent gradient toward the saturation boundary of tanh.

**Impact**: The strongest retouching edits are systematically under-predicted. The model produces "timid" results on subjects who need the most correction.

**Mitigation (already partially addressed)**: The 0.013% error is small in aggregate, but qualitatively important. Consider tracking a "clipping fraction" metric — what percentage of predicted blend map values are saturated at 0.2 or 0.8. If this is consistently >1%, increase `blend_scale` to 0.35.

---

## 🟡 Subtle / Non-Obvious Risks

### S1. Two Data Sources May Have Inconsistent Retouching Styles

The dataset has two sources: `prd-update` (3,880 images) and `pegu_double_chin` (3,400 images). If these were retouched by different artists, tools, or guidelines, the "correct" blend map for the same face could differ between sources. MSE loss averages over this inconsistency, teaching the model to predict the *mean* of two distributions — which may satisfy neither.

**How to detect**: Compute mean blend map magnitude per source. If they differ significantly, the model is learning a compromise. Consider adding source-conditional normalization or training separate heads.

---

### S2. Rotation Augmentation Wastes Model Capacity

With `expand=False`, a 45° rotation of a 1024×1024 image creates ~30% black corner pixels. The blend map fill of 0.5 is mathematically consistent (since `R = 0 + 2*0.5 - 1 = 0 = GT` for black corners). But:

- The model is supervised on these artificial black-corner patterns in ~10% of training samples (20% chance of 45° static angle × 50% rotation probability)
- It must allocate parameters to handle an input distribution (partially black frames) that never appears in production
- At aggressive rotation (75° = 45° static + 30° range), most of the face is cropped, and the model is learning from a mostly-black image

**Fix**: Limit total rotation to ±15° for subtle retouching tasks, or use `expand=True` with center-crop back to 1024.

---

### S3. `torch.clamp` in `apply_blend_formula` Creates Dead Gradient Zones

```python
def apply_blend_formula(image, blend_map):
    return torch.clamp(image + 2 * blend_map - 1, 0.0, 1.0)
```

For any pixel where `image + 2*blend_map - 1 > 1.0` or `< 0.0`, the gradient through `clamp` is exactly zero. This means `image_mse_loss` and `perc_loss` provide no gradient for those pixels, even if the prediction is wrong.

With ResidualTanh [0.2, 0.8]:
- Max retouched = image + 2×0.8 − 1 = image + 0.6 → clamped when image > 0.4
- Min retouched = image + 2×0.2 − 1 = image − 0.6 → clamped when image < 0.6

For bright skin tones (image ≈ 0.7–0.9), any brightening blend (B > 0.5) will be clamped. For dark skin tones (image ≈ 0.1–0.3), any darkening blend (B < 0.5) will be clamped.

The `blend_map_loss` term still provides gradient (it doesn't go through `clamp`), which is the safety net. But the model receives conflicting signals: `blend_map_loss` says "move toward target B", while `image_mse_loss` and `perc_loss` say nothing (zero gradient) for clamped pixels.

**Impact**: Subtle bias toward predicting moderate blend values for extreme skin tones. May cause the model to under-retouch on very light or very dark skin — a **bias/fairness concern**.

---

### S4. `DistributedSampler` + Validation Creates Redundant Computation

```python
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
```

Each GPU computes validation on a *different shard* of the val set, then losses are `dist.reduce`'d. But `save_visualization_batch` only runs on rank 0, which only sees rank 0's shard. The visualizations are not representative of the full val set — they always show the same subset of samples.

More importantly, `DistributedSampler` pads the dataset to make shards equal-sized by repeating samples. For 728 val images and 2 GPUs, each gets 364 samples. With batch_size=8, that's 45.5 batches → padded to 46 batches. The last batch on each GPU repeats some samples. This slightly inflates val metrics (repeated samples are counted twice in the loss average).

---

### S5. Seeded Split Is Fragile to Data Ordering

```python
perm = torch.randperm(n, generator=split_gen).tolist()
```

The permutation is deterministic for a given `n` and seed. But `n` depends on which entries survive the `source != "aadan_double_chin"` filter. If the JSON file is re-exported with entries in a different order, or if a few entries are added/removed, the permutation changes *entirely* — the same images may swap between train and val. All previous checkpoints become incomparable.

**Fix**: Hash the image path to assign train/val, rather than using positional indexing:
```python
import hashlib
def is_val(path, val_fraction=0.1):
    h = int(hashlib.md5(path.encode()).hexdigest(), 16)
    return (h % 1000) < int(val_fraction * 1000)
```

---

### S6. No Learning Rate Warmup

Training starts at `lr=1e-4` from epoch 0 with freshly Kaiming-initialized weights. The first few gradient steps use the full learning rate on a model that produces essentially random blend maps. This can push Adam's running statistics (`v_t`, second moment) to values calibrated for the initial random-output regime, requiring many epochs to "forget" the initial noise.

With cosine annealing (`T_max=150`), the LR starts at 1e-4, stays high for ~30 epochs, then decays. The first 5–10 epochs of noisy gradients at full LR are wasted.

**Fix**: Add 5 epochs of linear warmup from `lr=1e-6` to `lr=1e-4`.

---

### S7. `compute_target_blend_map_np` Uses Resized uint8 Images

In `__getitem__`:
```python
image = cv2.imread(img_path)      # uint8
image = cv2.resize(image, (1024, 1024))  # uint8, bilinear interpolation
gt = cv2.imread(gt_path)           # uint8
gt = cv2.resize(gt, (1024, 1024))  # uint8, bilinear interpolation
# ...
blend_map = compute_target_blend_map_np(image, gt)  # receives uint8
```

Inside `compute_target_blend_map_np`, the images are divided by 255.0. But the resize was done on uint8, which quantizes interpolated values to integers. The blend map computation sees `I_resized / 255` and `G_resized / 255`, which have 8-bit quantization artifacts from the resize.

A more accurate pipeline would resize in float32 space:
```python
image = image.astype(np.float32) / 255.0
image = cv2.resize(image, (1024, 1024))  # float32 resize, no quantization
```

The error is ~0.5/255 per pixel, which is below `noise_threshold`, so it's masked for most pixels. But for pixels near the threshold boundary, it can flip them between "noise" (→0.5) and "signal", creating inconsistent supervision.

---

## 🧪 Suggested Stress Tests

### T1. "Do-Nothing" Baseline Test
Train for 0 epochs. Set `pred_blend_maps = torch.full_like(target, 0.5)`. Compute all losses and PSNR. Log as the baseline. **If your trained model's val metrics are within 10% of this baseline after 150 epochs, the model has not learned meaningful retouching.**

### T2. Memorization Test
Train on 100 images for 500 epochs. If val loss diverges while train loss approaches 0, the model has sufficient capacity to memorize — good. If val loss tracks train loss closely even with 100 images, something is wrong (likely predicting near-constant output).

### T3. Per-Source Evaluation
Split evaluation by `source` field. Report metrics separately for `prd-update` and `pegu_double_chin`. If one source has significantly worse metrics, the retouching styles are incompatible and the model is averaging.

### T4. Adversarial Input Test
Feed the model:
- A blank white image (1.0 everywhere) → should predict ~0.5
- A blank black image (0.0 everywhere) → should predict ~0.5
- An image with no face (landscape) → should predict ~0.5
- A horizontally flipped face → blend map should be horizontally flipped

If the model predicts significantly non-neutral values for non-face inputs, it has learned a spatial prior rather than face understanding.

### T5. Gradient Flow Diagnostic
After 1 epoch, log `grad.norm()` for:
- `blend_map_loss` gradient at the model output
- `image_mse_loss` gradient at the model output
- `perc_loss` gradient at the model output
- `tv_loss` gradient at the model output

If `perc_loss` gradient is 10×–100× larger than the others, it dominates training despite `λ=0.1`. If it's 100× smaller, it contributes nothing.

### T6. Clipping Fraction Monitor
Per epoch, log: `(pred_blend_maps == 0.2).float().mean()` and `(pred_blend_maps == 0.8).float().mean()`. If these are >0.1%, the model is hitting the activation ceiling and needs a larger `blend_scale`.

### T7. Cache Consistency Test
Delete all `.npy` caches. Run one epoch. Compare the computed blend maps against the previously cached versions. If max absolute difference > 0.01, there's a cache bug.

---

## 🧠 How the Model Might Be "Cheating"

### Cheat 1: Constant Output ≈ 0.5

The strongest possible shortcut. With `ResidualTanh`, the model can trivially output 0.5 by having zero pre-activation (since `0.5 + 0.3*tanh(0) = 0.5`). The Kaiming initialization already starts weights near zero, so the initial output IS near 0.5. If the loss landscape has a local minimum near "do nothing" (which it does, because the task signal is so subtle), the model may never escape.

**Detection**: Plot the histogram of predicted blend map values per epoch. If it remains a sharp spike at 0.5 that barely broadens, the model isn't learning.

### Cheat 2: Learned Spatial Prior

If all training faces are centered and roughly aligned (which they are — they're cropped portrait photos), the model can learn:
- A fixed 2D mask that darkens the chin area (bottom 30-40% of image)
- No per-image adaptation

This would generalize to centered portraits but fail on rotated, off-center, or profile faces.

**Detection**: Run inference on the same face at different positions within the frame. If the blend map doesn't follow the face, it's a spatial prior.

### Cheat 3: Luminance Shortcut

The Linear Light formula is `R = I + 2B - 1`. The MSE loss on the retouched image is:
$$\mathcal{L} = \|I + 2B_{pred} - 1 - G\|^2 = \|2B_{pred} - (G - I + 1)\|^2 = 4\|B_{pred} - B_{target}\|^2$$

Wait — this means `image_mse_loss` is literally 4× the `blend_map_loss` (in non-clamped regions)! They provide **redundant supervision**. The model is effectively being trained with `(λ_blend + 4*λ_image) * MSE(B_pred, B_target)` plus noise from clamped regions and perceptual loss.

This is not "cheating" per se, but it means `lambda_image_mse` is not an independent control — it's mechanically coupled to `lambda_blend_mse`. Setting both to 1.0 effectively gives blend map MSE a weight of 5.0.

### Cheat 4: Color-Channel Correlation

Skin retouching edits are typically correlated across R, G, B channels (skin darkening affects all three similarly). A lazy model can predict the same value for all 3 channels of the blend map, effectively predicting a 1-channel map broadcast to 3. This reduces the effective task to 1 channel, making it easier.

**Detection**: Compute `std(pred_blend_map, dim=1)` across channels per pixel. If this is near-zero, the model is treating it as a 1-channel problem. Consider switching to a 1-channel blend map with broadcast.

---

## 🏗️ Architectural Recommendations

### R1. Add a "Do-Nothing" Baseline Logger (Priority: P0)

Before ANY further training, implement:
```python
with torch.no_grad():
    neutral = torch.full_like(target_blend_maps, 0.5)
    neutral_retouched = apply_blend_formula(images, neutral)
    _, baseline_losses = criterion(neutral, target_blend_maps, neutral_retouched, gt_images)
    baseline_psnr = compute_metrics(neutral_retouched, gt_images)['psnr']
```
Log `baseline_psnr` and `baseline_total_loss` as horizontal lines on your training charts. Your model must **clearly beat these**.

### R2. Add ImageNet Normalization to VGG (Priority: P0)

Three-line fix in `PerceptualLoss.forward()`. No retraining needed — just corrects the input distribution. May significantly improve perceptual quality of results.

### R3. Implement Masked Metrics (Priority: P1)

Add a metric that evaluates only on the edited region:
```python
edit_mask = (target_blend_maps - 0.5).abs() > noise_threshold
masked_psnr = -10 * log10(MSE(retouched[edit_mask], gt[edit_mask]))
```
This is the only metric that tells you whether retouching quality is improving.

### R4. Subject-Aware Validation Split (Priority: P1)

Group by subject directory, split groups. This gives honest generalization metrics.

### R5. Gradient Clipping + LR Warmup (Priority: P2)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Plus 5-epoch linear warmup. Standard practice for stable training.

### R6. Consider 1-Channel Blend Map (Priority: P2)

If empirical analysis confirms channels are correlated (Cheat 4), switch to `n_classes=1` and broadcast. Reduces model parameters by ~3× in the output head, provides a stronger inductive bias.

### R7. AMP (Automatic Mixed Precision) (Priority: P3)

Free 1.5–2× speedup and ~40% VRAM reduction. Critical for scaling to larger batch sizes:
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    pred = model(images)
    loss, _ = criterion(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📊 Risk Matrix Summary

| ID | Severity | Category | Description | Effort |
|----|----------|----------|-------------|--------|
| C1 | 🔴 Critical | Evaluation | "Do-nothing" baseline is deceptively good | 1h |
| C2 | 🔴 Critical | Loss | VGG receives un-normalized input | 15min |
| C3 | 🔴 Critical | Data | Val split leaks identity across train/val | 2h |
| H1 | 🟠 High | Training | No gradient clipping | 5min |
| H2 | 🟠 High | Data | JPEG artifacts corrupt ground truth | 4h |
| H3 | 🟠 High | Data | Blend map cache has no invalidation | 1h |
| H4 | 🟠 High | Architecture | CombinedLoss is not nn.Module | 30min |
| H5 | 🟠 High | Architecture | Output range clips strongest edits | Monitor |
| S1 | 🟡 Subtle | Data | Two sources may have incompatible styles | 2h |
| S2 | 🟡 Subtle | Augmentation | Rotation wastes capacity on black corners | 30min |
| S3 | 🟡 Subtle | Loss | torch.clamp creates dead gradients per skin tone | Analysis |
| S4 | 🟡 Subtle | Training | Val visualizations only from rank 0's shard | 30min |
| S5 | 🟡 Subtle | Data | Seeded split fragile to data ordering | 1h |
| S6 | 🟡 Subtle | Training | No LR warmup with cosine schedule | 15min |
| S7 | 🟡 Subtle | Data | Blend map computed from quantized uint8 resize | 30min |

---

## 🔬 The Deepest Question

*Is this task fundamentally well-posed?*

The model must predict, from an original image alone, a blend map that reproduces a human retoucher's artistic decisions. But:

1. **The retouching is not deterministic** — different retouchers would produce different blend maps for the same face. MSE loss averages over this ambiguity.
2. **The signal is extremely subtle** — mean edit magnitude is 4.5% of the dynamic range. The model must detect and reproduce edits that are barely visible to the human eye.
3. **The dataset is small** — 6,552 training images for a 1024×1024 → 1024×1024 mapping is thin. The model has ~1.2M parameters but the effective task dimensionality (predicting 3×1024×1024 outputs) is enormous.
4. **The inductive bias is appropriate** — UNet with bilinear down/up and ResidualTanh centered at neutral is a good prior for "mostly do nothing, with subtle local edits." The architecture is well-matched to the task.

**Verdict**: The task is well-posed *enough* for a product — the model can learn a useful prior over "where and how much to darken the chin." But the evaluation framework (C1, C3, R3) must be fixed first, or you will not be able to tell whether training is succeeding or failing.

---

*This audit was performed by tracing every line of code, every gradient flow, every data transformation, and every numerical operation in the pipeline. Each finding includes a concrete mechanism, not just a label.*
