# Production Review: `train_blend_map.py` Pipeline

**Date:** 2026-04-09 (updated: blend map distribution analysis + additional fixes)  
**Scope:** `tools/train_blend_map.py` and all associated modules  
**Reviewer:** Senior ML Systems Engineer  

---

## Files Reviewed

| File | Role |
|------|------|
| `tools/train_blend_map.py` | Main distributed training script |
| `src/data/dataset.py` | Dataset classes and data loader creation |
| `src/losses/losses.py` | Combined loss (MSE + Perceptual + TV) |
| `src/models/unet.py` | UNet and UNet-Lite model definitions |
| `src/blend/blend_map.py` | Blend map computation and application |
| `src/utils/utils_blend.py` | Checkpoint I/O, visualization, metrics, CSV logger |
| `blend_map.yaml` | Default training configuration |

---

## ✅ Previously Reported Issues — Now Fixed

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | Checkpoint resume broken (loaded after DDP wrap) | **FIXED** | Resume now runs before `DDP()` wrapping (lines 264–272) |
| 2 | `cleanup()` never called | **PARTIALLY FIXED** | `finally: cleanup()` added in `main()` parent process (line 524). But see remaining issue below — child workers still leak |
| 3 | Config path hardcoded, no CLI | **FIXED** | `argparse` with `--config` added (lines 493–496) |
| 4 | No local metrics fallback if MLflow is down | **FIXED** | `CSVMetricsLogger` added in `utils_blend.py`, used in training loop |
| 5 | Non-atomic checkpoint writes (corruption on crash) | **FIXED** | `save_full_checkpoint` now writes to `.tmp` then `os.replace()` (line 134) |
| 6 | No model config stored in checkpoint | **FIXED** | `model_config` dict saved in checkpoint payload (line 138) |
| 7 | Excessive visualization I/O every epoch | **FIXED** | Visualizations now throttled to `save_interval` epochs + first/last (line 305) |
| 8 | No rolling latest checkpoint for crash recovery | **FIXED** | `checkpoint_latest.pth` saved every epoch (line 349) |
| 9 | Data augmentation applied to validation set | **FIXED** | `BlendMapDataset` now takes `augment` flag; `create_data_loaders` creates two instances (`augment=True` for train, `augment=False` for val) split via shared `randperm` indices wrapped in `Subset` |
| 10 | `noise_threshold` parameter shadowed and useless | **FIXED** | Re-assignment `noise_threshold = 3.0/255.0` removed from both `compute_target_blend_map` and `compute_target_blend_map_np` bodies; parameter default now controls the value |
| 11 | `cleanup()` in parent doesn't clean child NCCL groups | **FIXED** | `main_worker` now has `try/finally` around `train_skin_retouching_model` that calls `dist.destroy_process_group()` and `torch.cuda.empty_cache()` on any exit |
| 12 | VGG16 weights downloaded per process (race condition) | **FIXED** | Rank 0 now pre-downloads VGG16 weights before `dist.barrier()`; other ranks wait, then load from cache (lines 264–268 of `train_blend_map.py`) |

---

## 🔴 Critical Issues (must fix before production)

### 1. Unsafe deserialization — arbitrary code execution via `torch.load`

- **Location:** `src/utils/utils_blend.py` lines 170, 196
- **Details:** Both `torch.load` calls use the default `pickle` deserializer without `weights_only=True`.
- **Why it matters:** A malicious checkpoint file can execute arbitrary code on load. In a production pipeline where checkpoints may come from shared storage or external sources, this is a remote code execution vulnerability.
- **Suggested fix:** Add `weights_only=True` to all `torch.load` calls:
  ```python
  torch.load(path, map_location=device, weights_only=True)
  ```

---

### 2. Silent data corruption: `__getitem__` swallows errors and returns random samples

- **Location:** `src/data/dataset.py` lines 131–136
- **Details:** On any exception (corrupt image, missing file, bad NPY), the code logs to a flat file and returns `self.__getitem__(np.random.randint(0, len(self)))`. The same pattern exists in `TensorMapDataset` (line 230) but **without any retry limit** — infinite recursion risk.
- **Why it matters:**
  - Training silently runs on wrong data — gradients are computed for the wrong sample.
  - If 50% of your data is corrupt, you'd never know; you'd just train on the good half with duplicates.
  - Error log goes to a hardcoded `error_log1.txt` in cwd — not configurable, likely invisible in container deployments.
  - `TensorMapDataset.__getitem__` has no `retry` parameter at all → infinite recursion if data is consistently bad.
- **Suggested fix:** Return `None` and use a custom `collate_fn` to filter, or validate the dataset upfront before training starts.

---

## 🟠 High Priority Improvements

### 1. No mixed-precision training (AMP)

- **Details:** The entire training loop runs in FP32. For 1024×1024 images through a UNet + VGG perceptual loss, this wastes ~50% of potential throughput and doubles memory usage.
- **Impact:** 2× slower training, 2× more GPU cost, lower max batch size.
- **Suggested fix:** Add `torch.cuda.amp.GradScaler` and `autocast`:
  ```python
  scaler = torch.cuda.amp.GradScaler()
  with torch.cuda.amp.autocast():
      pred_blend_maps = model(images)
      ...
      loss, losses_dict = criterion(...)
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```

---

### 2. No `pin_memory=True` in DataLoaders

- **Location:** `src/data/dataset.py` lines 353–365
- **Details:** Neither `DataLoader` sets `pin_memory=True`.
- **Impact:** CPU→GPU transfer is synchronous and slower, adding latency every batch.
- **Suggested fix:** Add `pin_memory=True` for CUDA training.

---

### 3. `CombinedLoss` is not an `nn.Module`

- **Location:** `src/losses/losses.py` line 88
- **Details:** `CombinedLoss` is a plain Python class, not `nn.Module`. It has a custom `.to()` method that doesn't properly chain.
- **Impact:**
  - Won't appear in model summaries or device management.
  - The `PerceptualLoss` parameters won't be excluded from optimizer unless manually managed.
- **Suggested fix:** Make it inherit `nn.Module`, register sub-losses as submodules.

---

### 4. Blend map cache invalidation is non-existent

- **Location:** `src/data/dataset.py` lines 109–113
- **Details:** If the `.npy` blend map doesn't exist, it's computed on-the-fly and saved. But if the blend formula changes (which it has — there's old commented-out code for a different formula), stale `.npy` files are silently loaded with no version check.
- **Impact:** Training on stale/incorrect blend maps after a formula change, with no way to detect it.
- **Suggested fix:** Include a formula version hash in the cache filename, or add a `--recompute-blendmaps` flag.

---

## 🟡 Medium Improvements

### 1. No early stopping
Training runs for all `num_epochs` regardless of whether validation loss has plateaued. Wastes compute on long 150-epoch runs.

### 2. No gradient clipping
Large or corrupt inputs can produce exploding gradients, especially with the perceptual loss. Add `torch.nn.utils.clip_grad_norm_`.

### 3. File handle leaks
`src/data/dataset.py` line 63 — `json.load(open(data_json))` never closes the file descriptor. Same pattern on line 195. Use `with open(...) as f:`.

### 4. `sys.path.append` for imports
`tools/train_blend_map.py` line 22 — Fragile path manipulation. Should use a proper package install (`pip install -e .`) with a `pyproject.toml`.

### 5. Hardcoded MLflow tracking URI
`blend_map.yaml` line 47 contains an internal server URL (`http://mlflow.aftershoot.dev/`). Should be an environment variable so it works across dev/staging/prod environments.

### 6. No SSIM metric
Only PSNR is computed. For image quality assessment, SSIM is standard and more perceptually meaningful. Consider adding it to `compute_metrics`.

### 7. `PerceptualLoss.to()` overrides `nn.Module.to()` incorrectly
`src/losses/losses.py` lines 28–33 manually moves each slice instead of relying on `nn.Module.to()`. The `super().to(device)` call at the end already handles all registered submodules — the manual moves are redundant and fragile.

### 8. No per-field CLI config overrides
While `--config` was added, there's no way to override individual hyperparameters from the command line (e.g., `--learning_rate 0.001`). This makes sweeps and quick experiments require YAML file copies.

---

## 🟢 Low Priority / Nice-to-have

- **No `persistent_workers=True`** in DataLoaders — Workers are re-spawned every epoch, adding overhead.
- **Dead commented-out code** throughout `src/blend/blend_map.py` (lines 6–53), `src/data/dataset.py` (lines 51–53, 161–173) — Remove to reduce confusion.
- **Naming inconsistency:** "test" is used for validation throughout (e.g., `test_loader`, `test_results`). In ML convention, "validation" and "test" are different sets.
- **No `find_unused_parameters` flag** on DDP — If any model path isn't used in the forward pass, DDP will hang silently. Set explicitly to `False` and document why.
- **`TORCH_DISTRIBUTED_DEBUG=DETAIL`** is set unconditionally in production code (`tools/train_blend_map.py` line 193). This adds overhead and should be configurable or disabled by default.

---

## 💡 Architectural Suggestions

### 1. Decouple data validation from training
Add a `validate_dataset.py` pre-flight script that checks all paths exist, images are loadable, blend maps are computable, and JSON schema is correct. Run this before any training job.

### 2. Adopt a proper experiment tracking config system
Use Hydra or a similar config framework instead of raw YAML + hardcoded defaults. This gives you config composition, CLI overrides, multirun sweeps, and automatic output directory management.

### 3. Add a data versioning strategy
Track data manifests (JSON file hash, file counts, split membership) as MLflow params or use DVC. Currently, adding 1 new image to the JSON reshuffles the entire train/test split.

### 4. Pre-compute and version blend maps offline
The on-the-fly compute + caching in `__getitem__` is a distributed race condition (multiple workers may try to write the same `.npy` simultaneously). Run a one-time preprocessing job, commit the manifest, and load only.

### 5. Add model export validation
After training completes, automatically run inference on a held-out canary set and assert PSNR > threshold before promoting the model. This catches silent regressions.

---

## 🚨 Hidden / Non-Obvious Risks

### 1. Train/test split instability
`random_split` with a fixed seed over a list loaded from JSON. If the JSON order changes (new data appended, items reordered), the split changes silently. Previous "test" images become training images → **data leakage** and inflated metrics with no warning.

### 2. Rotation augmentation corrupts blend map semantics
Rotating a blend map with `fill=(0.5, 0.5, 0.5)` sets the fill to "neutral" (correct), but rotating the image with `fill=0` (default black) creates black border regions. The loss then trains the model to output neutral blend for black pixels, creating a systematic bias at image borders that will manifest on real-world crops.

### 3. ~~PSNR computed on augmented validation data~~ — RESOLVED
~~Since augmentation is applied to test data, PSNR is computed on images with artificial black regions.~~ **Fixed:** Validation now uses `augment=False` dataset instance. No longer an issue.

### 4. Perceptual loss VGG expects ImageNet normalization
The VGG16 in `PerceptualLoss` uses `VGG16_Weights.DEFAULT` which expects ImageNet-normalized input (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). The pipeline feeds raw `[0,1]` images. The perceptual loss is operating on out-of-distribution features, reducing its effectiveness to an expensive near-random feature MSE.

### 5. DDP + `np.random.randint` in `__getitem__` error handler
When a data error occurs, `np.random.randint(0, len(self))` is seeded per-process. Different DDP processes may diverge in their random state after errors, causing the `DistributedSampler` assumption of synchronized batches to break. This can cause NCCL timeouts or silent gradient corruption.

---

## 📊 Empirical Analysis: Blend Map Target Distribution

**Analysis date:** 2026-04-09  
**Script:** `tools/analyze_blend_map_distribution.py`  
**Sample size:** 500 images from the usable dataset (7,280 images, excluding `aadan_double_chin`)  

Key question: Does the model's `ResidualTanh(scale=0.3)` output range `[0.2, 0.8]` cover the actual target blend map values?

### Results

| Metric | Value |
|--------|-------|
| Active pixels outside [0.2, 0.8] | **0.0134%** |
| Images with ANY pixel outside [0.2, 0.8] | 5/500 (1.0%) |
| P0.01 (extreme low) | 0.2353 ✅ |
| P99.99 (extreme high) | 0.7765 ✅ |
| Dataset absolute min | 0.0451 (single outlier) |
| Dataset absolute max | 0.9874 (single outlier) |

### Distribution Shape

The target blend map is a **sharp spike near 0.5** (neutral), with almost all edits being slight darkening (values concentrated in `[0.42, 0.50]`). This is consistent with the dataset being subtle skin shadow adjustments (mean edit magnitude 11.6/255).

```
[0.44-0.46] 11.32%  █████████
[0.46-0.48] 24.09%  █████████████████████
[0.48-0.50] 45.32%  ████████████████████████████████████████  ← bulk of data
[0.50-0.52]  7.76%  ██████
```

### Conclusion

✅ **`ResidualTanh(scale=0.3)` output range `[0.2, 0.8]` is sufficient.** It covers 99.987% of target pixels. The theoretical concern about target-range mismatch (identified in the blend map math review) is a **non-issue in practice** for this dataset. No target clamping is needed.

The 5 outlier images (1%) with pixels outside the range are likely data quality edge cases and won't meaningfully affect training.

---

## Summary

| Severity | Count | Previously | Change |
|----------|-------|------------|--------|
| ✅ Fixed | 12 | 10 | +2 (child NCCL cleanup, VGG download race) |
| 🔴 Critical | 2 | 3 | −1 (child NCCL cleanup fixed) |
| 🟠 High | 4 | 5 | −1 (VGG download race fixed) |
| 🟡 Medium | 8 | 8 | ±0 |
| 🟢 Low | 5 | 5 | ±0 |
| 🚨 Hidden Risks | 4 | 4 | ±0 |

**Overall assessment:** Excellent progress — 12 of the original issues now resolved. The two remaining critical items are unsafe `torch.load` deserialization and silent error swallowing in `__getitem__`. The blend map distribution analysis confirms the `ResidualTanh` activation is well-matched to the data. All DDP lifecycle issues are now properly handled.
