# Production Review: `train_blend_map.py` Pipeline

**Date:** 2026-04-09 (updated after `git pull`)  
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

---

## 🔴 Critical Issues (must fix before production)

### 1. Data augmentation is applied to the validation/test set

- **Location:** `src/data/dataset.py` line 130
- **Details:** `apply_augmentation` (random flip, random rotation) is called unconditionally inside `__getitem__`. Since `create_data_loaders` uses `random_split` on a *single* `BlendMapDataset` instance, the test subset shares the same augmented `__getitem__`.
- **Why it matters:** Every validation metric (PSNR, loss) is computed on randomly-augmented images. Metrics are noisy, non-reproducible, and overestimate real-world performance. You cannot trust best-model selection.
- **Suggested fix:** Add a `training` flag to the dataset or use separate dataset instances for train/test:
  ```python
  class BlendMapDataset(Dataset):
      def __init__(self, ..., augment=True):
          self.augment = augment
      def __getitem__(self, idx, ...):
          ...
          if self.augment:
              image, blend_map, gt = self.apply_augmentation(image, blend_map, gt)
  ```

---

### 2. Unsafe deserialization — arbitrary code execution via `torch.load`

- **Location:** `src/utils/utils_blend.py` lines 149, 175, 196
- **Details:** All three `torch.load` calls use the default `pickle` deserializer without `weights_only=True`.
- **Why it matters:** A malicious checkpoint file can execute arbitrary code on load. In a production pipeline where checkpoints may come from shared storage or external sources, this is a remote code execution vulnerability.
- **Suggested fix:** Add `weights_only=True` to all `torch.load` calls:
  ```python
  torch.load(path, map_location=device, weights_only=True)
  ```

---

### 3. `cleanup()` in parent process doesn't clean up child NCCL groups

- **Location:** `tools/train_blend_map.py` lines 475–478 (cleanup def), 524 (finally block)
- **Details:** `cleanup()` is now called in `main()`'s `finally` block, but `main()` is the **parent** process which never calls `dist.init_process_group()`. So `dist.is_initialized()` is always `False` there — it only calls `torch.cuda.empty_cache()`. The actual child worker processes spawned by `mp.spawn` in `main_worker` still have no `try/finally` around `train_skin_retouching_model`, so NCCL process groups are leaked on any child crash.
- **Why it matters:** On shared GPU machines, leaked NCCL groups leave zombie processes holding GPU memory. Subsequent runs fail with `NCCL error` or port-already-in-use.
- **Suggested fix:** Add `try/finally` **inside `main_worker`**, where `dist.init_process_group` is actually called:
  ```python
  def main_worker(local_rank, world_size, args):
      ...
      dist.init_process_group(...)
      try:
          train_skin_retouching_model(local_rank, world_size, args)
      finally:
          if dist.is_initialized():
              dist.destroy_process_group()
          torch.cuda.empty_cache()
  ```

---

### 4. Silent data corruption: `__getitem__` swallows errors and returns random samples

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

### 3. `noise_threshold` parameter is shadowed and useless

- **Location:** `src/blend/blend_map.py` lines 55–62
- **Details:** Both `compute_target_blend_map` and `compute_target_blend_map_np` accept `noise_threshold` as a parameter, then immediately re-assign it to `noise_threshold = 3.0/255.0` inside the function body. The parameter is completely inert.
- **Impact:** Any caller trying to tune the noise threshold will get silent no-op behavior.
- **Suggested fix:** Remove the re-assignment inside the function body.

---

### 4. `CombinedLoss` is not an `nn.Module`

- **Location:** `src/losses/losses.py` line 88
- **Details:** `CombinedLoss` is a plain Python class, not `nn.Module`. It has a custom `.to()` method that doesn't properly chain.
- **Impact:**
  - Won't appear in model summaries or device management.
  - VGG16 inside `PerceptualLoss` gets downloaded independently per DDP process (race condition on shared FS).
  - The `PerceptualLoss` parameters won't be excluded from optimizer unless manually managed.
- **Suggested fix:** Make it inherit `nn.Module`, register sub-losses as submodules.

---

### 5. VGG16 weights downloaded per process in distributed training

- **Location:** `src/losses/losses.py` line 15
- **Details:** `vgg16(weights=VGG16_Weights.DEFAULT)` is called inside `PerceptualLoss.__init__`, which is called on every DDP process. If the model cache is cold, all processes download simultaneously.
- **Impact:** Race condition on download, potential crash or corruption on shared filesystems.
- **Suggested fix:** Download/cache weights on rank 0 first, then `dist.barrier()`, then initialize on all ranks.

---

### 6. Blend map cache invalidation is non-existent

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

### 3. PSNR computed on augmented validation data is meaningless
Since augmentation is applied to test data (Critical Issue #1), and rotation introduces black borders, PSNR is computed on images with artificial black regions. The reported PSNR is systematically higher than real performance (black matches black), giving false confidence.

### 4. Perceptual loss VGG expects ImageNet normalization
The VGG16 in `PerceptualLoss` uses `VGG16_Weights.DEFAULT` which expects ImageNet-normalized input (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). The pipeline feeds raw `[0,1]` images. The perceptual loss is operating on out-of-distribution features, reducing its effectiveness to an expensive near-random feature MSE.

### 5. DDP + `np.random.randint` in `__getitem__` error handler
When a data error occurs, `np.random.randint(0, len(self))` is seeded per-process. Different DDP processes may diverge in their random state after errors, causing the `DistributedSampler` assumption of synchronized batches to break. This can cause NCCL timeouts or silent gradient corruption.

---

## Summary

| Severity | Count | Previously | Change |
|----------|-------|------------|--------|
| ✅ Fixed | 8 | — | +8 resolved |
| 🔴 Critical | 4 | 5 | −1 (resume fixed) |
| 🟠 High | 6 | 7 | −1 (CLI config fixed) |
| 🟡 Medium | 8 | 8 | ±0 |
| 🟢 Low | 5 | 5 | ±0 |
| 🚨 Hidden Risks | 5 | 5 | ±0 |

**Overall assessment:** Good progress since last review — checkpoint resume, atomic saves, CSV logging, and CLI config are solid improvements. The remaining critical issues (augmented validation, unsafe deserialization, child process cleanup, silent data corruption) still need to be addressed before metrics and trained models can be fully trusted in production.
