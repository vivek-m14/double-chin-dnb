# Training Pipeline — Changelog

> All changes from initial commit (`b928a07`) to current HEAD.
> Covers infrastructure fixes, architecture additions, training improvements, and experiment setup.

---

## 1. DDP Crash Diagnostics

**Files:** `tools/train_blend_map.py`

The original training script would crash silently on multi-GPU runs — rank 0 would die without printing anything, and ranks 1–3 would timeout with cryptic `TCPStore` errors.

- **VGG16 download guard** — rank 0 downloads VGG16 weights first, then `dist.barrier()` so other ranks don't race to download simultaneously
- **`torch.cuda.synchronize()`** before DDP wrapping — flushes pending CUDA errors so they surface with a clear traceback instead of a cryptic NCCL timeout
- **Pre-DDP barrier** — `dist.barrier()` before `DDP()` to ensure all ranks are alive
- **`flush=True` + `traceback.print_exc()`** on all error paths — rank 0 crash output was being buffered and lost; now it always prints before the process dies
- **`device_id` in `init_process_group()`** — silences the NCCL "device used by this process is currently unknown" warning

---

## 2. MLflow Robustness

**Files:** `tools/train_blend_map.py`, `blend_map.yaml`

MLflow connection failures (401 auth, 405 redirect) were killing the entire training run.

- **Non-fatal MLflow** — wrapped all MLflow calls in `try/except` with an `mlflow_active` flag; training continues without MLflow if init fails
- **Bare `except Exception: pass`** on every `mlflow.log_metrics()` / `mlflow.log_artifact()` call inside the training loop
- **HTTP → HTTPS** — fixed `mlflow_tracking_uri` from `http://` to `https://mlflow.aftershoot.dev/` (server returns 301 redirect that the MLflow client doesn't follow on POST, causing 405)

---

## 3. CSV Metrics Logging

**Files:** `src/utils/utils_blend.py`, `tools/train_blend_map.py`

Provides a local fallback when MLflow is unavailable.

- **`CSVMetricsLogger`** class — append-friendly CSV logger that writes a `metrics.csv` with immediate flush per epoch; always active regardless of MLflow status

---

## 4. Checkpoint System

**Files:** `src/utils/utils_blend.py`, `tools/train_blend_map.py`

Full resumable training with architecture metadata for inference auto-detection.

- **`save_full_checkpoint()`** — saves model + optimizer + scheduler + epoch + best_val_loss + git_sha + `model_config` dict; uses atomic write (`tmp` + `os.replace`)
- **`load_full_checkpoint()`** — restores everything for resumable training, strips DDP `module.` prefix
- **`model_config` in checkpoint** — stores `model_variant`, `last_layer_activation`, `blend_scale` so inference can auto-detect architecture from any checkpoint
- **`checkpoint_best.pth` + `checkpoint_latest.pth`** — best model saved on val loss improvement; latest overwritten every epoch for crash recovery
- **Resume before DDP** — checkpoint loading happens before `DDP()` wrapping (correct order)

---

## 5. LR Scheduler Support

**Files:** `tools/train_blend_map.py`

- **`CosineAnnealingLR`** added alongside existing `StepLR`, configurable via `lr_scheduler: 'cosine'` in YAML
- Parameters: `lr_min` (eta_min for cosine), `lr_step_size` / `lr_gamma` (for step)

---

## 6. Model Architecture

**Files:** `src/models/unet.py`

Two new components for the blend-map prediction task.

- **`BaseUNetHalfLite`** — lighter variant that bilinear-downscales input to half resolution, uses 3 encoder levels instead of 4, ~50% fewer parameters (~1.2M vs ~6.3M). Well-suited for blend maps which are inherently smooth.
- **`ResidualTanh`** activation — `0.5 + scale * tanh(x)`, output centered at 0.5 (blend-map neutral) with configurable range. With `scale=0.3`, output range is [0.2, 0.8] which covers 99.99% of target blend map values.

---

## 7. Data Pipeline

**Files:** `src/data/dataset.py`

- **Separate train/val dataset instances** — `BlendMapDataset(..., augment=True)` for train, `BlendMapDataset(..., augment=False)` for val, with seeded index split shared across both
- **Blend map cache** — `.npy` files computed on first access via `compute_target_blend_map_np()`, reused on subsequent epochs

---

## 8. P0 Fixes — Loss & Optimization

**Files:** `src/losses/losses.py`, `tools/train_blend_map.py`

Two critical correctness fixes identified during adversarial code review.

- **VGG ImageNet normalization** — `PerceptualLoss.forward()` now normalizes inputs with `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` before extracting VGG16 features. Previously raw [0, 1] images were fed directly, making deeper VGG layers (slice3, slice4) produce meaningless features and corrupting the perceptual gradient signal.
- **Gradient clipping** — `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` added between `loss.backward()` and `optimizer.step()`. Prevents gradient spikes from perceptual loss outliers from destabilizing Adam's momentum estimates.

---

## 9. Training Data Montage

**Files:** `src/utils/utils_blend.py`, `tools/train_blend_map.py`

YOLO-style training data visualization for sanity checking before spending GPU hours.

- **`save_training_data_montage()`** — generates 5 montage JPGs, each with 8 rows × 3 columns (Input | Blend Map | Ground Truth), 256px thumbnails, column headers burned in, dark grey background
- **Runs on rank 0 only** before training starts, with `dist.barrier()` after so other ranks wait
- **`num_workers=0` temp DataLoader** — creates a separate single-process DataLoader from the same dataset to avoid forked worker process crashes when breaking out of the iterator early

---

## 10. Experiment Configs

**Files:** `experiments/exp1_original_default.yaml`, `experiments/exp2_lite_residual_tanh.yaml`, `experiments/exp3_lite_sigmoid.yaml`

Three experiment configurations under unified `project_name: 'vk-dbc-blendmap-exp'` for MLflow comparison.

| Experiment | Model | Activation | Scheduler | Batch Size | Port |
|------------|-------|------------|-----------|------------|------|
| **exp1** | BaseUNetHalf (~6.3M params) | sigmoid [0, 1] | StepLR (step=50, γ=0.1) | 12 | 12350 |
| **exp2** | BaseUNetHalfLite (~1.2M params) | residual_tanh [0.2, 0.8] | CosineAnnealingLR | 8 | 12350 |
| **exp3** | BaseUNetHalfLite (~1.2M params) | sigmoid [0, 1] | CosineAnnealingLR | 24 | 12351 |

All share: seed=42, lr=1e-4, 150 epochs, img_size=1024, same loss weights (λ_blend=1, λ_image=1, λ_perc=0.1, λ_tv=0.1).

**Concurrent run setup:**
- exp1 on `CUDA_VISIBLE_DEVICES=0,1` (port 12350)
- exp3 on `CUDA_VISIBLE_DEVICES=2,3` (port 12351)

---

## 11. Documentation

**Files:** `docs/adversarial_audit.md`

Comprehensive adversarial audit of the full pipeline:
- **3 critical issues**: "do-nothing" baseline is deceptively good, VGG receives un-normalized input (fixed), val split leaks identity across train/val
- **5 high-risk issues**: no gradient clipping (fixed), JPEG artifacts corrupt ground truth, blend map cache has no invalidation, CombinedLoss is not nn.Module, output range clips strongest edits
- **7 subtle risks**: inconsistent retouching styles across data sources, rotation augmentation wastes capacity, torch.clamp dead gradients, distributed val visualization bias, fragile seeded split, no LR warmup, uint8 quantization in blend map computation
- Includes stress test proposals, "cheating" detection methods, and architectural recommendations

---

## Remaining Known Issues (not yet fixed)

| Priority | Issue | Status |
|----------|-------|--------|
| P1 | `CombinedLoss` is not `nn.Module` | Deferred |
| P1 | `torch.load` missing `weights_only=True` | Deferred |
| P1 | Val split leaks identity (subject-level split needed) | Deferred |
| P2 | No "do-nothing" baseline logging | Deferred |
| P2 | No LR warmup with cosine schedule | Deferred |
| P3 | Blend map cache has no invalidation key | Deferred |
| P3 | AMP (automatic mixed precision) not implemented | Deferred |
