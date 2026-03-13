# Double Chin Removal

## Environment & data

- **RunPod network volume:** This repo is set up on RunPod’s network volume.
  - **Volume name:** `vineet-vol-2`
  - **Volume ID:** `529g08aem0`
  - **Path:** `/workspace/double_chin`
- **Dataset:** Training/inference images live in Google Cloud Storage:
  - [Double Chin images (GCS)](https://console.cloud.google.com/storage/browser/retouching-ai/Double_Chin/double_chin_images?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&invt=AbtTXA&project=aftershoot-co) — `retouching-ai/Double_Chin/double_chin_images`

Download or mount this data and point `data_root` / your dataset paths to it when training. Refer to this section if anything about paths or data location is unclear.

---

Deep learning model for removing double chins from portrait images. The model predicts **blend maps** (3-channel RGB) that are applied to the input via a Linear Light–style formula to produce a retouched image, preserving skin texture and avoiding obvious warping artifacts.

## Overview

- **Input:** RGB portrait image (e.g. 1024×1024).
- **Output:** Retouched image with reduced double chin.
- **Method:** A U-Net (`BaseUNetHalf`) predicts a 3-channel blend map. The final image is computed as `retouched = clamp(image + 2 * blend_map - 1)` (Linear Light–style blend). Training uses paired data: original images and retouched (edited) images; target blend maps are derived from these pairs.

## Repository structure

```
double_chin/
├── inference.py          # Inference script & DoubleChinRemover class
├── example_inference.py  # Example usage of inference
├── INFERENCE_README.md   # Detailed inference docs & CLI options
├── default.yaml          # Default training config (dataset, hyperparams)
├── blend_map.yaml        # Blend-map training config (used by train_blend_map.py)
├── src/
│   ├── models/
│   │   └── unet.py       # BaseUNetHalf architecture
│   ├── blend/
│   │   └── blend_map.py  # apply_blend_formula, compute_target_blend_map, flow utils
│   ├── data/
│   │   └── dataset.py    # BlendMapDataset, data loaders
│   ├── losses/
│   │   └── losses.py     # CombinedLoss (MSE, perceptual, TV)
│   └── utils/            # Checkpoint loading, metrics, viz
└── tools/
    ├── train_blend_map.py  # Distributed training for blend-map model
    ├── train.py            # Training for flow-based model (alternative)
    └── run_training.py     # Runner that uses YAML config
```

## Requirements

```bash
pip install torch torchvision opencv-python numpy pyyaml
# Optional: for training logging
pip install wandb tqdm
```

## Dataset

Training expects:

1. **Data root** directory containing:
   - `original_image/` — original photos
   - `edited_image/` — retouched (ground-truth) photos
   - `blendmap_image/` — optional; precomputed blend maps as `.npy` (3-channel). If missing, they are computed from (original, edited) pairs and cached.

2. **JSON manifest** (e.g. `full_dataset.json`) listing samples. Each entry should include:
   - `original_image`: path relative to data root (e.g. `original_image/xxx.jpg`)
   - `edited_image`: path to retouched image (e.g. `edited_image/xxx.jpg`)
   - `blendmap_image`: path for cached blend map (optional; can be derived from `edited_image` path)
   - `source`: optional; samples with `source == "aadan_double_chin"` are excluded

Config in `blend_map.yaml` / `default.yaml`:
- `data_root`: path to dataset root
- `data_json`: path to the JSON file

## Training

Blend-map model (recommended):

```bash
# From repo root; uses blend_map.yaml by default
python tools/train_blend_map.py
```

- Config is read from `blend_map.yaml` (see `load_config()` in `train_blend_map.py`). Override by changing the default path in that function or adding a CLI argument.
- Uses **distributed data parallel** (DDP) over all visible GPUs; key training options (epochs, batch size, lr, loss weights, save dir, WandB) are set in the YAML.
- Checkpoints are saved under `save_dir` (e.g. `weights-blend-maps/<project_suffix>/`), including best and periodic epoch checkpoints.

Flow-based model (alternative):

- `tools/train.py` trains a flow-based variant; see `default.yaml` and `run_training.py` for config and entry point.

## Inference

**CLI:**

```bash
python inference.py --input path/to/input.jpg --output path/to/output.jpg
```

Options: `--model` (weights path), `--device` (cpu/cuda/auto), `--img_size` (default 1024), `--save_flow` (save flow/blend viz). See `INFERENCE_README.md` for full CLI and examples.

**Python API:**

```python
from inference import DoubleChinRemover, run_inference, save_img
import cv2

model_path = "weights-blend-maps/blend_maps_v2_5k/double_chin_bmap_best.pth"
remover = DoubleChinRemover(model_path=model_path, device="auto", img_size=1024)

# Load image (RGB)
image_bgr = cv2.imread("input.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Run pipeline
retouched, blend_map = run_inference(remover, image_rgb)
save_img(retouched, "output.jpg")
```

Model weights are expected at a path like `weights-blend-maps/blend_maps_v2_5k/double_chin_bmap_best.pth` unless overridden with `--model` or in code.

## Model architecture

- **Network:** `BaseUNetHalf` in `src/models/unet.py`: encoder–decoder U-Net with skip connections.
- **Input:** 3-channel RGB, normalized to [0, 1], resized to `img_size` (e.g. 1024).
- **Output:** 3-channel blend map (same resolution); last-layer activation is sigmoid (blend map in [0, 1]).
- **Application:** `apply_blend_formula(image, blend_map)` in `src/blend/blend_map.py`: `clamp(image + 2 * blend_map - 1)` (numpy or torch).

## Losses (training)

- Blend map MSE
- Retouched image MSE (vs ground truth)
- Perceptual loss (VGG16 features)
- Total variation (smoothness)

Weights are set in the YAML (`lambda_blend_mse`, `lambda_image_mse`, `lambda_perc`, `lambda_tv`).

## References

- **Inference details:** `INFERENCE_README.md` (CLI, programmatic usage, troubleshooting, performance).
- **Weights:** Place trained `.pth` under e.g. `weights-blend-maps/<run>/double_chin_bmap_best.pth` or pass `--model` / `model_path` accordingly.
