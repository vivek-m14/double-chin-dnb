# Double Chin Removal — Inference & Pipeline

This repository provides two ways to run double chin removal:

1. **`inference.py`** — single-image inference on a pre-cropped face
2. **`run_pipeline.py`** — end-to-end pipeline: face detection → crop → double-chin removal → paste back

---

## Model Overview

The model uses a `BaseUNetHalf` U-Net architecture to predict a **3-channel blend map** ∈ [0, 1] (sigmoid output). The blend map is applied via a Linear Light formula:

```
retouched = clamp(image + 2 * blend_map - 1, 0, 1)
```

A neutral (no-change) blend map is all **0.5**. The edits are predominantly shadow/brightness corrections in the chin area.

## Requirements

```bash
pip install torch torchvision opencv-python numpy
```

For the full pipeline (`run_pipeline.py`), you also need access to `FaceHeightDetector` from the YOLO spline detection project.

---

## 1. Single-Image Inference (`inference.py`)

Use this when you already have a cropped face image.

### CLI

```bash
# Basic usage
python inference.py -i input.jpg -o output.jpg

# Custom model path
python inference.py -i input.jpg -o output.jpg -m path/to/model.pth

# Save blend map visualization
python inference.py -i input.jpg -o output.jpg --save_flow

# Force CPU
python inference.py -i input.jpg -o output.jpg -d cpu
```

#### Options

| Flag | Description | Default |
|---|---|---|
| `-i, --input` | Input image path | (required) |
| `-o, --output` | Output image path | (required) |
| `-m, --model` | Model weights (.pth) | `weights-blend-maps/blend_maps_v3_2/double_chin_bmap_best.pth` |
| `-d, --device` | `cpu`, `cuda`, `mps`, or `auto` | `auto` |
| `--img_size` | Input size for model | `1024` |
| `--save_flow` | Save blend map as PNG | off |

### Programmatic Usage

```python
from inference import DoubleChinRemover, run_inference, save_img
import cv2

# Load image (BGR → RGB)
image_bgr = cv2.imread("input.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Initialize model
remover = DoubleChinRemover(
    model_path="weights-blend-maps/blend_maps_v3_2/double_chin_bmap_best.pth",
    device="auto",
)

# Run inference — returns RGB uint8 retouched image + blend map
retouched, blend_map = run_inference(remover, image_rgb)

# Save (handles RGB → BGR internally)
save_img(retouched, "output.jpg")
```

---

## 2. Full Pipeline (`run_pipeline.py`)

Handles arbitrary photos with one or more faces: detects faces → crops with padding → runs blend-map model per face → pastes back.

### CLI

```bash
# Single image
python run_pipeline.py -i photo.jpg -o result.jpg --save-composite

# Batch (directory)
python run_pipeline.py -i /path/to/photos/ -o /path/to/output/ --save-composite

# Custom crop padding (default 0.5 = 50% around face bbox)
python run_pipeline.py -i photo.jpg -o result.jpg --crop-padding 0.6 --save-composite

# Save per-face crops and blend maps
python run_pipeline.py -i photo.jpg -o result.jpg --save-crops --save-blend-maps --save-composite
```

#### Options

| Flag | Description | Default |
|---|---|---|
| `-i, --input` | Input image(s) or directory | (required) |
| `-o, --output` | Output file or directory | (required) |
| `-m, --model` | Blend-map model weights | `weights-blend-maps/blend_maps_v3_2/double_chin_bmap_best.pth` |
| `-d, --device` | Device for model | `auto` |
| `--crop-padding` | Fractional padding around face bbox | `0.5` |
| `--save-composite` | Save the full composite image | off |
| `--save-crops` | Save per-face before/after crops | off |
| `--save-blend-maps` | Save per-face blend maps (.npy + .png) | off |
| `--retina-onnx` | Path to RetinaFace ONNX model | cached |
| `--decoder-onnx` | Path to decoder ONNX model | cached |

### Output Structure

When using `--save-crops` and `--save-blend-maps`:

```
output/
├── photo.jpg                           # composite (--save-composite)
├── crops/
│   ├── photo_face0_before.jpg
│   └── photo_face0_after.jpg
└── blend_maps/
    ├── photo_face0_blend.npy           # raw blend map (float32)
    ├── photo_face0_blend.png           # visual blend map
    ├── photo_face0_blend_diff.png      # deviation from neutral (4x amplified)
    └── photo_face0_crop.jpg            # original face crop
```

---

## Model Architecture

| Property | Value |
|---|---|
| Architecture | `BaseUNetHalf` (U-Net, 5-level encoder-decoder) |
| Input | 3-channel RGB, 1024×1024 |
| Output | 3-channel blend map ∈ [0, 1] (sigmoid) |
| Parameters | ~6.3M |
| Channels | 64 → 128 → 256 → 256 → 256 |

## Checkpoint Compatibility

`load_checkpoint()` handles:
- Checkpoints with `model_state_dict` or `state_dict` keys
- Raw state dicts (no wrapper key)
- DDP-saved checkpoints (`module.` prefix is automatically stripped)

## Troubleshooting

| Issue | Solution |
|---|---|
| Model not found | Verify the `.pth` path exists |
| CUDA out of memory | Use `--device cpu` or reduce `--img_size` |
| Import errors | Run from repo root so `src/` is importable |
| DDP checkpoint error | Already handled — `module.` prefix is auto-stripped |