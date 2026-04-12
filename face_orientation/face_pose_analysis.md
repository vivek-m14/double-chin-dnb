# Face Pose Analysis — Checkpoint Summary

## Date: 12 April 2026

## Model
- **ONNX model**: `face_orientation/face_pose.onnx`
- **Inference script**: `face_orientation/get_face_scores.py`
- **Preprocessing**: resize to 256×256 → affine warp to 120×120 → normalize [-1, 1] → ONNX inference → rotation matrix → pitch/yaw/roll (degrees)
- **Validation verdict**: ✅ Model predictions match visual inspection. Trusted for filtering.

## Dataset Scored
- **Input directory**: `double_chin_data_v3/original/`
- **Total images scored**: 9,386 (0 errors)
- **Speed**: ~110 images/sec on CPU (onnxruntime 1.23.2)
- **Full CSV**: `face_pose_validation/face_pose_scores.csv` (columns: filename, yaw, pitch, roll, bucket, abs_yaw)

## Yaw Distribution

| Bucket | |Yaw| Range | Count | % |
|--------|-----------|------:|----:|
| frontal | 0°–10° | 4,973 | 53.0% |
| mild | 10°–20° | 2,416 | 25.7% |
| moderate | 20°–30° | 930 | 9.9% |
| three_quarter | 30°–45° | 623 | 6.6% |
| profile | 45°–60° | 302 | 3.2% |
| extreme | 60°+ | 142 | 1.5% |

**Summary stats**: mean yaw = 2.3°, std = 19.2°, range = [-85.5°, 84.0°]

## Pitch Distribution

| Bucket | Pitch Range | Count | % |
|--------|------------|------:|----:|
| looking_down | -90° to -20° | 687 | 7.3% |
| slight_down | -20° to -5° | 559 | 26.0% |
| level | -5° to 5° | 3,273 | 34.9% |
| slight_up | 5° to 20° | 2,020 | 21.5% |
| looking_up | 20°+ | 961 | 10.2% |

## Key Finding: Side Faces Have Warping, Not Just Shadows

Visual inspection of original vs. ground truth pairs for side-face images (|yaw| > 30°) revealed that the GT edits for these images contain **geometric warping** (chin reshaping, skin fold changes), not just shadow/color corrections. 

Our blend-map formulation `blend = (gt - orig + 1) / 2` assumes a **per-pixel color shift**. It cannot represent geometric warps. This means:
- Blend maps computed for side-face images are **broken labels** — noisy, high-magnitude values with no meaningful interpretation.
- Training on these labels causes **gradient pollution** that degrades frontal-face predictions.
- The model cannot intrinsically learn to ignore side faces because the loss actively penalizes the correct neutral output.

## Recommended Strategy: Force Neutral Blend Maps

For images above the yaw threshold:
- **Keep them in training** (so the model sees side faces and learns to output neutral).
- **Force blend map target = 0.5** (neutral — "no change") instead of computing from GT.
- This gives a clean learning signal: "side face → don't modify."

| Strategy | What model learns | Quality |
|----------|------------------|---------|
| Do nothing (current) | Tries to fit broken warped blend maps | ❌ Corrupts frontal predictions |
| Remove side faces entirely | Never sees side faces | ⚠️ Unpredictable at inference |
| **Force neutral for side faces** | "Side face → output 0.5" | ✅ Clean signal |

## Threshold: |yaw| > 30° ✅ (decided)

| Threshold | Images forced neutral | % of dataset |
|-----------|----------------------:|-------------:|
| |yaw| > 20° | 1,997 | 21.3% |
| |yaw| > 25° | 1,420 | 15.1% |
| **|yaw| > 30°** | **1,067** | **11.4%** |
| |yaw| > 45° | 444 | 4.7% |

Chosen threshold: **30°** — visual inspection confirmed warping starts appearing around this angle.

## Implementation: `gt = original` for side faces ✅ (done)

Instead of manually setting blend=0.5, we set **`gt = original`** for side-face images in `__getitem__`. This makes ALL losses self-consistent automatically:

- Blend map = `(original - original + 1) / 2` = **0.5** ✓
- Perceptual target = original ✓ (no conflicting warped GT)
- Image MSE target = original ✓
- Masked L1: mask is empty (blend=0.5 everywhere, below threshold) → loss = 0 naturally ✓
- **Zero special-casing in loss code**

Design decision: **Hard threshold** (binary, not soft ramp).

### Config keys added
```yaml
face_pose_csv: '/path/to/face_pose_scores.csv'  # empty to disable
yaw_threshold: 30                                 # |yaw| > this → neutral
```

### Files modified
- `src/data/dataset.py`: `BlendMapDataset` accepts `face_pose_yaw` dict + `yaw_threshold`; `create_data_loaders` loads CSV
- `src/utils/utils_blend.py`: Added `n_side_face_train`, `n_side_face_val` columns (backward-compatible schema migration)
- `tools/train_blend_map.py`: Accumulates side-face count per epoch, reduces across DDP ranks, logs to CSV
- `default.yaml`: Added default keys (disabled)
- `experiments/exp4_lite_residual_tanh_roi.yaml`: Enabled with threshold=30

### Backward compatibility: ✅ 6/6 tests passed
- Old configs without `face_pose_csv` → filtering disabled (empty dict, old behavior)
- Old CSVs without `n_side_face` columns → backed up and migrated automatically
- Resume training: works (schema migration handles the column mismatch)

## Artifacts

| File | Description |
|------|-------------|
| `face_pose_validation/face_pose_scores.csv` | Full CSV with per-image yaw, pitch, roll |
| `face_pose_validation/grid_*.jpg` | Visual grids per yaw bucket (10 samples each) |
| `face_pose_validation/grid_combined.jpg` | Combined grid with samples from all buckets |
| `face_pose_validation/grid_side_poses_input_vs_gt.jpg` | Top 20 side poses: original vs GT |
| `face_pose_validation/by_yaw/` | Symlinked folders bucketed by |yaw| (original + edited) |
| `tools/validate_face_pose.py` | Script to reproduce the analysis |

## Next Steps

1. **Copy `face_pose_scores.csv` to training server** at `/workspace/double_chin_data_v3/face_pose_scores.csv`
2. **Run experiment** with face pose filtering enabled (exp4 config already updated)
3. **Compare** exp4-without-pose-filter vs exp4-with-pose-filter (ablation)
