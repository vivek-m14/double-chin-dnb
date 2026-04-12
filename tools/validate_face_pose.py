#!/usr/bin/env python3
"""
Validate the ONNX face-pose model on training data.

1. Run the pose model on all original face crops → CSV with yaw, pitch, roll.
2. Print histogram / distribution summary.
3. Create visual grid: sample N images per yaw bucket so you can eyeball
   whether the model's yaw predictions make sense.

Usage:
    python tools/validate_face_pose.py \
        --input-dir /path/to/original/ \
        --output-dir face_pose_validation/ \
        --sample-per-bucket 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Add project root so we can import face_orientation code
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from face_orientation.get_face_scores import score_face_image  # noqa: E402

import onnxruntime as ort  # noqa: E402


def list_images(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    out = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    out.sort()
    return out


# ── Yaw buckets for visual grid ──────────────────────────────────────────
BUCKETS = [
    ("profile_left",  -90, -45),
    ("three_q_left",  -45, -25),
    ("mild_left",     -25, -10),
    ("frontal",       -10,  10),
    ("mild_right",     10,  25),
    ("three_q_right",  25,  45),
    ("profile_right",  45,  90),
]


def bucket_label(yaw: float) -> str:
    for label, lo, hi in BUCKETS:
        if lo <= yaw < hi:
            return label
    return "extreme"


def make_grid(images: list[np.ndarray], cols: int = 8, cell_size: int = 200) -> np.ndarray:
    """Arrange images into a grid. Pads with black if fewer than cols*rows."""
    n = len(images)
    if n == 0:
        return np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        resized = cv2.resize(img, (cell_size, cell_size))
        grid[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size] = resized
    return grid


def main():
    ap = argparse.ArgumentParser(description="Validate face pose model on training data")
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory of face crop images (original/)",
    )
    ap.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "face_orientation" / "face_pose.onnx"),
        help="Path to face_pose.onnx",
    )
    ap.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "face_pose_validation"),
        help="Where to write CSV + visual grids",
    )
    ap.add_argument(
        "--sample-per-bucket",
        type=int,
        default=10,
        help="Number of sample images per yaw bucket for the visual grid",
    )
    ap.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Process at most N images (0 = all). Useful for quick tests.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print(f"Loading ONNX model from {args.model_path}")
    session = ort.InferenceSession(args.model_path, providers=["CPUExecutionProvider"])

    # ── List images ──
    images = list_images(input_dir)
    if args.max_images > 0:
        images = images[: args.max_images]
    print(f"Found {len(images)} images in {input_dir}")

    # ── Score all images ──
    csv_path = output_dir / "face_pose_scores.csv"
    bucket_samples: dict[str, list[tuple[float, Path]]] = {b[0]: [] for b in BUCKETS}
    bucket_samples["extreme"] = []

    results = []
    errors = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "yaw", "pitch", "roll", "bucket", "abs_yaw"])

        for p in tqdm(images, desc="Scoring faces"):
            img = cv2.imread(str(p))
            if img is None:
                errors += 1
                continue
            try:
                score = score_face_image(session, img)
            except Exception as e:
                tqdm.write(f"ERROR {p.name}: {e}")
                errors += 1
                continue

            bkt = bucket_label(score.yaw)
            row = (p.name, f"{score.yaw:.2f}", f"{score.pitch:.2f}", f"{score.roll:.2f}", bkt, f"{abs(score.yaw):.2f}")
            writer.writerow(row)
            results.append({"name": p.name, "yaw": score.yaw, "pitch": score.pitch, "roll": score.roll, "bucket": bkt, "path": p})

            # Collect samples for visual grid (keep first N per bucket)
            if len(bucket_samples[bkt]) < args.sample_per_bucket:
                bucket_samples[bkt].append((score.yaw, p))

    print(f"\nScored {len(results)} images ({errors} errors). CSV → {csv_path}")

    # ── Print distribution ──
    print("\n── Yaw Distribution ──")
    bucket_counts = {}
    for r in results:
        bucket_counts[r["bucket"]] = bucket_counts.get(r["bucket"], 0) + 1

    all_buckets = [b[0] for b in BUCKETS] + ["extreme"]
    total = len(results)
    for bkt in all_buckets:
        count = bucket_counts.get(bkt, 0)
        pct = 100.0 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {bkt:>16s}: {count:5d} ({pct:5.1f}%) {bar}")

    # ── Print pitch distribution too ──
    print("\n── Pitch Distribution ──")
    pitch_bins = [
        ("looking_down",  -90, -20),
        ("slight_down",   -20, -5),
        ("level",          -5,  5),
        ("slight_up",       5, 20),
        ("looking_up",     20, 90),
    ]
    for label, lo, hi in pitch_bins:
        count = sum(1 for r in results if lo <= r["pitch"] < hi)
        pct = 100.0 * count / total if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {label:>16s}: {count:5d} ({pct:5.1f}%) {bar}")

    # ── Print summary stats ──
    if results:
        yaws = [r["yaw"] for r in results]
        pitches = [r["pitch"] for r in results]
        print(f"\n── Summary Stats ──")
        print(f"  Yaw:   mean={np.mean(yaws):.1f}°  std={np.std(yaws):.1f}°  min={np.min(yaws):.1f}°  max={np.max(yaws):.1f}°")
        print(f"  Pitch: mean={np.mean(pitches):.1f}°  std={np.std(pitches):.1f}°  min={np.min(pitches):.1f}°  max={np.max(pitches):.1f}°")
        print(f"  |yaw| > 30°: {sum(1 for y in yaws if abs(y) > 30)} images ({100*sum(1 for y in yaws if abs(y) > 30)/total:.1f}%)")
        print(f"  |yaw| > 45°: {sum(1 for y in yaws if abs(y) > 45)} images ({100*sum(1 for y in yaws if abs(y) > 45)/total:.1f}%)")

    # ── Create visual grids ──
    print("\n── Creating visual grids ──")
    for bkt in all_buckets:
        samples = bucket_samples.get(bkt, [])
        if not samples:
            continue
        # Sort by yaw for nice ordering
        samples.sort(key=lambda x: x[0])
        grid_imgs = []
        for yaw_val, p in samples:
            img = cv2.imread(str(p))
            if img is None:
                continue
            # Add yaw text overlay
            img_resized = cv2.resize(img, (200, 200))
            cv2.putText(img_resized, f"y={yaw_val:.1f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            grid_imgs.append(img_resized)

        if grid_imgs:
            grid = make_grid(grid_imgs, cols=min(5, len(grid_imgs)), cell_size=200)
            grid_path = output_dir / f"grid_{bkt}.jpg"
            cv2.imwrite(str(grid_path), grid)
            print(f"  {bkt}: {len(grid_imgs)} samples → {grid_path}")

    # ── Also create one combined grid with a few from each bucket ──
    combined_imgs = []
    for bkt in all_buckets:
        samples = bucket_samples.get(bkt, [])
        if samples:
            samples.sort(key=lambda x: x[0])
            # Take up to 3 from each bucket for combined view
            for yaw_val, p in samples[:3]:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                img_resized = cv2.resize(img, (200, 200))
                cv2.putText(img_resized, f"{bkt}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(img_resized, f"y={yaw_val:.1f}", (5, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                combined_imgs.append(img_resized)

    if combined_imgs:
        grid = make_grid(combined_imgs, cols=7, cell_size=200)
        combined_path = output_dir / "grid_combined.jpg"
        cv2.imwrite(str(combined_path), grid)
        print(f"\n  Combined grid → {combined_path}")

    print(f"\nDone! Check {output_dir}/ for results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
