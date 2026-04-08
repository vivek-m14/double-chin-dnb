#!/usr/bin/env python3
"""
End-to-end double-chin removal pipeline.

1. RetinaFace detects all faces in the input image.
2. Each face is cropped (with configurable padding).
3. The blend-map model removes the double chin on the crop.
4. The retouched crop is pasted back into the full image.

Usage:
    python run_pipeline.py -i photo.jpg -o result.jpg
    python run_pipeline.py -i photo.jpg -o result.jpg --crop-padding 0.6
    python run_pipeline.py -i /path/to/folder -o /path/to/output_folder
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Add face-height detector from the YOLO spline project ──
_YOLO_SPLINE_DIR = str(
    Path(__file__).resolve().parent.parent
    / "ml_training/engine_clean/YOLO_spline_detection"
)
sys.path.insert(0, _YOLO_SPLINE_DIR)
from face_height_detector import FaceHeightDetector

# ── Add src/ for this project's imports ──
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference import DoubleChinRemover, run_inference, save_img

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Default model path
DEFAULT_MODEL = "weights-blend-maps/blend_maps_v3_2/double_chin_bmap_best.pth"


def collect_images(paths: list[str]) -> list[Path]:
    """Expand CLI paths (files or directories) into image file paths."""
    result: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for ext in IMAGE_EXTS:
                result.extend(sorted(pp.glob(f"*{ext}")))
                result.extend(sorted(pp.glob(f"*{ext.upper()}")))
        elif pp.is_file() and pp.suffix.lower() in IMAGE_EXTS:
            result.append(pp)
    # deduplicate
    seen: set = set()
    deduped: list[Path] = []
    for p in result:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            deduped.append(p)
    return deduped


def process_image(
    image_bgr: np.ndarray,
    face_detector: FaceHeightDetector,
    remover: DoubleChinRemover,
    crop_padding: float = 0.5,
) -> dict:
    """
    Detect faces → crop → run double-chin removal → paste back.

    Returns dict with:
        composite   – full image with retouched faces pasted back (BGR)
        faces       – list of per-face info dicts
        n_faces     – number of detected faces
    """
    ih, iw = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ── Detect faces ──
    faces = face_detector.detect_all_faces(
        image_bgr,
        crop_padding=(crop_padding, crop_padding),
    )

    if not faces:
        return {"composite": image_bgr.copy(), "faces": [], "n_faces": 0}

    composite = image_bgr.copy()
    face_results: list[dict] = []

    for fi, face_info in enumerate(faces):
        x1, y1, x2, y2 = face_info["crop_bbox"]
        crop_rgb = image_rgb[y1:y2, x1:x2].copy()

        # Run blend-map inference on the crop
        retouched_crop_rgb, blend_map = run_inference(remover, crop_rgb)

        # Paste back (convert retouched crop to BGR)
        composite[y1:y2, x1:x2] = cv2.cvtColor(retouched_crop_rgb, cv2.COLOR_RGB2BGR)

        face_results.append({
            "face_index": fi,
            "crop_bbox": [x1, y1, x2, y2],
            "face_height": face_info["face_height"],
            "crop_size": (x2 - x1, y2 - y1),
            "blend_map": blend_map,
        })

    return {
        "composite": composite,
        "faces": face_results,
        "n_faces": len(faces),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Face-detect → crop → double-chin removal pipeline",
    )
    parser.add_argument(
        "--input", "-i", type=str, nargs="+", required=True,
        help="Input image file(s) or directory",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output image path (single file) or directory (batch)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=DEFAULT_MODEL,
        help="Path to blend-map model weights (.pth)",
    )
    parser.add_argument(
        "--device", "-d", type=str, default="auto",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for blend-map model",
    )
    parser.add_argument(
        "--crop-padding", type=float, default=0.5,
        help="Fractional padding around detected face bbox (default: 0.5)",
    )
    parser.add_argument(
        "--save-crops", action="store_true",
        help="Save per-face before/after crops as separate images",
    )
    parser.add_argument(
        "--save-blend-maps", action="store_true",
        help="Save per-face blend maps (raw .npy + visual .png)",
    )
    parser.add_argument(
        "--save-composite", action="store_true",
        help="Save the full composite image with retouched faces",
    )
    parser.add_argument(
        "--retina-onnx", type=str, default=None,
        help="Path to RetinaFace ONNX model (default: cached)",
    )
    parser.add_argument(
        "--decoder-onnx", type=str, default=None,
        help="Path to RetinaFace decoder ONNX model (default: cached)",
    )

    args = parser.parse_args()

    # ── Collect images ──
    images = collect_images(args.input)
    if not images:
        print("[ERROR] No images found.")
        sys.exit(1)

    is_batch = len(images) > 1 or Path(args.input[0]).is_dir()
    output_path = Path(args.output)

    if is_batch:
        output_path.mkdir(parents=True, exist_ok=True)

    # ── Load models ──
    print("Loading RetinaFace detector...")
    fh_kwargs = {}
    if args.retina_onnx:
        fh_kwargs["retina_onnx"] = args.retina_onnx
    if args.decoder_onnx:
        fh_kwargs["decoder_onnx"] = args.decoder_onnx
    face_detector = FaceHeightDetector(**fh_kwargs)

    print(f"Loading blend-map model from {args.model}...")
    remover = DoubleChinRemover(model_path=args.model, device=args.device)
    print(f"  Device: {remover.device}")

    # ── Process ──
    print(f"\nProcessing {len(images)} image(s)...\n")
    total_faces = 0

    for img_path in images:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [SKIP] Cannot read: {img_path}")
            continue

        result = process_image(
            image_bgr, face_detector, remover,
            crop_padding=args.crop_padding,
        )

        n = result["n_faces"]
        total_faces += n

        # ── Determine output file path ──
        if is_batch:
            out_file = output_path / img_path.name
        else:
            # single image: output is a file path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_file = output_path

        if args.save_composite:
            cv2.imwrite(str(out_file), result["composite"])
            print(f"  {img_path.name}  →  {n} face(s)  →  {out_file}")
        else:
            print(f"  {img_path.name}  →  {n} face(s)")

        # ── Optional: save per-face crops ──
        if args.save_crops and result["faces"]:
            crops_dir = (output_path if is_batch else output_path.parent) / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)

            for fi_info in result["faces"]:
                fi = fi_info["face_index"]
                x1, y1, x2, y2 = fi_info["crop_bbox"]
                
                # Save before crop
                before_crop = image_bgr[y1:y2, x1:x2]
                before_path = crops_dir / f"{img_path.stem}_face{fi}_before.jpg"
                cv2.imwrite(str(before_path), before_crop)
                
                # Save after crop
                after_crop = result["composite"][y1:y2, x1:x2]
                after_path = crops_dir / f"{img_path.stem}_face{fi}_after.jpg"
                cv2.imwrite(str(after_path), after_crop)

        # ── Optional: save blend maps ──
        if args.save_blend_maps and result["faces"]:
            bmap_dir = (output_path if is_batch else output_path.parent) / "blend_maps"
            bmap_dir.mkdir(parents=True, exist_ok=True)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            for fi_info in result["faces"]:
                fi = fi_info["face_index"]
                bmap = fi_info["blend_map"]  # [H, W, 3] float32 in [0, 1]
                x1, y1, x2, y2 = fi_info["crop_bbox"]

                # Raw .npy (lossless, at model output resolution 1024x1024)
                npy_path = bmap_dir / f"{img_path.stem}_face{fi}_blend.npy"
                np.save(str(npy_path), bmap)

                # Save the face crop (the image region this blend map corresponds to)
                crop_bgr = image_bgr[y1:y2, x1:x2]
                crop_path = bmap_dir / f"{img_path.stem}_face{fi}_crop.jpg"
                cv2.imwrite(str(crop_path), crop_bgr)

                # Visual .png (blend map scaled to 0-255)
                bmap_vis = (np.clip(bmap, 0, 1) * 255).astype(np.uint8)
                png_path = bmap_dir / f"{img_path.stem}_face{fi}_blend.png"
                cv2.imwrite(str(png_path), cv2.cvtColor(bmap_vis, cv2.COLOR_RGB2BGR))

                # Deviation from neutral (0.5) — highlights where edits happen
                diff = np.abs(bmap - 0.5)
                diff_vis = (np.clip(diff * 4, 0, 1) * 255).astype(np.uint8)  # 4x amplify
                diff_path = bmap_dir / f"{img_path.stem}_face{fi}_blend_diff.png"
                cv2.imwrite(str(diff_path), cv2.cvtColor(diff_vis, cv2.COLOR_RGB2BGR))

    print(f"\nDone. {len(images)} image(s), {total_faces} face(s) processed.")


if __name__ == "__main__":
    main()
