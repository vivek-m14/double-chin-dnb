from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


def rotation_mat_to_euler_angles(rotation_mat: np.ndarray) -> Tuple[float, float, float]:
    """
    Same math as exp_face_pose.py:
    returns (pitch_deg, yaw_deg, roll_deg).
    """
    sy = math.sqrt(float(rotation_mat[0, 0]) ** 2 + float(rotation_mat[1, 0]) ** 2)

    if sy < 1e-6:
        x_radians = math.atan2(-float(rotation_mat[1, 2]), float(rotation_mat[1, 1]))
        y_radians = math.atan2(-float(rotation_mat[2, 0]), sy)
        z_radians = 0.0
    else:
        x_radians = math.atan2(float(rotation_mat[2, 1]), float(rotation_mat[2, 2]))
        y_radians = math.atan2(-float(rotation_mat[2, 0]), sy)
        z_radians = math.atan2(float(rotation_mat[1, 0]), float(rotation_mat[0, 0]))

    return math.degrees(x_radians), math.degrees(y_radians), math.degrees(z_radians)


def generate_transformation_matrix(face_crop_bgr: np.ndarray) -> np.ndarray:
    # Copied intent from exp_face_pose.py
    rows, cols = face_crop_bgr.shape[:2]
    scaled_maximum_edge = max(rows, cols) * 2.7
    scale = (120.0 * 2.0) / float(scaled_maximum_edge)
    tx = 60.0 - scale * (float(cols) / 2.0)
    ty = 60.0 - scale * (float(rows) / 2.0)
    return np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)


def _preprocess_for_pose_model(face_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess exactly like exp_face_pose.get_model_score():
    - resize to 256x256
    - warp affine to 120x120 using generated matrix
    - BGR->RGB, float32
    - min-max normalize to [-1, 1]
    - HWC->CHW, add batch => (1,3,120,120)
    """
    face_bgr = cv2.resize(face_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
    M = generate_transformation_matrix(face_bgr)
    warped = cv2.warpAffine(face_bgr, M, (120, 120), flags=cv2.INTER_LINEAR)
    rgb_float = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32)

    normalized = np.zeros_like(rgb_float, dtype=np.float32)
    cv2.normalize(
        rgb_float,
        normalized,
        alpha=-1.0,
        beta=1.0,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    x = np.transpose(normalized, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return x


def _list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort()
    return out


@dataclass
class FaceScore:
    roll: float
    pitch: float
    yaw: float


def score_face_image(session: ort.InferenceSession, img_bgr: np.ndarray) -> FaceScore:
    x = _preprocess_for_pose_model(img_bgr)
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: x})[0]
    # Expect shape like (1, 3, 4) or (1, 1, 3, 4); handle both
    out = np.asarray(out)
    if out.ndim == 4:
        # (1,1,3,4) -> (1,3,4)
        out = out[:, 0, :, :]
    rotation_mat = out[0, 0:3, 0:3]
    pitch, yaw, roll = rotation_mat_to_euler_angles(rotation_mat)
    return FaceScore(roll=float(roll), pitch=float(pitch), yaw=float(yaw))


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score face crops with pose model; writes one consolidated JSON.")
    ap.add_argument("--model-path", default=str(Path(__file__).with_name("face_pose.onnx")))
    ap.add_argument(
        "--input-dir",
        default="/workspace/face_recog/experiments/clustering_pipeline/garbage_faces/good_faces",
        help="Directory containing face crop images (recursively).",
    )
    ap.add_argument(
        "--output-json",
        default='/workspace/face_recog/experiments/clustering_pipeline/garbage_faces/pose_model/good_faces.json',
        help="Single output JSON file path. If omitted, uses <parent>/<input_dir_name>.json",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    input_dir = Path(args.input_dir).resolve()
    if args.output_json is None:
        out_json = input_dir.parent / f"{input_dir.name}.json"
    else:
        out_json = Path(args.output_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model_path), providers=providers)

    images = _list_images(input_dir)
    if not images:
        raise SystemExit(f"No images found under: {input_dir}")

    results = []
    for p in tqdm(images, desc="Scoring faces", dynamic_ncols=True):
        p = p.resolve()
        try:
            rel = str(p.relative_to(input_dir))
        except Exception:
            rel = p.name

        img = cv2.imread(str(p))
        row = {"rel_path": rel}
        if img is None or img.size == 0:
            row["error"] = "read_failed"
        else:
            try:
                s = score_face_image(session, img)
                row.update({"roll": s.roll, "pitch": s.pitch, "yaw": s.yaw})
            except Exception as e:
                row["error"] = f"inference_failed: {type(e).__name__}: {e}"
        results.append(row)

    payload = {
        "input_dir": str(input_dir),
        "model_path": str(Path(args.model_path).resolve()),
        "n_images": int(len(images)),
        "results": results,
    }
    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_json)
    print(f"Wrote: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

