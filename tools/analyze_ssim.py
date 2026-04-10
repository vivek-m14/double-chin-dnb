#!/usr/bin/env python3
"""
SSIM-Only Warping Screening Tool
=================================
Lightweight alternative to ``analyze_warping.py`` — uses **only** Structural
Similarity (SSIM) to flag image pairs that may have geometric distortion.

Much faster than the full 6-technique pipeline (~50-100× depending on
resolution) because it skips optical flow, ORB features, template matching,
and edge analysis.

**Caveat**: SSIM drops for *any* structural change — shadow edits, colour
grading, and compression artefacts will also lower SSIM.  Use this as a
fast **screening pass**; feed flagged pairs into ``analyze_warping.py`` for
confirmation if needed.

Outputs:
  - Per-image SSIM metrics (CSV)
  - Aggregate statistics printed to console
"""

import argparse
import csv
import multiprocessing
import os
import re
import sys
import time
import traceback
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Whole-image SSIM classification thresholds
SSIM_NEGLIGIBLE = 0.995   # ≥ this → basically identical
SSIM_MILD       = 0.980   # ≥ this → minor edits, likely clean
SSIM_MODERATE   = 0.950   # ≥ this → noticeable change
# < SSIM_MODERATE           → significant structural change

# Edit-region SSIM thresholds (tighter, since we're only looking at the
# area that actually changed — SSIM there should still be high for a
# clean inpainting job)
ER_SSIM_NEGLIGIBLE = 0.98
ER_SSIM_MILD       = 0.92
ER_SSIM_MODERATE   = 0.88

# Pixel-diff threshold for detecting the edit region
EDIT_DIFF_THRESH   = 3
EDIT_DILATE_KERNEL = 15
EDIT_DILATE_ITERS  = 3

CSV_FIELDNAMES = [
    "base_name", "original_filename", "img_h", "img_w",
    "ssim_whole", "ssim_edit_region", "edit_region_pct",
    "ssim_level", "ssim_tier",
]

# ---------------------------------------------------------------------------
# Helpers (shared logic with analyze_warping.py, kept self-contained)
# ---------------------------------------------------------------------------

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _normalize_key(fname: str) -> str:
    """Canonical key: uuid_index regardless of naming convention."""
    base_no_ext, _ = os.path.splitext(fname)
    m = re.match(
        r'^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.\w+_(\d+)\.\w+$',
        fname,
    )
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return base_no_ext


def match_image_pairs(original_dir: str, edited_dir: str):
    """Match original ↔ edited images by base name."""
    def build_index(d):
        idx = {}
        for fname in os.listdir(d):
            _, ext = os.path.splitext(fname)
            if ext.lower() in VALID_EXTS:
                idx[_normalize_key(fname)] = fname
        return idx

    orig_idx = build_index(original_dir)
    edit_idx = build_index(edited_dir)
    common = sorted(set(orig_idx) & set(edit_idx))
    return [
        (os.path.join(original_dir, orig_idx[b]),
         os.path.join(edited_dir, edit_idx[b]), b)
        for b in common
    ]


def load_pair(orig_path: str, edit_path: str, max_dim: int = 512):
    """Load, normalise to 3-ch BGR, resize to common dimensions."""
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)
    if orig is None:
        raise ValueError(f"Cannot read original: {orig_path}")
    if edit is None:
        raise ValueError(f"Cannot read edited: {edit_path}")

    for tag, img in [("orig", orig), ("edit", edit)]:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if tag == "orig":
            orig = img
        else:
            edit = img

    if orig.shape[:2] != edit.shape[:2]:
        edit = cv2.resize(edit, (orig.shape[1], orig.shape[0]),
                          interpolation=cv2.INTER_AREA)

    if max_dim and max_dim > 0:
        h, w = orig.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            orig = cv2.resize(orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
            edit = cv2.resize(edit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return orig, edit


# ---------------------------------------------------------------------------
# SSIM computation
# ---------------------------------------------------------------------------


def compute_ssim(orig_gray: np.ndarray, edit_gray: np.ndarray):
    """Compute SSIM score and per-pixel SSIM map.

    Automatically adapts ``win_size`` for small images.
    """
    min_side = min(orig_gray.shape[:2])
    win = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if win < 3:
        return 1.0, np.ones_like(orig_gray, dtype=np.float32)
    score, smap = ssim(orig_gray, edit_gray, win_size=win, full=True)
    return float(score), smap.astype(np.float32)


def detect_edit_region(orig_gray: np.ndarray, edit_gray: np.ndarray):
    """Binary mask of the edited region (pixel-diff + dilation)."""
    diff = cv2.absdiff(orig_gray, edit_gray)
    mask = (diff > EDIT_DIFF_THRESH).astype(np.uint8)
    kernel = np.ones((EDIT_DILATE_KERNEL, EDIT_DILATE_KERNEL), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=EDIT_DILATE_ITERS)
    return mask


def compute_edit_region_ssim(ssim_map: np.ndarray, edit_mask: np.ndarray):
    """Mean SSIM inside the edit region only."""
    pixels = ssim_map[edit_mask > 0]
    if pixels.size < 10:
        return 1.0, 0.0
    pct = 100.0 * np.count_nonzero(edit_mask) / max(edit_mask.size, 1)
    return float(np.mean(pixels)), round(pct, 2)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_ssim(ssim_whole: float, ssim_edit: float) -> tuple:
    """Classify severity from whole-image and edit-region SSIM.

    Uses the *worse* of the two signals — whole-image catches large-area
    distortion while edit-region catches localised warping that would be
    diluted in the whole-image score.

    Returns:
        (label, tier) — *label* is a human-friendly emoji string for
        display; *tier* is a plain-text machine-friendly key
        (``negligible`` / ``mild`` / ``moderate`` / ``significant``).
    """
    # Edit-region tier
    if ssim_edit >= ER_SSIM_NEGLIGIBLE:
        er_tier = 0
    elif ssim_edit >= ER_SSIM_MILD:
        er_tier = 1
    elif ssim_edit >= ER_SSIM_MODERATE:
        er_tier = 2
    else:
        er_tier = 3

    # Whole-image tier
    if ssim_whole >= SSIM_NEGLIGIBLE:
        wi_tier = 0
    elif ssim_whole >= SSIM_MILD:
        wi_tier = 1
    elif ssim_whole >= SSIM_MODERATE:
        wi_tier = 2
    else:
        wi_tier = 3

    tier = max(er_tier, wi_tier)
    labels = ["✅ NEGLIGIBLE", "⚠️  MILD", "🔶 MODERATE", "🔴 SIGNIFICANT"]
    tiers  = ["negligible", "mild", "moderate", "significant"]
    return labels[tier], tiers[tier]


# ---------------------------------------------------------------------------
# Per-pair analysis
# ---------------------------------------------------------------------------


def analyze_pair(orig_path: str, edit_path: str, base_name: str, max_dim: int = 512):
    """Run SSIM analysis on a single image pair."""
    orig, edit = load_pair(orig_path, edit_path, max_dim=max_dim)
    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    edit_gray = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)

    ssim_whole, ssim_map = compute_ssim(orig_gray, edit_gray)
    edit_mask = detect_edit_region(orig_gray, edit_gray)
    ssim_edit, edit_pct = compute_edit_region_ssim(ssim_map, edit_mask)

    level, tier = classify_ssim(ssim_whole, ssim_edit)

    # Raw filename for reliable join with the training JSON manifest
    # (base_name is normalised and may differ from the JSON stem for
    # files using the uuid.ext_N.ext naming convention)
    original_filename = os.path.basename(orig_path)

    return {
        "base_name": base_name,
        "original_filename": original_filename,
        "img_h": orig.shape[0],
        "img_w": orig.shape[1],
        "ssim_whole": round(ssim_whole, 5),
        "ssim_edit_region": round(ssim_edit, 5),
        "edit_region_pct": edit_pct,
        "ssim_level": level,
        "ssim_tier": tier,
    }


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------


def _worker_init():
    cv2.setNumThreads(1)


def _worker(args_tuple):
    idx, total, orig_path, edit_path, base_name, max_dim = args_tuple
    try:
        metrics = analyze_pair(orig_path, edit_path, base_name, max_dim=max_dim)
        return {"status": "ok", "idx": idx, "total": total,
                "metrics": metrics, "base_name": base_name}
    except Exception as e:
        return {"status": "error", "idx": idx, "total": total,
                "error": str(e), "traceback": traceback.format_exc(),
                "base_name": base_name}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fast SSIM-only screening for warped image pairs."
    )
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--edited_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="ssim_analysis_output")
    parser.add_argument("--max_pairs", type=int, default=0,
                        help="0 = all")
    parser.add_argument("--max_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 = auto, 1 = sequential")
    args = parser.parse_args()

    # Input validation
    for label, path in [("original", args.original_dir), ("edited", args.edited_dir)]:
        if not os.path.isdir(path):
            print(f"ERROR: {label} directory not found: {path}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  SSIM SCREENING")
    print(f"  Original: {args.original_dir}")
    print(f"  Edited:   {args.edited_dir}")
    print(f"{'='*60}")

    pairs = match_image_pairs(args.original_dir, args.edited_dir)
    print(f"\nFound {len(pairs)} matched pairs.")

    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(pairs), args.max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]
        print(f"Sampled {args.max_pairs} pairs.")

    n_workers = args.workers if args.workers > 0 else min(os.cpu_count() or 4, 8)
    use_mp = n_workers > 1 and len(pairs) > 1

    all_metrics = []
    errors = 0
    t0 = time.time()
    n_done = 0

    def _log(result):
        nonlocal errors, n_done
        n_done += 1
        total = result["total"]
        bn = result["base_name"]
        if result["status"] == "ok":
            m = result["metrics"]
            all_metrics.append(m)
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (total - n_done) / rate if rate > 0 else 0
            print(
                f"  [{n_done:4d}/{total}] {bn:50s} "
                f"whole={m['ssim_whole']:.4f}  "
                f"edit={m['ssim_edit_region']:.4f}  "
                f"region={m['edit_region_pct']:.1f}%  "
                f"{m['ssim_level']}  "
                f"({rate:.1f} img/s  ETA {int(eta//60)}m{int(eta%60):02d}s)"
            )
        else:
            errors += 1
            print(f"  [{n_done:4d}/{total}] {bn:50s} ERROR: {result['error']}")
            tb = result.get("traceback", "")
            if tb:
                for line in tb.strip().splitlines()[-3:]:
                    print(f"      {line}")

    try:
        if use_mp:
            print(f"Using {n_workers} workers.")
            items = [
                (i, len(pairs), op, ep, bn, args.max_dim)
                for i, (op, ep, bn) in enumerate(pairs)
            ]
            with multiprocessing.Pool(n_workers, initializer=_worker_init) as pool:
                for r in pool.imap_unordered(_worker, items, chunksize=8):
                    _log(r)
        else:
            print("Running sequentially.")
            for i, (op, ep, bn) in enumerate(pairs):
                _log(_worker((i, len(pairs), op, ep, bn, args.max_dim)))
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted! Saving {len(all_metrics)} partial results …")

    wall = time.time() - t0
    print(f"\nProcessed {len(all_metrics)} pairs in {wall:.1f}s "
          f"({len(all_metrics)/max(wall,1):.1f} img/s), {errors} errors.")

    if not all_metrics:
        print("\nNo pairs analysed successfully.")
        return

    all_metrics.sort(key=lambda m: m["base_name"])

    # Save CSV
    csv_path = os.path.join(args.output_dir, "ssim_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nMetrics saved to: {csv_path}")

    # Aggregate stats
    whole_vals = np.array([m["ssim_whole"] for m in all_metrics])
    edit_vals  = np.array([m["ssim_edit_region"] for m in all_metrics])
    region_pcts = np.array([m["edit_region_pct"] for m in all_metrics])

    print(f"\n{'='*60}")
    print(f"  AGGREGATE ({len(all_metrics)} images)")
    print(f"{'='*60}")
    for name, arr in [
        ("SSIM whole-image", whole_vals),
        ("SSIM edit-region", edit_vals),
        ("Edit region size (%)", region_pcts),
    ]:
        print(f"  {name:22s}  mean={np.mean(arr):.4f}  "
              f"median={np.median(arr):.4f}  std={np.std(arr):.4f}  "
              f"min={np.min(arr):.4f}  max={np.max(arr):.4f}")

    # Classification distribution
    levels = [m["ssim_level"] for m in all_metrics]
    print(f"\n  Classification:")
    for lbl in ["✅ NEGLIGIBLE", "⚠️  MILD", "🔶 MODERATE", "🔴 SIGNIFICANT"]:
        cnt = levels.count(lbl)
        pct = 100.0 * cnt / len(levels)
        bar = "█" * int(pct / 2)
        print(f"    {lbl:22s}  {cnt:5d} ({pct:5.1f}%)  {bar}")

    # Worst cases
    sorted_m = sorted(all_metrics, key=lambda m: m["ssim_edit_region"])
    print(f"\n  Bottom 10 by edit-region SSIM:")
    for j, m in enumerate(sorted_m[:10]):
        print(f"    {j+1:2d}. {m['base_name']:50s}  "
              f"whole={m['ssim_whole']:.4f}  "
              f"edit={m['ssim_edit_region']:.4f}  "
              f"{m['ssim_level']}")

    print(f"\nDone! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
