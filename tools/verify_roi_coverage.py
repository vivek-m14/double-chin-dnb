#!/usr/bin/env python3
"""Verify that a fixed bottom-N% crop captures all (or nearly all) edit pixels.

This script loads image pairs, computes blend maps and edit masks, then
reports what fraction of edit pixels fall within the bottom crop region
for a range of crop fractions (default: 30%–70%).

Usage
-----
Folder-scan mode (no JSON needed):
    python tools/verify_roi_coverage.py \
        --original_dir /path/to/original/ \
        --edited_dir   /path/to/edited/ \
        --resize 1024

JSON manifest mode:
    python tools/verify_roi_coverage.py \
        --data_root /workspace \
        --data_json /workspace/data_v3/data.json \
        --resize 1024

Options:
    --crop_fractions  0.3 0.4 0.5 0.6  (which bottom-N% values to test)
    --max_pairs 0                       (0 = all, otherwise subsample)
    --noise_threshold 0.01176           (3/255, for blend map neutral noise)
    --mask_threshold 0.02               (deviation from 0.5 to count as edit)
    --output_csv /path/to/results.csv   (optional, save per-pair results)
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def compute_blend_map_np(original: np.ndarray, edited: np.ndarray,
                         noise_threshold: float = 3.0 / 255.0) -> np.ndarray:
    """Compute blend map: (edited - original + 1) / 2, noise → 0.5."""
    orig = original.astype(np.float32) / 255.0
    edit = edited.astype(np.float32) / 255.0

    blend = (edit - orig + 1.0) / 2.0
    diff = np.abs(edit - orig)
    blend = np.where(diff < noise_threshold, 0.5, blend)
    return np.clip(blend, 0.0, 1.0)


def compute_edit_mask(blend_map: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """Binary edit mask: 1 where any channel deviates > threshold from 0.5.

    Returns (H, W) uint8 array.
    """
    deviation = np.abs(blend_map - 0.5)  # (H, W, 3)
    max_dev = deviation.max(axis=2)       # (H, W)
    mask = (max_dev > threshold).astype(np.uint8)
    # 3×3 dilation (like generate_blend_mask in losses.py)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def collect_pairs_from_dirs(original_dir: str, edited_dir: str):
    """Yield (original_path, edited_path) by matching filenames."""
    orig_files = {f.name: f for f in Path(original_dir).iterdir()
                  if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')}
    edit_files = {f.name: f for f in Path(edited_dir).iterdir()
                  if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')}
    common = sorted(orig_files.keys() & edit_files.keys())
    for name in common:
        yield str(orig_files[name]), str(edit_files[name])


def collect_pairs_from_json(data_root: str, data_json: str):
    """Yield (original_path, edited_path) from a JSON manifest."""
    with open(data_json) as f:
        manifest = json.load(f)
    for entry in manifest:
        orig = os.path.join(data_root, entry["original_image"])
        edit = os.path.join(data_root, entry["edited_image"])
        yield orig, edit


def analyze_pair(orig_path: str, edit_path: str, resize: int,
                 crop_fractions: list[float],
                 noise_threshold: float, mask_threshold: float) -> dict | None:
    """Analyze one pair, return per-crop-fraction coverage stats."""
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)
    if orig is None or edit is None:
        return None
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    edit = cv2.cvtColor(edit, cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (resize, resize))
    edit = cv2.resize(edit, (resize, resize))

    blend = compute_blend_map_np(orig, edit, noise_threshold)
    mask = compute_edit_mask(blend, mask_threshold)

    H, W = mask.shape
    total_edit_pixels = int(mask.sum())

    if total_edit_pixels == 0:
        # No edit region — trivially covered by any crop
        result = {
            "filename": os.path.basename(orig_path),
            "total_edit_pixels": 0,
            "edit_fraction": 0.0,
            "topmost_edit_row": -1,
            "topmost_edit_row_pct": -1.0,
            "bottommost_edit_row": -1,
        }
        for frac in crop_fractions:
            result[f"coverage_{frac:.0%}"] = 1.0
            result[f"missed_{frac:.0%}"] = 0
        return result

    # Find the topmost edit row (smallest row index with edit pixels)
    edit_rows = np.where(mask.any(axis=1))[0]
    topmost_edit_row = int(edit_rows[0])
    bottommost_edit_row = int(edit_rows[-1])

    result = {
        "filename": os.path.basename(orig_path),
        "total_edit_pixels": total_edit_pixels,
        "edit_fraction": total_edit_pixels / (H * W),
        "topmost_edit_row": topmost_edit_row,
        "topmost_edit_row_pct": topmost_edit_row / H,
        "bottommost_edit_row": bottommost_edit_row,
    }

    for frac in crop_fractions:
        crop_start_row = int(H * (1.0 - frac))  # bottom frac of image
        crop_mask = mask[crop_start_row:, :]
        edit_in_crop = int(crop_mask.sum())
        coverage = edit_in_crop / total_edit_pixels
        result[f"coverage_{frac:.0%}"] = coverage
        # How many edit pixels are ABOVE the crop boundary
        result[f"missed_{frac:.0%}"] = total_edit_pixels - edit_in_crop

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--original_dir", help="Folder of original images")
    g.add_argument("--data_json", help="JSON manifest path")

    parser.add_argument("--edited_dir", help="Folder of edited images (required with --original_dir)")
    parser.add_argument("--data_root", default="", help="Root prefix for JSON manifest paths")
    parser.add_argument("--resize", type=int, default=1024, help="Resize images to this square size")
    parser.add_argument("--crop_fractions", type=float, nargs="+",
                        default=[0.30, 0.40, 0.50, 0.55, 0.60, 0.70],
                        help="Bottom-N%% crop fractions to evaluate")
    parser.add_argument("--max_pairs", type=int, default=0, help="Max pairs to process (0=all)")
    parser.add_argument("--noise_threshold", type=float, default=3.0/255.0,
                        help="Blend map noise threshold (default: 3/255)")
    parser.add_argument("--mask_threshold", type=float, default=0.02,
                        help="Edit mask deviation threshold (default: 0.02)")
    parser.add_argument("--output_csv", default="", help="Save per-pair results to CSV")

    args = parser.parse_args()

    # Collect pairs
    if args.original_dir:
        if not args.edited_dir:
            parser.error("--edited_dir is required with --original_dir")
        pairs = list(collect_pairs_from_dirs(args.original_dir, args.edited_dir))
    else:
        pairs = list(collect_pairs_from_json(args.data_root, args.data_json))

    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]

    print(f"Pairs to analyze: {len(pairs)}")
    print(f"Crop fractions:   {args.crop_fractions}")
    print(f"Resize:           {args.resize}×{args.resize}")
    print()

    results = []
    skipped = 0

    for i, (orig_path, edit_path) in enumerate(pairs):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(pairs)}...")

        r = analyze_pair(orig_path, edit_path, args.resize,
                         args.crop_fractions, args.noise_threshold, args.mask_threshold)
        if r is None:
            skipped += 1
            continue
        results.append(r)

    if not results:
        print("ERROR: No valid pairs found.")
        sys.exit(1)

    # ── Save per-pair CSV ──
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        fieldnames = list(results[0].keys())
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Per-pair results saved to: {args.output_csv}")
        print()

    # ── Aggregate stats ──
    n = len(results)
    n_with_edits = sum(1 for r in results if r["total_edit_pixels"] > 0)
    results_with_edits = [r for r in results if r["total_edit_pixels"] > 0]

    print("=" * 70)
    print(f"RESULTS  ({n} pairs analyzed, {skipped} skipped, "
          f"{n_with_edits} with edit pixels)")
    print("=" * 70)

    # Edit region position stats
    if results_with_edits:
        top_rows_pct = [r["topmost_edit_row_pct"] for r in results_with_edits]
        print(f"\nTopmost edit row (as fraction of image height):")
        print(f"  Min:    {min(top_rows_pct):.3f}  (row {min(r['topmost_edit_row'] for r in results_with_edits)})")
        print(f"  Mean:   {np.mean(top_rows_pct):.3f}")
        print(f"  Median: {np.median(top_rows_pct):.3f}")
        print(f"  Max:    {max(top_rows_pct):.3f}  (row {max(r['topmost_edit_row'] for r in results_with_edits)})")

        edit_fracs = [r["edit_fraction"] for r in results_with_edits]
        print(f"\nEdit region size (fraction of total pixels):")
        print(f"  Min:    {min(edit_fracs):.4f}  ({min(edit_fracs)*100:.2f}%)")
        print(f"  Mean:   {np.mean(edit_fracs):.4f}  ({np.mean(edit_fracs)*100:.2f}%)")
        print(f"  Median: {np.median(edit_fracs):.4f}  ({np.median(edit_fracs)*100:.2f}%)")
        print(f"  Max:    {max(edit_fracs):.4f}  ({max(edit_fracs)*100:.2f}%)")

    # Per-crop-fraction coverage
    print(f"\n{'Crop':>8}  {'Mean':>8}  {'Min':>8}  {'p5':>8}  {'Median':>8}  "
          f"{'100% covered':>14}  {'Missed>0':>10}")
    print("-" * 78)

    for frac in args.crop_fractions:
        key = f"coverage_{frac:.0%}"
        coverages = [r[key] for r in results_with_edits] if results_with_edits else [1.0]
        mean_cov = np.mean(coverages)
        min_cov = min(coverages)
        p5_cov = np.percentile(coverages, 5)
        median_cov = np.median(coverages)
        full_coverage = sum(1 for c in coverages if c >= 1.0)
        partial = sum(1 for c in coverages if c < 1.0)

        print(f"  {frac:>5.0%}   {mean_cov:>7.4f}  {min_cov:>7.4f}  {p5_cov:>7.4f}  "
              f"{median_cov:>7.4f}  {full_coverage:>6}/{n_with_edits:<6}  {partial:>6}")

    # ── Outliers: pairs where 50% crop misses edit pixels ──
    if results_with_edits:
        target_frac = 0.50
        key = f"coverage_{target_frac:.0%}"
        outliers = [(r["filename"], r[key], r["topmost_edit_row_pct"])
                    for r in results_with_edits if r[key] < 1.0]
        outliers.sort(key=lambda x: x[1])

        if outliers:
            print(f"\n⚠️  Outliers — pairs where bottom-50% crop misses edit pixels "
                  f"({len(outliers)}/{n_with_edits}):")
            for fname, cov, top_pct in outliers[:20]:
                print(f"    {fname:40s}  coverage={cov:.4f}  topmost_edit_row={top_pct:.3f}")
            if len(outliers) > 20:
                print(f"    ... and {len(outliers) - 20} more")
        else:
            print(f"\n✅  Bottom-50% crop captures 100% of edit pixels for ALL {n_with_edits} pairs.")

    # ── Recommendation ──
    print(f"\n{'─' * 70}")
    print("RECOMMENDATION:")
    if results_with_edits:
        # Find minimum crop fraction that gives ≥99.9% coverage for ≥95% of pairs
        best = None
        for frac in sorted(args.crop_fractions):
            key = f"coverage_{frac:.0%}"
            coverages = [r[key] for r in results_with_edits]
            p5 = np.percentile(coverages, 5)
            if p5 >= 0.999:
                best = frac
                break
        if best:
            print(f"  Bottom {best:.0%} crop gives ≥99.9% coverage at the 5th percentile.")
            print(f"  Safe to use roi_top_fraction={best:.2f} in config.")
        else:
            print("  No tested crop fraction gives ≥99.9% coverage at p5.")
            print("  Consider using a larger crop or dynamic ROI.")
    print()


if __name__ == "__main__":
    main()
