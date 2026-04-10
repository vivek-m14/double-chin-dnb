#!/usr/bin/env python3
"""
Create SSIM-tier folders with flat paired symlinks inside each.

Reads the SSIM CSV produced by ``analyze_ssim.py`` and creates one sub-folder
per tier (``negligible/``, ``mild/``, ``moderate/``, ``significant/``).
Inside each tier folder every matched pair gets two symlinks::

    {normalized_key}_orig{ext}   →  /abs/path/to/original/file
    {normalized_key}_edit{ext}   →  /abs/path/to/edited/file

This lets you browse each severity bucket side-by-side in Finder or any
image viewer — just sort by name.

Optionally pass ``--data_json`` + ``--exclude_source`` to drop images
from a particular data source (e.g. ``aadan_double_chin``) before
creating symlinks.

Usage::

    python tools/make_paired_flat_folder.py \\
        --original_dir  .../original/ \\
        --edited_dir    .../edited/ \\
        --ssim_csv      .../ssim_analysis_v4/ssim_metrics.csv \\
        --output_dir    .../paired_by_ssim \\
        --data_json     .../data_v3_data_final.json \\
        --exclude_source aadan_double_chin
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Matching helpers  (mirrors analyze_ssim.py)
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


def build_file_index(directory: str) -> dict[str, str]:
    """Map normalized_key → filename for every valid image in *directory*."""
    idx = {}
    for fname in os.listdir(directory):
        _, ext = os.path.splitext(fname)
        if ext.lower() in VALID_EXTS:
            idx[_normalize_key(fname)] = fname
    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Create SSIM-tier folders with paired symlinks inside."
    )
    ap.add_argument("--original_dir", required=True, help="Dir with original images")
    ap.add_argument("--edited_dir", required=True, help="Dir with edited images")
    ap.add_argument("--ssim_csv", required=True,
                    help="SSIM metrics CSV from analyze_ssim.py (must have base_name & ssim_tier columns)")
    ap.add_argument("--output_dir", required=True, help="Root output folder (tier sub-folders created inside)")
    ap.add_argument(
        "--data_json", default=None,
        help="JSON manifest with 'original_image' and 'source' fields (for source-based exclusion)",
    )
    ap.add_argument(
        "--exclude_source", default=None, nargs="+",
        help="Source value(s) to exclude, e.g. 'aadan_double_chin'. Requires --data_json.",
    )
    ap.add_argument(
        "--relative", action="store_true",
        help="Create relative symlinks instead of absolute (default: absolute)",
    )
    args = ap.parse_args()

    if args.exclude_source and not args.data_json:
        ap.error("--exclude_source requires --data_json")

    original_dir = os.path.abspath(args.original_dir)
    edited_dir = os.path.abspath(args.edited_dir)
    output_dir = os.path.abspath(args.output_dir)

    # ------------------------------------------------------------------
    # 1. Read the SSIM CSV → { base_name: (ssim_tier, original_filename) }
    # ------------------------------------------------------------------
    tier_map: dict[str, str] = {}
    orig_filename_map: dict[str, str] = {}   # base_name → original_filename
    with open(args.ssim_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tier_map[row["base_name"]] = row["ssim_tier"]
            orig_filename_map[row["base_name"]] = row.get("original_filename", "")

    if not tier_map:
        print("ERROR: SSIM CSV is empty.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1b. (Optional) Build source-exclusion set from JSON manifest
    # ------------------------------------------------------------------
    excluded_stems: set[str] = set()
    if args.data_json and args.exclude_source:
        exclude_set = set(args.exclude_source)
        with open(args.data_json) as f:
            manifest = json.load(f)
        for entry in manifest:
            if entry.get("source") in exclude_set:
                stem = os.path.splitext(os.path.basename(entry["original_image"]))[0]
                excluded_stems.add(stem)
        print(f"Excluding sources: {sorted(exclude_set)}")
        print(f"Excluded stems   : {len(excluded_stems):,d}")

    # Filter tier_map: remove entries whose original_filename stem is excluded
    if excluded_stems:
        before = len(tier_map)
        to_remove = []
        for bn in tier_map:
            orig_fn = orig_filename_map.get(bn, "")
            stem = os.path.splitext(orig_fn)[0] if orig_fn else bn
            if stem in excluded_stems:
                to_remove.append(bn)
        for bn in to_remove:
            del tier_map[bn]
        print(f"Filtered out     : {len(to_remove):,d} / {before:,d} rows\n")

    # ------------------------------------------------------------------
    # 2. Build file-name indexes for both directories
    # ------------------------------------------------------------------
    orig_idx = build_file_index(original_dir)
    edit_idx = build_file_index(edited_dir)

    # ------------------------------------------------------------------
    # 3. Create tier folders and populate with symlinks
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    stats: dict[str, int] = defaultdict(int)
    created = 0
    skipped = 0
    no_file = 0

    for base_name, tier in sorted(tier_map.items()):
        if base_name not in orig_idx or base_name not in edit_idx:
            no_file += 1
            continue

        orig_fname = orig_idx[base_name]
        edit_fname = edit_idx[base_name]
        orig_path = os.path.join(original_dir, orig_fname)
        edit_path = os.path.join(edited_dir, edit_fname)

        tier_dir = os.path.join(output_dir, tier)
        os.makedirs(tier_dir, exist_ok=True)

        orig_ext = os.path.splitext(orig_fname)[1]
        edit_ext = os.path.splitext(edit_fname)[1]

        link_orig = os.path.join(tier_dir, f"{base_name}_orig{orig_ext}")
        link_edit = os.path.join(tier_dir, f"{base_name}_edit{edit_ext}")

        for link_path, target_path in [(link_orig, orig_path), (link_edit, edit_path)]:
            if os.path.exists(link_path) or os.path.islink(link_path):
                skipped += 1
                continue
            if args.relative:
                target_path = os.path.relpath(target_path, tier_dir)
            os.symlink(target_path, link_path)
            created += 1

        stats[tier] += 1

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    total_pairs = sum(stats.values())
    print(f"SSIM CSV rows (after filter): {len(tier_map)}")
    print(f"Pairs linked   : {total_pairs}")
    print(f"Symlinks created: {created}")
    if skipped:
        print(f"Symlinks skipped: {skipped} (already exist)")
    if no_file:
        print(f"CSV rows w/o matching files: {no_file}")
    print()
    print("Per-tier breakdown:")
    for tier in ("negligible", "mild", "moderate", "significant"):
        n = stats.get(tier, 0)
        tier_dir = os.path.join(output_dir, tier)
        exists = "✓" if os.path.isdir(tier_dir) else "–"
        print(f"  {exists} {tier:12s}  {n:>6,d} pairs")
    print(f"\nOutput root: {output_dir}")


if __name__ == "__main__":
    main()
