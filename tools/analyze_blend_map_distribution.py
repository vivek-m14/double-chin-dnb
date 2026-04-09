#!/usr/bin/env python3
"""
Blend Map Value Distribution Analysis
Scans the dataset and reports the actual min/max/percentiles of target blend map values.
Key question: Do targets fall within [0.2, 0.8] (the model's ResidualTanh output range)?
"""

import json
import os
import random
import sys

import cv2
import numpy as np


DATA_ROOT = "/Users/vivekmittal/vk_ws/projects/double_chin/data/Double_Chin/double_chin_images"
JSON_PATH = os.path.join(
    DATA_ROOT,
    "double_chin_data_v3",
    "Double_Chin_double_chin_images_double_chin_data_v3_data_v3_data_final.json",
)

NOISE_THRESHOLD = 3.0 / 255.0
N_SAMPLES = 500


def load_pair(entry, data_root=DATA_ROOT):
    orig_path = os.path.join(data_root, entry["original_image"])
    edit_path = os.path.join(data_root, entry["edited_image"])
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)
    if orig is None or edit is None:
        return None, None
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    edit = cv2.cvtColor(edit, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if orig.shape[:2] != edit.shape[:2]:
        edit = cv2.resize(edit, (orig.shape[1], orig.shape[0]))
    return orig, edit


def compute_blend_map(original, edited, noise_threshold=NOISE_THRESHOLD):
    blend_map = (edited - original + 1.0) / 2.0
    diff = np.abs(edited - original)
    mask_noise = diff < noise_threshold
    blend_map = np.where(mask_noise, 0.5, blend_map)
    return np.clip(blend_map, 0.0, 1.0)


def main():
    with open(JSON_PATH) as f:
        manifest = json.load(f)
    usable = [e for e in manifest if e.get("source") != "aadan_double_chin"]
    print(f"Total: {len(manifest)}, Usable: {len(usable)}")

    random.seed(456)
    sample = random.sample(usable, min(N_SAMPLES, len(usable)))

    global_mins, global_maxs = [], []
    active_mins, active_maxs = [], []
    all_active_pixels = []
    per_image_outside_count = []
    failed = 0

    for idx, entry in enumerate(sample):
        if idx % 100 == 0:
            print(f"  Scanning {idx + 1}/{len(sample)}...", file=sys.stderr)
        orig, edit = load_pair(entry)
        if orig is None:
            failed += 1
            continue

        bmap = compute_blend_map(orig, edit)
        global_mins.append(float(bmap.min()))
        global_maxs.append(float(bmap.max()))

        active_mask = np.abs(bmap - 0.5) > 0.001
        if active_mask.any():
            av = bmap[active_mask]
            active_mins.append(float(av.min()))
            active_maxs.append(float(av.max()))

            n_outside = int(((av < 0.2) | (av > 0.8)).sum())
            per_image_outside_count.append(n_outside)

            if len(av) > 3000:
                all_active_pixels.extend(
                    np.random.choice(av, 3000, replace=False).tolist()
                )
            else:
                all_active_pixels.extend(av.tolist())

    all_active_pixels = np.array(all_active_pixels)
    n = len(global_mins)

    print(f"\nScanned {n} images ({failed} failed to load)")
    print("=" * 70)
    print("BLEND MAP VALUE DISTRIBUTION")
    print("=" * 70)

    print(f"\n  GLOBAL (all pixels including neutral 0.5):")
    print(f"    Absolute Min across dataset:  {min(global_mins):.6f}")
    print(f"    Absolute Max across dataset:  {max(global_maxs):.6f}")
    print(f"    Mean per-image min:           {np.mean(global_mins):.6f}")
    print(f"    Mean per-image max:           {np.mean(global_maxs):.6f}")

    print(f"\n  ACTIVE REGIONS ONLY (where edits exist, |bmap - 0.5| > 0.001):")
    print(f"    Absolute Min across dataset:  {min(active_mins):.6f}")
    print(f"    Absolute Max across dataset:  {max(active_maxs):.6f}")
    print(f"    Mean per-image active min:    {np.mean(active_mins):.6f}")
    print(f"    Mean per-image active max:    {np.mean(active_maxs):.6f}")

    print(f"\n  PERCENTILES OF ACTIVE BLEND MAP PIXELS ({len(all_active_pixels):,} sampled):")
    for p in [0.01, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 99.99]:
        val = np.percentile(all_active_pixels, p)
        flag = "  ✅" if 0.2 <= val <= 0.8 else "  ❌ OUTSIDE [0.2, 0.8]"
        print(f"    P{p:>6.2f}: {val:.6f}{flag}")

    # Range checks
    print(f"\n  MODEL RANGE COMPATIBILITY:")
    for lo, hi, label in [
        (0.2, 0.8, "ResidualTanh scale=0.3"),
        (0.15, 0.85, "scale=0.35"),
        (0.1, 0.9, "scale=0.4"),
    ]:
        outside = ((all_active_pixels < lo) | (all_active_pixels > hi)).mean() * 100
        imgs = sum(
            1
            for mn, mx in zip(active_mins, active_maxs)
            if mn < lo or mx > hi
        )
        print(
            f"    [{lo:.2f}, {hi:.2f}] ({label}):  "
            f"{outside:.4f}% pixels outside,  "
            f"{imgs}/{n} images ({imgs / n * 100:.1f}%) have any pixel outside"
        )

    # Histogram data (text-based)
    print(f"\n  HISTOGRAM OF ACTIVE PIXEL VALUES (excluding neutral 0.5):")
    bins = np.linspace(0.0, 1.0, 51)
    counts, edges = np.histogram(all_active_pixels, bins=bins)
    total = counts.sum()
    max_count = counts.max()
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        pct = counts[i] / total * 100
        bar = "█" * int(counts[i] / max_count * 40)
        if counts[i] > 0:
            marker = " ◄" if (lo < 0.2 or hi > 0.8) else ""
            print(f"    [{lo:.2f}-{hi:.2f}] {pct:>5.2f}% {bar}{marker}")

    # Verdict
    outside_02_08 = ((all_active_pixels < 0.2) | (all_active_pixels > 0.8)).mean() * 100
    print(f"\n{'=' * 70}")
    if outside_02_08 < 0.01:
        print(
            f"  ✅ VERDICT: Target blend maps are effectively ALWAYS within [0.2, 0.8].\n"
            f"     Only {outside_02_08:.4f}% of active pixels fall outside.\n"
            f"     ResidualTanh(scale=0.3) output range is SUFFICIENT.\n"
            f"     NO target clamping needed — the model can learn all targets."
        )
    elif outside_02_08 < 0.5:
        print(
            f"  ⚠️  VERDICT: Targets are MOSTLY within [0.2, 0.8].\n"
            f"     {outside_02_08:.3f}% of active pixels leak outside.\n"
            f"     This is negligible — no action required."
        )
    else:
        print(
            f"  ❌ VERDICT: {outside_02_08:.2f}% of target pixels are outside [0.2, 0.8].\n"
            f"     Consider clamping targets or increasing blend_scale."
        )


if __name__ == "__main__":
    main()
