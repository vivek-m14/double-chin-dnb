#!/usr/bin/env python3
"""
Visualize Blend Map Loss Functions
====================================
Loads a real image pair from the dataset, computes the target blend map,
simulates a "predicted" blend map (with controllable error), and visualizes
how the three blend-map losses see the error:

  1. **Global L1** — uniform penalty everywhere
  2. **Masked L1** — penalty only in the edit region
  3. **Unmasked L1** — penalty only outside the edit region (identity preservation)

Also shows the auto-derived edit mask and per-pixel loss heatmaps.

Usage::

    # Simple: just point at the folder containing original/ and edited/
    python tools/visualize_losses.py \
        --data_dir /path/to/double_chin_data_v3 \
        --index 42

    # Or with the data JSON for exact pair matching
    python tools/visualize_losses.py \
        --data_root /path/to/dataset_root \
        --data_json /path/to/data.json \
        --index 42
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.blend.blend_map import compute_target_blend_map
from src.losses.losses import generate_blend_mask, masked_l1_loss, unmasked_l1_loss


def load_pair_from_paths(img_path, gt_path, size=1024):
    """Load and resize an original / gt pair to float32 CHW tensors."""
    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    if gt is None:
        raise FileNotFoundError(f"Cannot read: {gt_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    gt = cv2.resize(gt, (size, size))

    img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    gt_t = torch.from_numpy(gt.astype(np.float32) / 255.0).permute(2, 0, 1)
    return img_t, gt_t, img, gt


def collect_pairs_from_dirs(data_dir):
    """Scan original/ and edited/ folders, match by stem name."""
    orig_dir = os.path.join(data_dir, "original")
    edit_dir = os.path.join(data_dir, "edited")
    if not os.path.isdir(orig_dir):
        raise FileNotFoundError(f"Missing folder: {orig_dir}")
    if not os.path.isdir(edit_dir):
        raise FileNotFoundError(f"Missing folder: {edit_dir}")

    # Build stem → path maps (stem = filename without extension)
    orig_map = {}
    for fn in sorted(os.listdir(orig_dir)):
        stem = os.path.splitext(fn)[0]
        orig_map[stem] = os.path.join(orig_dir, fn)

    edit_map = {}
    for fn in sorted(os.listdir(edit_dir)):
        stem = os.path.splitext(fn)[0]
        edit_map[stem] = os.path.join(edit_dir, fn)

    # Match by stem
    pairs = []
    for stem in sorted(orig_map.keys()):
        if stem in edit_map:
            pairs.append((orig_map[stem], edit_map[stem]))
    return pairs


def simulate_prediction(target_blend, mask, noise_scale=0.15, edit_error_scale=0.10):
    """Create a fake predicted blend map with realistic error patterns.

    - **Edit region**: target + moderate error (the model is "trying" here)
    - **Non-edit region**: target + small error (should stay near 0.5)

    This lets you see how masked vs unmasked losses respond differently.
    """
    pred = target_blend.clone()

    # Larger error inside the edit region (model is actively predicting)
    edit_noise = torch.randn_like(pred) * edit_error_scale
    # Smaller error outside (model should predict neutral 0.5)
    bg_noise = torch.randn_like(pred) * (noise_scale * 0.3)

    mask_3ch = mask.expand_as(pred)
    pred = pred + mask_3ch * edit_noise + (1 - mask_3ch) * bg_noise
    pred = pred.clamp(0, 1)
    return pred


def make_figure(
    original_np, gt_np,
    target_blend, pred_blend, mask,
    global_loss_map, masked_loss_map, unmasked_loss_map,
    losses, output_path,
):
    """Build the visualization figure."""
    fig = plt.figure(figsize=(28, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

    def to_np(t):
        """CHW tensor → HWC numpy for display."""
        return t.permute(1, 2, 0).numpy()

    def blend_for_display(t):
        """Blend map CHW → HWC, shift so 0.5 maps to gray."""
        return to_np(t)

    # ── Row 0: Inputs ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(original_np)
    ax.set_title("Original Image", fontsize=11)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(gt_np)
    ax.set_title("Ground Truth (edited)", fontsize=11)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    diff = np.abs(original_np.astype(np.float32) - gt_np.astype(np.float32)) / 255.0
    ax.imshow((diff * 5).clip(0, 1))
    ax.set_title("Pixel Diff (5× amplified)", fontsize=11)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 3])
    mask_np = mask.squeeze().numpy()
    ax.imshow(mask_np, cmap="Reds", vmin=0, vmax=1)
    edit_pct = 100.0 * mask_np.mean()
    ax.set_title(f"Edit Mask (auto-derived, {edit_pct:.1f}% of image)", fontsize=11)
    ax.axis("off")

    # ── Row 1: Blend maps ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(blend_for_display(target_blend))
    ax.set_title("Target Blend Map (GT)", fontsize=11)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(blend_for_display(pred_blend))
    ax.set_title("Predicted Blend Map (simulated)", fontsize=11)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    error = (pred_blend - target_blend).abs()
    im = ax.imshow(to_np(error), cmap="hot", vmin=0, vmax=0.2)
    ax.set_title("Absolute Error |pred − target|", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Deviation from 0.5 for target
    ax = fig.add_subplot(gs[1, 3])
    dev = (target_blend - 0.5).abs().max(dim=0, keepdim=True)[0].squeeze().numpy()
    im = ax.imshow(dev, cmap="hot", vmin=0, vmax=0.15)
    ax.set_title(f"Target deviation from 0.5\n(threshold={0.02} → mask)", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ── Row 2: Per-pixel loss heatmaps ────────────────────────────────
    vmax = max(
        global_loss_map.max().item(),
        masked_loss_map.max().item(),
        unmasked_loss_map.max().item(),
        0.01,
    )

    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(to_np(global_loss_map), cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(
        f"Global L1 (per-pixel)\nloss = {losses['global_l1']:.5f}",
        fontsize=11,
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(to_np(masked_loss_map), cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(
        f"Masked L1 (edit region only)\nloss = {losses['masked_l1']:.5f}",
        fontsize=11,
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(to_np(unmasked_loss_map), cmap="hot", vmin=0, vmax=vmax)
    ax.set_title(
        f"Unmasked L1 (background only)\nloss = {losses['unmasked_l1']:.5f}",
        fontsize=11,
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Overlay: mask boundary on top of error
    ax = fig.add_subplot(gs[2, 3])
    err_np = to_np(error).mean(axis=2)  # grayscale error
    ax.imshow(err_np, cmap="gray", vmin=0, vmax=0.15)
    ax.contour(mask_np, levels=[0.5], colors=["cyan"], linewidths=1.5)
    ax.set_title("Error with mask boundary", fontsize=11)
    ax.axis("off")

    # ── Row 3: Bar chart + explanation ─────────────────────────────────
    ax = fig.add_subplot(gs[3, 0:2])
    names = ["Global L1\n(λ=1.0)", "Masked L1\n(λ=1.0)", "Unmasked L1\n(λ=0.0)"]
    values = [losses["global_l1"], losses["masked_l1"], losses["unmasked_l1"]]
    weighted = [
        1.0 * losses["global_l1"],
        1.0 * losses["masked_l1"],
        0.0 * losses["unmasked_l1"],
    ]
    x = np.arange(len(names))
    bars1 = ax.bar(x - 0.15, values, 0.3, label="Raw loss", color="steelblue")
    bars2 = ax.bar(x + 0.15, weighted, 0.3, label="Weighted (λ × loss)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Loss value")
    ax.set_title("Loss Comparison (raw vs weighted contribution)", fontsize=11)
    ax.legend()
    # Annotate values
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    ax = fig.add_subplot(gs[3, 2:4])
    ax.axis("off")
    total_weighted = sum(weighted)
    explanation = (
        f"How these losses work:\n"
        f"─────────────────────────────────────\n"
        f"1. Global L1:   mean(|pred − target|) over ALL pixels\n"
        f"   → Penalises errors everywhere equally.\n"
        f"   → Dominated by the large neutral (0.5) background.\n\n"
        f"2. Masked L1:   mean(|pred − target|) over EDIT pixels only\n"
        f"   → Focuses on the chin/skin correction region.\n"
        f"   → Normalized by mask area, so small edits aren't diluted.\n\n"
        f"3. Unmasked L1: mean(|pred − target|) over NON-EDIT pixels\n"
        f"   → Ensures background stays neutral (0.5).\n"
        f"   → Currently λ=0.0 (disabled) — the global L1 already\n"
        f"     covers this; enabling it double-counts background.\n\n"
        f"Weighted total: {total_weighted:.5f}\n"
        f"Edit region covers {edit_pct:.1f}% of image"
    )
    ax.text(
        0.05, 0.95, explanation, transform=ax.transAxes, fontsize=10,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle("Blend Map Loss Visualization", fontsize=14, fontweight="bold")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {output_path}")


def compute_effective_weights(mask_np, lambda_g, lambda_m, lambda_u):
    """Compute the effective per-pixel gradient weight for a given lambda config.

    Math:
      Global L1 = sum(|e|) / N         → each pixel contributes  1/N
      Masked L1 = sum(|e|·mask) / M     → each edit pixel: 1/M,  bg: 0
      Unmasked L1 = sum(|e|·(1-mask))/B → each bg pixel:   1/B, edit: 0

    Effective weight per edit pixel   = λ_g/N  +  λ_m/M
    Effective weight per bg pixel     = λ_g/N  +  λ_u/B
    Amplification = edit_weight / bg_weight
    """
    N = float(mask_np.size)            # total pixels (single channel)
    M = float(mask_np.sum())           # edit pixels
    B = N - M                          # background pixels
    M = max(M, 1.0)
    B = max(B, 1.0)

    w_edit = lambda_g / N + lambda_m / M
    w_bg   = lambda_g / N + lambda_u / B
    w_bg   = max(w_bg, 1e-12)
    amplification = w_edit / w_bg

    # Build spatial weight map
    weight_map = np.where(mask_np > 0.5, w_edit, w_bg)
    return weight_map, w_edit, w_bg, amplification, M / N


def sample_edit_fractions(pairs, num_samples, img_size, threshold):
    """Sample a subset of pairs and return the edit-region fraction for each."""
    rng = np.random.RandomState(42)
    indices = rng.choice(len(pairs), size=min(num_samples, len(pairs)), replace=False)
    fractions = []
    for i in indices:
        try:
            img_t, gt_t, _, _ = load_pair_from_paths(pairs[i][0], pairs[i][1], img_size)
            tb = compute_target_blend_map(img_t, gt_t).unsqueeze(0)
            m = generate_blend_mask(tb, threshold=threshold).squeeze().numpy()
            fractions.append(m.mean())
        except Exception:
            continue
    return np.array(fractions)


def make_weightage_figure(mask_np, edit_fractions, output_path):
    """Build the lambda-weightage analysis figure."""
    fig = plt.figure(figsize=(26, 22))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    edit_frac = mask_np.mean()
    N = mask_np.size
    M = max(mask_np.sum(), 1)
    B = N - M

    # ── (0,0) Effective weight map for current config ──────────────
    wmap, w_e, w_b, amp, _ = compute_effective_weights(mask_np, 1.0, 1.0, 0.0)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(wmap, cmap="RdYlBu_r")
    ax.set_title(f"Effective per-pixel weight\nλ_g=1, λ_m=1, λ_u=0", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ── (0,1) Same but with λ_u=0.5 enabled ──────────────────────
    wmap2, w_e2, w_b2, amp2, _ = compute_effective_weights(mask_np, 1.0, 1.0, 0.5)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(wmap2, cmap="RdYlBu_r")
    ax.set_title(f"Effective per-pixel weight\nλ_g=1, λ_m=1, λ_u=0.5", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # ── (0,2) Summary table ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    configs = [
        ("Current: λ_g=1, λ_m=1, λ_u=0",   1.0, 1.0, 0.0),
        ("Alt A:   λ_g=1, λ_m=2, λ_u=0",   1.0, 2.0, 0.0),
        ("Alt B:   λ_g=1, λ_m=1, λ_u=0.5", 1.0, 1.0, 0.5),
        ("Alt C:   λ_g=1, λ_m=0.5, λ_u=0", 1.0, 0.5, 0.0),
        ("Only global: λ_g=1, λ_m=0, λ_u=0", 1.0, 0.0, 0.0),
        ("Only masked: λ_g=0, λ_m=1, λ_u=0", 0.0, 1.0, 0.0),
    ]
    table_text = f"Edit region: {100*edit_frac:.1f}% of image  (N/M = {N/M:.0f})\n"
    table_text += "═" * 55 + "\n"
    table_text += f"{'Config':<38} {'Amp':>6} {'w_edit':>9} {'w_bg':>9}\n"
    table_text += "─" * 55 + "\n"
    for name, lg, lm, lu in configs:
        _, we, wb, a, _ = compute_effective_weights(mask_np, lg, lm, lu)
        table_text += f"{name:<38} {a:>5.1f}× {we:>.2e} {wb:>.2e}\n"
    table_text += "─" * 55 + "\n"
    table_text += (
        "\nAmplification = how many times MORE gradient\n"
        "flows to an edit pixel vs a background pixel.\n"
        "Higher → model focuses more on the edit region."
    )
    ax.text(
        0.02, 0.95, table_text, transform=ax.transAxes, fontsize=9.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    ax.set_title("Lambda Configurations Comparison", fontsize=11)

    # ── (1, 0:2) Lambda_masked sweep ──────────────────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    lm_values = np.linspace(0, 5, 200)
    # Three scenarios for lambda_unmasked
    for lu_val, color, ls in [(0.0, "coral", "-"), (0.5, "steelblue", "--"), (1.0, "green", ":")]:
        amps = []
        for lm in lm_values:
            _, _, _, a, _ = compute_effective_weights(mask_np, 1.0, lm, lu_val)
            amps.append(a)
        ax.plot(lm_values, amps, color=color, ls=ls, lw=2,
                label=f"λ_u={lu_val}")
    ax.axvline(1.0, color="gray", ls="--", alpha=0.5, label="Current λ_m=1.0")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("λ_masked", fontsize=11)
    ax.set_ylabel("Amplification (edit / background)", fontsize=11)
    ax.set_title(
        f"Edit-region amplification vs λ_masked  (edit={100*edit_frac:.1f}%, λ_global=1.0)",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # ── (1, 2) The math explanation ───────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    math_text = (
        "The Math\n"
        "════════════════════════════════════\n\n"
        "Per-pixel gradient weight:\n\n"
        "  Edit pixel:    λ_g/N  +  λ_m/M\n"
        "  BG pixel:      λ_g/N  +  λ_u/B\n\n"
        "Where:\n"
        f"  N = {N:,} (total pixels)\n"
        f"  M = {int(M):,} (edit pixels, {100*edit_frac:.1f}%)\n"
        f"  B = {int(B):,} (background, {100*(1-edit_frac):.1f}%)\n"
        f"  N/M = {N/M:.0f}\n\n"
        "With λ_g=1, λ_m=1, λ_u=0:\n"
        f"  edit weight  = 1/{N} + 1/{int(M)}\n"
        f"  bg weight    = 1/{N}\n"
        f"  amplification = 1 + N/M = {1 + N/M:.0f}×\n\n"
        "→ The model sees each edit pixel\n"
        f"  as {1 + N/M:.0f}× more important than\n"
        "  each background pixel."
    )
    ax.text(
        0.02, 0.95, math_text, transform=ax.transAxes, fontsize=9.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#e8f4fd", alpha=0.8),
    )

    # ── (2, 0:2) Edit-region distribution across dataset ──────────
    if len(edit_fractions) > 0:
        ax = fig.add_subplot(gs[2, 0:2])
        pcts = edit_fractions * 100
        ax.hist(pcts, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(100 * edit_frac, color="red", ls="--", lw=2,
                   label=f"This image ({100*edit_frac:.1f}%)")
        ax.axvline(pcts.mean(), color="orange", ls="--", lw=2,
                   label=f"Dataset mean ({pcts.mean():.1f}%)")
        ax.set_xlabel("Edit region (% of image)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(
            f"Edit Region Size Distribution (n={len(edit_fractions)} samples)",
            fontsize=11,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # ── (2, 2) Amplification distribution ─────────────────────
        ax = fig.add_subplot(gs[2, 2])
        amps_dataset = 1.0 + 1.0 / np.clip(edit_fractions, 1e-6, 1.0)  # 1 + N/M = 1 + 1/frac
        ax.hist(amps_dataset, bins=40, color="coral", edgecolor="white", alpha=0.8)
        ax.axvline(1 + N / M, color="red", ls="--", lw=2,
                   label=f"This image ({1 + N/M:.0f}×)")
        ax.axvline(amps_dataset.mean(), color="orange", ls="--", lw=2,
                   label=f"Mean ({amps_dataset.mean():.0f}×)")
        ax.set_xlabel("Amplification factor (λ_g=1, λ_m=1)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("Amplification Factor Distribution", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    else:
        ax = fig.add_subplot(gs[2, :])
        ax.text(0.5, 0.5, "(dataset sampling skipped — no pairs provided)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.axis("off")

    fig.suptitle(
        "Loss Weightage Analysis — How Lambdas Affect Per-Pixel Importance",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize blend map loss functions")
    # Mode 1: simple directory scanning
    parser.add_argument("--data_dir", default=None,
                        help="Folder containing original/ and edited/ sub-folders")
    # Mode 2: explicit JSON
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--data_json", default=None)
    parser.add_argument("--exclude_source", default="aadan_double_chin")
    # Common
    parser.add_argument("--index", type=int, default=42,
                        help="Index of the image pair to use")
    parser.add_argument("--noise_scale", type=float, default=0.15,
                        help="Scale of simulated prediction noise (0 = perfect prediction)")
    parser.add_argument("--edit_error_scale", type=float, default=0.10,
                        help="Scale of error specifically in the edit region")
    parser.add_argument("--img_size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Blend mask threshold (deviation from 0.5)")
    parser.add_argument("--output", default="loss_visualization.png")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of images to sample for edit-region distribution")
    args = parser.parse_args()

    # Resolve image pair
    if args.data_dir:
        pairs = collect_pairs_from_dirs(args.data_dir)
        print(f"Found {len(pairs)} pairs in {args.data_dir}")
        idx = args.index % len(pairs)
        img_path, gt_path = pairs[idx]
        print(f"Using pair #{idx}: {os.path.basename(img_path)}")
    elif args.data_json and args.data_root:
        with open(args.data_json) as f:
            data_map = json.load(f)
        if args.exclude_source:
            data_map = [d for d in data_map if d.get("source") != args.exclude_source]
        print(f"Dataset: {len(data_map)} entries (after filtering)")
        idx = args.index % len(data_map)
        entry = data_map[idx]
        img_path = os.path.join(args.data_root, entry["original_image"])
        gt_path = os.path.join(args.data_root, entry["edited_image"])
        print(f"Using pair #{idx}: {os.path.basename(entry['original_image'])}")
    else:
        parser.error("Provide either --data_dir or both --data_root and --data_json")

    # Load images
    img_t, gt_t, img_np, gt_np = load_pair_from_paths(img_path, gt_path, args.img_size)

    # Compute target blend map
    target_blend = compute_target_blend_map(img_t, gt_t)  # (3, H, W)

    # Derive edit mask (same logic as CombinedLoss)
    target_4d = target_blend.unsqueeze(0)  # (1, 3, H, W)
    mask = generate_blend_mask(target_4d, threshold=args.threshold)  # (1, 1, H, W)

    # Simulate a predicted blend map
    pred_blend = simulate_prediction(
        target_blend, mask.squeeze(0),
        noise_scale=args.noise_scale,
        edit_error_scale=args.edit_error_scale,
    )
    pred_4d = pred_blend.unsqueeze(0)  # (1, 3, H, W)

    # ── Compute scalar losses (same functions used in training) ──────
    global_l1_val = torch.nn.functional.l1_loss(pred_4d, target_4d).item()
    masked_l1_val = masked_l1_loss(pred_4d, target_4d, mask).item()
    unmasked_l1_val = unmasked_l1_loss(pred_4d, target_4d, mask).item()

    losses = {
        "global_l1": global_l1_val,
        "masked_l1": masked_l1_val,
        "unmasked_l1": unmasked_l1_val,
    }
    print(f"  Global L1:   {global_l1_val:.5f}")
    print(f"  Masked L1:   {masked_l1_val:.5f}")
    print(f"  Unmasked L1: {unmasked_l1_val:.5f}")

    # ── Per-pixel loss maps (for heatmap visualization) ──────────────
    pixel_error = (pred_4d - target_4d).abs()  # (1, 3, H, W)

    # Global: just the raw error
    global_loss_map = pixel_error.squeeze(0)  # (3, H, W)

    # Masked: error × mask (only edit region visible)
    mask_3ch = mask.expand_as(pred_4d)
    masked_loss_map = (pixel_error * mask_3ch).squeeze(0)

    # Unmasked: error × (1 - mask) (only background visible)
    unmasked_loss_map = (pixel_error * (1 - mask_3ch)).squeeze(0)

    # ── Build loss visualization figure ────────────────────────────
    make_figure(
        img_np, gt_np,
        target_blend, pred_blend, mask.squeeze(0),
        global_loss_map, masked_loss_map, unmasked_loss_map,
        losses, args.output,
    )

    # ── Build weightage analysis figure ──────────────────────────
    mask_np = mask.squeeze().numpy()

    # Sample edit-region fractions across the dataset
    pairs = []
    if args.data_dir:
        pairs = collect_pairs_from_dirs(args.data_dir)
    elif args.data_json and args.data_root:
        with open(args.data_json) as f:
            data_map_all = json.load(f)
        if args.exclude_source:
            data_map_all = [d for d in data_map_all if d.get("source") != args.exclude_source]
        pairs = [
            (os.path.join(args.data_root, d["original_image"]),
             os.path.join(args.data_root, d["edited_image"]))
            for d in data_map_all
        ]

    print(f"\nSampling {args.num_samples} images for edit-region distribution...")
    edit_fractions = sample_edit_fractions(
        pairs, args.num_samples, args.img_size, args.threshold,
    )
    if len(edit_fractions) > 0:
        print(f"  Edit region: mean={100*edit_fractions.mean():.1f}%, "
              f"median={100*np.median(edit_fractions):.1f}%, "
              f"range=[{100*edit_fractions.min():.1f}%, {100*edit_fractions.max():.1f}%]")

    weight_output = args.output.replace(".png", "_weightage.png")
    make_weightage_figure(mask_np, edit_fractions, weight_output)


if __name__ == "__main__":
    main()
