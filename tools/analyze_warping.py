#!/usr/bin/env python3
"""
Warping Analysis Tool
=====================
Detects and quantifies geometric warping between original and edited image pairs.

Uses multiple complementary techniques:
  1. Dense Optical Flow (Farneback) — pixel-level displacement field
  2. Feature-based Homography (ORB) — global geometric transform residual
  3. Structural Similarity (SSIM) — perceptual structural change map
  4. Edge straightness analysis — detects curved distortions in originally straight edges
  5. Face-region focused analysis — chin / jawline warping detection via MediaPipe

Outputs:
  - Per-image warping metrics (CSV)
  - Visual diagnostic grids (saved to output dir)
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------------------------
# Constants — classification thresholds (two-factor system)
# ---------------------------------------------------------------------------

# Geometric corroboration indicator thresholds
GEO_REPROJ_ERROR_THRESH = 0.05
GEO_GRID_WARP_MEAN_THRESH = 0.5
GEO_FLOW_MEAN_THRESH = 0.3
GEO_SSIM_THRESH = 0.98
GEO_EDGE_DIFF_RATIO_THRESH = 0.01

# Flow severity tier boundaries (edit-region metrics)
FLOW_NEGLIGIBLE_MEAN = 0.1
FLOW_NEGLIGIBLE_P95 = 0.5
FLOW_NEGLIGIBLE_PCT1 = 2.0
FLOW_MILD_MEAN = 0.5
FLOW_MILD_P95 = 3.0
FLOW_MILD_PCT1 = 10.0
FLOW_MODERATE_MEAN = 1.5
FLOW_MODERATE_PCT1 = 25.0

# Minimum geometric score required per tier
GEO_MODERATE_MIN = 1
GEO_SIGNIFICANT_MIN = 2

# Explicit CSV column order (guards against dict-key reordering)
CSV_FIELDNAMES = [
    "base_name", "img_h", "img_w",
    "edit_region_pct", "edit_flow_mean", "edit_flow_p95", "edit_flow_max",
    "edit_pct_over_1px", "edit_pct_over_2px", "edit_pct_over_5px",
    "flow_mean", "flow_median", "flow_max", "flow_std", "flow_p95", "flow_p99",
    "ssim", "reproj_error", "grid_warp_mean", "grid_warp_max",
    "edge_preservation", "edge_diff_ratio",
    "chin_flow_mean", "chin_flow_max",
    "geometric_score", "warp_level",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def match_image_pairs(original_dir: str, edited_dir: str):
    """
    Match original ↔ edited images by base name (ignoring extension differences
    like .jpg vs .jpeg vs .png).

    Also handles UUID naming convention differences where originals may use
    ``uuid_index.ext`` while edited may use ``uuid.ext_index.ext``.
    """
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def _normalize_key(fname):
        """Canonical key: uuid_index regardless of naming convention."""
        base_no_ext, _ = os.path.splitext(fname)
        # Pattern: uuid.ext_index.ext  (e.g. abc.jpg_0.jpeg)
        m = re.match(
            r'^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.\w+_(\d+)\.\w+$',
            fname,
        )
        if m:
            return f"{m.group(1)}_{m.group(2)}"
        # Fallback: strip extension (covers uuid_index.ext and non-uuid names)
        return base_no_ext

    def build_index(d):
        idx = {}
        for fname in os.listdir(d):
            _, ext = os.path.splitext(fname)
            if ext.lower() in VALID_EXTS:
                key = _normalize_key(fname)
                idx[key] = fname
        return idx

    orig_idx = build_index(original_dir)
    edit_idx = build_index(edited_dir)
    common_bases = sorted(set(orig_idx.keys()) & set(edit_idx.keys()))

    pairs = []
    for base in common_bases:
        pairs.append(
            (
                os.path.join(original_dir, orig_idx[base]),
                os.path.join(edited_dir, edit_idx[base]),
                base,
            )
        )
    return pairs


def load_pair(orig_path: str, edit_path: str, max_dim: int = 512):
    """Load and resize an image pair to the same dimensions.

    If *max_dim* is 0, images are kept at their original resolution (edit is
    resized to match orig if shapes differ).
    """
    orig = cv2.imread(orig_path)
    edit = cv2.imread(edit_path)
    if orig is None:
        raise ValueError(f"Cannot read original: {orig_path}")
    if edit is None:
        raise ValueError(f"Cannot read edited: {edit_path}")

    # Ensure 3-channel BGR (handle grayscale or BGRA inputs)
    if orig.ndim == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    elif orig.shape[2] == 4:
        orig = cv2.cvtColor(orig, cv2.COLOR_BGRA2BGR)
    if edit.ndim == 2:
        edit = cv2.cvtColor(edit, cv2.COLOR_GRAY2BGR)
    elif edit.shape[2] == 4:
        edit = cv2.cvtColor(edit, cv2.COLOR_BGRA2BGR)

    # Make sure both are the same size (edit may differ)
    if orig.shape[:2] != edit.shape[:2]:
        edit = cv2.resize(edit, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)

    if max_dim and max_dim > 0:
        h, w = orig.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            orig = cv2.resize(orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
            edit = cv2.resize(edit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return orig, edit


# ---------------------------------------------------------------------------
# Analysis methods
# ---------------------------------------------------------------------------


def compute_optical_flow(orig_gray: np.ndarray, edit_gray: np.ndarray):
    """
    Dense optical flow (Farneback) between original → edited.
    Returns flow field (H, W, 2) and magnitude image.
    """
    flow = cv2.calcOpticalFlowFarneback(
        orig_gray,
        edit_gray,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=15,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return flow, mag, ang


def compute_flow_metrics(mag: np.ndarray):
    """Summarize flow magnitude into scalar metrics."""
    return {
        "flow_mean": float(np.mean(mag)),
        "flow_median": float(np.median(mag)),
        "flow_max": float(np.max(mag)),
        "flow_std": float(np.std(mag)),
        "flow_p95": float(np.percentile(mag, 95)),
        "flow_p99": float(np.percentile(mag, 99)),
    }


def compute_ssim_map(orig_gray: np.ndarray, edit_gray: np.ndarray):
    """Compute SSIM and return the full SSIM map."""
    min_side = min(orig_gray.shape[:2])
    # win_size must be odd and ≤ image dimension; default 7
    win = min(7, min_side if min_side % 2 == 1 else min_side - 1)
    if win < 3:
        # Image too small for meaningful SSIM
        return 1.0, np.ones_like(orig_gray, dtype=np.float32)
    score, ssim_map = ssim(orig_gray, edit_gray, win_size=win, full=True)
    return float(score), ssim_map.astype(np.float32)


def feature_based_warp_analysis(orig_gray: np.ndarray, edit_gray: np.ndarray):
    """
    ORB feature matching + homography estimation.
    Returns homography matrix and warp residual (mean reprojection error on inliers).
    Also returns per-match displacement vectors for visualization.
    """
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(orig_gray, None)
    kp2, des2 = orb.detectAndCompute(edit_gray, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, None, np.inf, [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 10:
        return None, None, np.inf, [], []

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        return None, None, np.inf, pts1, pts2

    inlier_mask = mask.ravel().astype(bool)
    inlier_pts1 = pts1[inlier_mask]
    inlier_pts2 = pts2[inlier_mask]

    # Reprojection error
    pts1_h = np.hstack([inlier_pts1, np.ones((len(inlier_pts1), 1))])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        projected = (H @ pts1_h.T).T
    w_coords = projected[:, 2:3]
    # Guard against degenerate homography producing nan via division by zero
    valid = np.abs(w_coords.ravel()) > 1e-10
    if valid.sum() < 4:
        return H, inlier_mask, np.inf, inlier_pts1, inlier_pts2 - inlier_pts1
    projected = projected[valid, :2] / w_coords[valid]
    reproj_err = np.sqrt(np.sum((projected - inlier_pts2[valid]) ** 2, axis=1))
    mean_err = float(np.mean(reproj_err))
    if np.isnan(mean_err) or np.isinf(mean_err):
        mean_err = np.inf

    # Displacement vectors (original feature locations → edited locations)
    displacements = inlier_pts2 - inlier_pts1

    return H, inlier_mask, mean_err, inlier_pts1, displacements


def compute_local_warp_field(
    orig_gray: np.ndarray, edit_gray: np.ndarray, grid_size: int = 16
):
    """
    Divide the image into a grid and compute block-wise displacement using
    template matching. Returns a grid of displacement vectors.
    """
    h, w = orig_gray.shape
    bh, bw = h // grid_size, w // grid_size
    search_margin = max(bh // 2, bw // 2, 8)

    displacements = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    confidence = np.zeros((grid_size, grid_size), dtype=np.float32)

    for gy in range(grid_size):
        for gx in range(grid_size):
            # Template from original
            y0, x0 = gy * bh, gx * bw
            template = orig_gray[y0 : y0 + bh, x0 : x0 + bw]
            if template.size == 0:
                continue

            # Search region in edited (expanded)
            sy0 = max(0, y0 - search_margin)
            sx0 = max(0, x0 - search_margin)
            sy1 = min(h, y0 + bh + search_margin)
            sx1 = min(w, x0 + bw + search_margin)
            search = edit_gray[sy0:sy1, sx0:sx1]

            if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
                continue

            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # Displacement = found location − expected location
            found_x = sx0 + max_loc[0]
            found_y = sy0 + max_loc[1]
            displacements[gy, gx, 0] = found_x - x0  # dx
            displacements[gy, gx, 1] = found_y - y0  # dy
            confidence[gy, gx] = max_val

    return displacements, confidence


def compute_edge_distortion(orig_gray: np.ndarray, edit_gray: np.ndarray):
    """
    Detect edges in both images and compute how much the edge structure differs.
    High difference near straight edges → warping.
    """
    edges_orig = cv2.Canny(orig_gray, 50, 150)
    edges_edit = cv2.Canny(edit_gray, 50, 150)

    # Dilate edges slightly for robust comparison
    kernel = np.ones((3, 3), np.uint8)
    edges_orig_d = cv2.dilate(edges_orig, kernel, iterations=1)
    edges_edit_d = cv2.dilate(edges_edit, kernel, iterations=1)

    # Edge preservation: what fraction of original edges are present in edited
    if edges_orig_d.sum() == 0:
        preservation = 1.0
    else:
        preservation = float(
            np.sum(edges_edit_d[edges_orig > 0] > 0) / np.sum(edges_orig > 0)
        )

    # XOR of edges → displaced / new / removed edges
    edge_diff = cv2.bitwise_xor(edges_orig, edges_edit)
    edge_diff_ratio = float(np.sum(edge_diff > 0)) / max(edge_diff.size, 1)

    return edges_orig, edges_edit, edge_diff, preservation, edge_diff_ratio


def compute_chin_region_flow(
    flow_mag: np.ndarray, orig_bgr: np.ndarray
):
    """
    Focus on the lower-face / chin region (bottom 40% of image, middle 60% width)
    since that's where double-chin editing warping is most likely.
    """
    h, w = flow_mag.shape
    y_start = int(h * 0.6)
    x_start = int(w * 0.2)
    x_end = int(w * 0.8)

    chin_flow = flow_mag[y_start:, x_start:x_end]
    if chin_flow.size == 0:
        return 0.0, 0.0, None

    # Create mask for visualization
    mask = np.zeros_like(flow_mag)
    mask[y_start:, x_start:x_end] = 1.0

    return float(np.mean(chin_flow)), float(np.max(chin_flow)), mask


def compute_edit_region_flow(
    orig_gray: np.ndarray, edit_gray: np.ndarray, flow_mag: np.ndarray,
    diff_threshold: int = 3, dilate_kernel: int = 15, dilate_iters: int = 3,
):
    """
    Compute optical-flow metrics **only within the edited region**.

    The edited region is detected by pixel-difference thresholding followed by
    morphological dilation (to capture the boundary displacement zone).

    This avoids the whole-image averaging problem where a small, heavily-warped
    chin edit gets diluted by a large unchanged background.

    Returns
    -------
    metrics : dict
        edit_region_pct, edit_flow_mean, edit_flow_p95, edit_flow_max,
        edit_pct_over_1px, edit_pct_over_2px, edit_pct_over_5px
    edit_mask : np.ndarray  (H, W) uint8, 0/255
        Visualization-ready mask of the detected edit region.
    """
    diff = cv2.absdiff(orig_gray, edit_gray)
    edit_mask = (diff > diff_threshold).astype(np.uint8)
    kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
    edit_mask = cv2.dilate(edit_mask, kernel, iterations=dilate_iters)

    n_edit = int(np.count_nonzero(edit_mask))
    pct_edit = 100.0 * n_edit / max(edit_mask.size, 1)

    edited_px = flow_mag[edit_mask > 0]

    if edited_px.size > 10:
        metrics = {
            "edit_region_pct": round(pct_edit, 2),
            "edit_flow_mean": round(float(edited_px.mean()), 3),
            "edit_flow_p95": round(float(np.percentile(edited_px, 95)), 3),
            "edit_flow_max": round(float(edited_px.max()), 1),
            "edit_pct_over_1px": round(
                100.0 * np.count_nonzero(edited_px > 1.0) / edited_px.size, 2
            ),
            "edit_pct_over_2px": round(
                100.0 * np.count_nonzero(edited_px > 2.0) / edited_px.size, 2
            ),
            "edit_pct_over_5px": round(
                100.0 * np.count_nonzero(edited_px > 5.0) / edited_px.size, 2
            ),
        }
    else:
        metrics = {
            "edit_region_pct": round(pct_edit, 2),
            "edit_flow_mean": 0.0,
            "edit_flow_p95": 0.0,
            "edit_flow_max": 0.0,
            "edit_pct_over_1px": 0.0,
            "edit_pct_over_2px": 0.0,
            "edit_pct_over_5px": 0.0,
        }

    return metrics, (edit_mask * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def create_diagnostic_figure(
    orig_bgr,
    edit_bgr,
    flow,
    flow_mag,
    ssim_map,
    grid_disp,
    grid_conf,
    edges_orig,
    edges_edit,
    edge_diff,
    chin_mask,
    metrics,
    base_name,
):
    """Create a comprehensive diagnostic figure."""
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.25)

    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    edit_rgb = cv2.cvtColor(edit_bgr, cv2.COLOR_BGR2RGB)

    # Row 1: Original | Edited | Pixel Difference | Absolute Diff Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_rgb)
    ax1.set_title("Original", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(edit_rgb)
    ax2.set_title("Edited", fontsize=10)
    ax2.axis("off")

    diff = cv2.absdiff(orig_bgr, edit_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(diff_gray, cmap="hot")
    ax3.set_title("Pixel Difference", fontsize=10)
    ax3.axis("off")

    # Amplified difference
    ax4 = fig.add_subplot(gs[0, 3])
    diff_amp = np.clip(diff_gray.astype(np.float32) * 5, 0, 255).astype(np.uint8)
    ax4.imshow(diff_amp, cmap="hot")
    ax4.set_title("Amplified Diff (5×)", fontsize=10)
    ax4.axis("off")

    # Row 2: Optical Flow Magnitude | Flow X | Flow Y | Flow HSV
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(flow_mag, cmap="hot", vmin=0, vmax=max(np.percentile(flow_mag, 99), 1))
    ax5.set_title(f"Flow Magnitude (mean={metrics['flow_mean']:.2f})", fontsize=10)
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    ax6 = fig.add_subplot(gs[1, 1])
    vmax_xy = max(np.percentile(np.abs(flow[..., 0]), 99), 1)
    im6 = ax6.imshow(flow[..., 0], cmap="RdBu_r", vmin=-vmax_xy, vmax=vmax_xy)
    ax6.set_title("Flow X (horizontal)", fontsize=10)
    ax6.axis("off")
    plt.colorbar(im6, ax=ax6, fraction=0.046)

    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(flow[..., 1], cmap="RdBu_r", vmin=-vmax_xy, vmax=vmax_xy)
    ax7.set_title("Flow Y (vertical)", fontsize=10)
    ax7.axis("off")
    plt.colorbar(im7, ax=ax7, fraction=0.046)

    # Flow as HSV color wheel
    ax8 = fig.add_subplot(gs[1, 3])
    hsv = np.zeros((*flow_mag.shape, 3), dtype=np.uint8)
    _, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(flow_mag, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ax8.imshow(flow_rgb)
    ax8.set_title("Flow Direction (HSV)", fontsize=10)
    ax8.axis("off")

    # Row 3: SSIM Map | Grid Warp Vectors | Edge Diff | Chin Region Flow
    ax9 = fig.add_subplot(gs[2, 0])
    im9 = ax9.imshow(1.0 - ssim_map, cmap="hot", vmin=0, vmax=0.5)
    ax9.set_title(f"1 − SSIM (score={metrics['ssim']:.3f})", fontsize=10)
    ax9.axis("off")
    plt.colorbar(im9, ax=ax9, fraction=0.046)

    # Grid displacement vectors
    ax10 = fig.add_subplot(gs[2, 1])
    ax10.imshow(orig_rgb, alpha=0.5)
    h, w = orig_bgr.shape[:2]
    gy, gx = grid_disp.shape[:2]
    bh, bw = h // gy, w // gx
    for yi in range(gy):
        for xi in range(gx):
            cx = xi * bw + bw // 2
            cy = yi * bh + bh // 2
            dx, dy = grid_disp[yi, xi]
            mag_v = np.sqrt(dx**2 + dy**2)
            color = "red" if mag_v > 2.0 else "green"
            ax10.arrow(
                cx, cy, dx * 3, dy * 3,
                head_width=3, head_length=1.5,
                fc=color, ec=color, alpha=0.7,
            )
    ax10.set_title("Block Displacement Vectors", fontsize=10)
    ax10.axis("off")

    ax11 = fig.add_subplot(gs[2, 2])
    ax11.imshow(edge_diff, cmap="hot")
    ax11.set_title(
        f"Edge Diff (preserve={metrics['edge_preservation']:.2f})", fontsize=10
    )
    ax11.axis("off")

    # Chin region focused
    ax12 = fig.add_subplot(gs[2, 3])
    if chin_mask is not None:
        chin_vis = flow_mag * chin_mask
        im12 = ax12.imshow(chin_vis, cmap="hot", vmin=0, vmax=max(np.percentile(flow_mag, 99), 1))
        ax12.set_title(
            f"Chin Region Flow (mean={metrics['chin_flow_mean']:.2f})", fontsize=10
        )
        plt.colorbar(im12, ax=ax12, fraction=0.046)
    else:
        ax12.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax12.transAxes)
        ax12.set_title("Chin Region Flow", fontsize=10)
    ax12.axis("off")

    # Row 4: Overlay with warping vectors | Warp magnitude histogram | Summary text
    ax13 = fig.add_subplot(gs[3, 0:2])
    overlay = orig_rgb.copy().astype(np.float32)
    flow_norm = flow_mag / max(flow_mag.max(), 1e-5)
    heatmap = plt.cm.hot(flow_norm)[..., :3] * 255
    blended = (0.6 * overlay + 0.4 * heatmap).clip(0, 255).astype(np.uint8)
    ax13.imshow(blended)
    ax13.set_title("Warp Heatmap Overlay on Original", fontsize=10)
    ax13.axis("off")

    ax14 = fig.add_subplot(gs[3, 2])
    ax14.hist(flow_mag.ravel(), bins=100, color="steelblue", edgecolor="none", log=True)
    ax14.axvline(metrics["flow_mean"], color="red", linestyle="--", label=f"mean={metrics['flow_mean']:.2f}")
    ax14.axvline(metrics["flow_p95"], color="orange", linestyle="--", label=f"p95={metrics['flow_p95']:.2f}")
    ax14.set_xlabel("Flow Magnitude (pixels)")
    ax14.set_ylabel("Count (log)")
    ax14.set_title("Flow Magnitude Distribution", fontsize=10)
    ax14.legend(fontsize=8)

    # Summary text
    ax15 = fig.add_subplot(gs[3, 3])
    ax15.axis("off")
    warp_level = classify_warping(metrics)
    summary = (
        f"Image: {base_name}\n"
        f"─────────────────────\n"
        f"Warping Level: {warp_level}\n\n"
        f"── Edit Region (full-res) ──\n"
        f"Region:    {metrics.get('edit_region_pct', 0):.1f}% of image\n"
        f"Flow mean: {metrics.get('edit_flow_mean', 0):.3f} px\n"
        f"Flow p95:  {metrics.get('edit_flow_p95', 0):.3f} px\n"
        f">1px:      {metrics.get('edit_pct_over_1px', 0):.1f}%\n"
        f">2px:      {metrics.get('edit_pct_over_2px', 0):.1f}%\n"
        f">5px:      {metrics.get('edit_pct_over_5px', 0):.1f}%\n\n"
        f"── Whole Image (thumb) ──\n"
        f"Flow mean: {metrics['flow_mean']:.3f} px\n"
        f"SSIM:      {metrics['ssim']:.4f}\n"
    )
    ax15.text(
        0.05, 0.95, summary, transform=ax15.transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(f"Warping Analysis — {base_name}", fontsize=14, fontweight="bold")
    return fig


def compute_geometric_score(metrics: dict) -> int:
    """Count how many structure-based indicators confirm geometric warping.

    Optical flow in the edit region reacts to *any* pixel-intensity change
    (shadow, colour grading, brightness) — not just spatial displacement.
    These five checks are largely **intensity-invariant** and only fire when
    the geometry genuinely shifts:

      1. reproj_error   — ORB features spatially displaced
      2. grid_warp_mean — template-matching blocks actually moved
      3. flow_mean      — whole-image flow elevated (not just edit region)
      4. ssim           — structural similarity dropped beyond brightness
      5. edge_diff_ratio— edges distorted, not just darkened / lightened

    Returns 0–5 (higher = stronger geometric evidence).
    """
    reproj = metrics.get("reproj_error", 0)
    # Sentinel -1 means homography failed → treat as "no evidence"
    reproj_fires = reproj >= 0 and reproj > GEO_REPROJ_ERROR_THRESH

    checks = [
        reproj_fires,
        metrics.get("grid_warp_mean", 0) > GEO_GRID_WARP_MEAN_THRESH,
        metrics.get("flow_mean", 0) > GEO_FLOW_MEAN_THRESH,
        metrics.get("ssim", 1.0) < GEO_SSIM_THRESH,
        metrics.get("edge_diff_ratio", 0) > GEO_EDGE_DIFF_RATIO_THRESH,
    ]
    return int(sum(checks))


def classify_warping(metrics: dict) -> str:
    """Classify warping severity using **two factors**:

    1. *Flow severity* — edit-region optical-flow statistics capture the
       magnitude of pixel displacement inside the edited area.
    2. *Geometric corroboration* — structure-based indicators that are
       robust to shadow / brightness changes confirm whether the flow is
       caused by real spatial warping or just intensity edits.

    For the higher severity tiers (MODERATE / SIGNIFICANT) the flow signal
    must be backed by at least 1–2 geometric indicators; otherwise the
    level is downgraded.
    """
    er_mean = metrics.get("edit_flow_mean", 0.0)
    er_p95 = metrics.get("edit_flow_p95", 0.0)
    er_pct1 = metrics.get("edit_pct_over_1px", 0.0)
    geo = metrics.get("geometric_score")
    if geo is None:
        geo = compute_geometric_score(metrics)

    # --- Tier 1: clearly negligible flow — no corroboration needed ----------
    if er_mean < FLOW_NEGLIGIBLE_MEAN and er_p95 < FLOW_NEGLIGIBLE_P95 and er_pct1 < FLOW_NEGLIGIBLE_PCT1:
        return "✅ NONE / NEGLIGIBLE"

    # --- Tier 2: mild flow — no corroboration needed ------------------------
    if er_mean < FLOW_MILD_MEAN and er_p95 < FLOW_MILD_P95 and er_pct1 < FLOW_MILD_PCT1:
        return "⚠️  MILD"

    # --- Tier 3: moderate flow — require ≥1 geometric indicator -------------
    if er_mean < FLOW_MODERATE_MEAN and er_pct1 < FLOW_MODERATE_PCT1:
        if geo >= GEO_MODERATE_MIN:
            return "🔶 MODERATE"
        return "⚠️  MILD"            # downgrade: likely shadow / brightness

    # --- Tier 4: high flow — require ≥2 geometric indicators ----------------
    if geo >= GEO_SIGNIFICANT_MIN:
        return "🔴 SIGNIFICANT"
    if geo >= GEO_MODERATE_MIN:
        return "🔶 MODERATE"          # some evidence, not conclusive
    return "⚠️  MILD"                 # no geometric evidence → shadow only


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def analyze_pair(orig_path: str, edit_path: str, base_name: str, output_dir: Optional[str], max_dim: int = 512):
    """Run full warping analysis on a single image pair.

    The **edit-region** analysis always runs at full resolution so that
    localized warping (e.g. chin area) is not diluted by the unchanged
    background.  The traditional whole-image metrics are still computed
    at *max_dim* for speed and backward-compatibility.
    """

    # --- Full-resolution edit-region analysis --------------------------------
    orig_full, edit_full = load_pair(orig_path, edit_path, max_dim=0)
    orig_full_gray = cv2.cvtColor(orig_full, cv2.COLOR_BGR2GRAY)
    edit_full_gray = cv2.cvtColor(edit_full, cv2.COLOR_BGR2GRAY)

    _, flow_mag_full, _ = compute_optical_flow(orig_full_gray, edit_full_gray)
    er_metrics, edit_mask_full = compute_edit_region_flow(
        orig_full_gray, edit_full_gray, flow_mag_full,
    )

    # --- Thumbnail whole-image analysis (for vis & legacy metrics) ----------
    orig, edit = load_pair(orig_path, edit_path, max_dim=max_dim)
    orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    edit_gray = cv2.cvtColor(edit, cv2.COLOR_BGR2GRAY)

    # 1. Optical flow (thumbnail)
    flow, flow_mag, flow_ang = compute_optical_flow(orig_gray, edit_gray)
    flow_metrics = compute_flow_metrics(flow_mag)

    # 2. SSIM
    ssim_score, ssim_map = compute_ssim_map(orig_gray, edit_gray)

    # 3. Feature-based
    H, inlier_mask, reproj_err, feat_pts, feat_disp = feature_based_warp_analysis(
        orig_gray, edit_gray
    )

    # 4. Grid-based displacement
    grid_disp, grid_conf = compute_local_warp_field(orig_gray, edit_gray, grid_size=16)
    grid_mag = np.sqrt(grid_disp[..., 0] ** 2 + grid_disp[..., 1] ** 2)

    # 5. Edge distortion
    edges_orig, edges_edit, edge_diff, edge_pres, edge_diff_ratio = (
        compute_edge_distortion(orig_gray, edit_gray)
    )

    # 6. Chin-region analysis (thumbnail)
    chin_flow_mean, chin_flow_max, chin_mask = compute_chin_region_flow(flow_mag, orig)

    metrics = {
        "base_name": base_name,
        "img_h": orig_full.shape[0],
        "img_w": orig_full.shape[1],
        # --- edit-region metrics (full-res, PRIMARY for classification) ---
        **er_metrics,
        # --- whole-image metrics (thumbnail, kept for reference) ---
        **flow_metrics,
        "ssim": ssim_score,
        "reproj_error": reproj_err if np.isfinite(reproj_err) else -1.0,
        "grid_warp_mean": float(np.mean(grid_mag)),
        "grid_warp_max": float(np.max(grid_mag)),
        "edge_preservation": edge_pres,
        "edge_diff_ratio": edge_diff_ratio,
        "chin_flow_mean": chin_flow_mean,
        "chin_flow_max": chin_flow_max,
    }

    metrics["geometric_score"] = compute_geometric_score(metrics)
    warp_level = classify_warping(metrics)
    metrics["warp_level"] = warp_level

    # Save visualization
    if output_dir:
        fig = create_diagnostic_figure(
            orig, edit, flow, flow_mag, ssim_map,
            grid_disp, grid_conf,
            edges_orig, edges_edit, edge_diff,
            chin_mask, metrics, base_name,
        )
        try:
            fig.savefig(
                os.path.join(output_dir, f"{base_name}_warp_analysis.png"),
                dpi=120, bbox_inches="tight",
            )
        finally:
            plt.close(fig)

    return metrics


def _worker_init():
    """Initializer for each pool worker — limit OpenCV threads to avoid over-subscription."""
    cv2.setNumThreads(1)
    import matplotlib
    matplotlib.use("Agg")


def _analyze_pair_worker(args_tuple):
    """Picklable wrapper for :func:`analyze_pair` used by multiprocessing Pool."""
    idx, total, orig_path, edit_path, base_name, output_dir, max_dim = args_tuple
    try:
        metrics = analyze_pair(orig_path, edit_path, base_name, output_dir, max_dim=max_dim)
        return {"status": "ok", "idx": idx, "total": total, "metrics": metrics, "base_name": base_name}
    except Exception as e:
        return {"status": "error", "idx": idx, "total": total, "error": str(e),
                "traceback": traceback.format_exc(), "base_name": base_name}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze warping between original and edited image pairs."
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        required=True,
        help="Directory containing original images",
    )
    parser.add_argument(
        "--edited_dir",
        type=str,
        required=True,
        help="Directory containing edited images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="warping_analysis_output",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=50,
        help="Max number of pairs to analyze (0 = all)",
    )
    parser.add_argument(
        "--max_dim",
        type=int,
        default=512,
        help="Max dimension for resizing images",
    )
    parser.add_argument(
        "--no_visuals",
        action="store_true",
        default=False,
        help="Skip saving visualizations (faster)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling pairs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel worker processes (0 = auto based on CPU count, 1 = sequential)",
    )
    args = parser.parse_args()

    # --- Input validation ---------------------------------------------------
    if not os.path.isdir(args.original_dir):
        print(f"ERROR: original directory not found: {args.original_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.edited_dir):
        print(f"ERROR: edited directory not found: {args.edited_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations") if not args.no_visuals else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"  WARPING ANALYSIS")
    print(f"  Original: {args.original_dir}")
    print(f"  Edited:   {args.edited_dir}")
    print(f"{'='*70}")

    pairs = match_image_pairs(args.original_dir, args.edited_dir)
    print(f"\nFound {len(pairs)} matched image pairs.")

    if args.max_pairs > 0 and len(pairs) > args.max_pairs:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(pairs), args.max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(indices)]
        print(f"Randomly sampled {args.max_pairs} pairs for analysis.")

    # Decide worker count
    n_workers = args.workers if args.workers > 0 else min(os.cpu_count() or 4, 8)
    use_mp = n_workers > 1 and len(pairs) > 1

    all_metrics = []
    errors = 0
    t0 = time.time()

    n_done = 0

    def _log_result(result):
        """Pretty-print a single result dict."""
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
                f"er_mean={m['edit_flow_mean']:.2f}  "
                f"er_p95={m['edit_flow_p95']:.2f}  "
                f">1px={m['edit_pct_over_1px']:.1f}%  "
                f"region={m['edit_region_pct']:.1f}%  "
                f"geo={m['geometric_score']}/5  "
                f"{m['warp_level']}  "
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
            print(f"Using {n_workers} worker processes.")
            work_items = [
                (i, len(pairs), op, ep, bn, vis_dir, args.max_dim)
                for i, (op, ep, bn) in enumerate(pairs)
            ]
            with multiprocessing.Pool(processes=n_workers, initializer=_worker_init) as pool:
                for result in pool.imap_unordered(_analyze_pair_worker, work_items, chunksize=4):
                    _log_result(result)
        else:
            print("Running sequentially (workers=1).")
            for i, (orig_path, edit_path, base_name) in enumerate(pairs):
                result = _analyze_pair_worker((i, len(pairs), orig_path, edit_path, base_name, vis_dir, args.max_dim))
                _log_result(result)
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted! Saving {len(all_metrics)} partial results …")

    wall_time = time.time() - t0
    print(f"\nProcessed {len(all_metrics)} pairs in {wall_time:.1f}s ({len(all_metrics)/max(wall_time,1):.1f} img/s), {errors} errors.")

    if not all_metrics:
        print("\nNo pairs analyzed successfully.")
        return

    # Sort by base_name for deterministic CSV output regardless of completion order
    all_metrics.sort(key=lambda m: m["base_name"])

    # Save CSV
    csv_path = os.path.join(args.output_dir, "warping_metrics.csv")
    fieldnames = CSV_FIELDNAMES
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nMetrics saved to: {csv_path}")

    # Aggregate statistics
    print(f"\n{'='*70}")
    print(f"  AGGREGATE STATISTICS ({len(all_metrics)} images)")
    print(f"{'='*70}")

    er_means = [m["edit_flow_mean"] for m in all_metrics]
    er_p95s = [m["edit_flow_p95"] for m in all_metrics]
    er_pct1s = [m["edit_pct_over_1px"] for m in all_metrics]
    er_pct5s = [m["edit_pct_over_5px"] for m in all_metrics]
    er_regions = [m["edit_region_pct"] for m in all_metrics]
    flow_means = [m["flow_mean"] for m in all_metrics]
    ssims = [m["ssim"] for m in all_metrics]

    for name, vals in [
        ("Edit-Rgn Flow Mean (px)", er_means),
        ("Edit-Rgn Flow P95 (px)", er_p95s),
        ("Edit-Rgn >1px (%)", er_pct1s),
        ("Edit-Rgn >5px (%)", er_pct5s),
        ("Edit Region Size (%)", er_regions),
        ("Whole-Img Flow Mean (px)", flow_means),
        ("SSIM", ssims),
    ]:
        arr = np.array(vals)
        print(
            f"  {name:27s}  "
            f"mean={np.mean(arr):.3f}  "
            f"median={np.median(arr):.3f}  "
            f"std={np.std(arr):.3f}  "
            f"min={np.min(arr):.3f}  "
            f"max={np.max(arr):.3f}"
        )

    # Classification distribution
    levels = [m["warp_level"] for m in all_metrics]
    print(f"\n  Warping Classification Distribution:")
    for level_name in ["✅ NONE / NEGLIGIBLE", "⚠️  MILD", "🔶 MODERATE", "🔴 SIGNIFICANT"]:
        count = levels.count(level_name)
        pct = 100.0 * count / len(levels)
        bar = "█" * int(pct / 2)
        print(f"    {level_name:25s}  {count:4d} ({pct:5.1f}%)  {bar}")

    # Geometric-score distribution
    geo_scores = [m.get("geometric_score", 0) for m in all_metrics]
    print(f"\n  Geometric Corroboration Score (0–5):")
    for s in range(6):
        cnt = geo_scores.count(s)
        if cnt > 0:
            pct_g = 100.0 * cnt / len(geo_scores)
            bar_g = "█" * int(pct_g / 2)
            print(f"    score {s}:  {cnt:4d} ({pct_g:5.1f}%)  {bar_g}")

    # Find worst cases
    print(f"\n  Top 10 most warped images (by edit-region flow + geometric score):")
    sorted_by_er = sorted(
        all_metrics,
        key=lambda m: (-m.get("geometric_score", 0), -m["edit_flow_mean"]),
    )
    for j, m in enumerate(sorted_by_er[:10]):
        print(
            f"    {j+1:2d}. {m['base_name']:50s}  "
            f"er_mean={m['edit_flow_mean']:.2f}  "
            f"er_p95={m['edit_flow_p95']:.2f}  "
            f">1px={m['edit_pct_over_1px']:.1f}%  "
            f"geo={m.get('geometric_score', 0)}/5  "
            f"{m['warp_level']}"
        )

    # Create aggregate visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    try:
        axes[0, 0].hist(er_means, bins=50, color="steelblue", edgecolor="none")
        axes[0, 0].set_xlabel("Edit-Region Mean Flow (px)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Edit-Region Mean Flow")
        axes[0, 0].axvline(np.mean(er_means), color="red", linestyle="--", label=f"mean={np.mean(er_means):.2f}")
        axes[0, 0].legend()

        axes[0, 1].hist(er_p95s, bins=50, color="darkorange", edgecolor="none")
        axes[0, 1].set_xlabel("Edit-Region P95 Flow (px)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Edit-Region P95 Flow")

        axes[0, 2].hist(er_pct1s, bins=50, color="coral", edgecolor="none")
        axes[0, 2].set_xlabel("% of edit pixels > 1px displacement")
        axes[0, 2].set_ylabel("Count")
        axes[0, 2].set_title("Edit-Region % Over 1px")

        axes[1, 0].hist(er_regions, bins=50, color="seagreen", edgecolor="none")
        axes[1, 0].set_xlabel("Edit Region Size (% of image)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Edit Region Size Distribution")

        axes[1, 1].hist(ssims, bins=50, color="mediumpurple", edgecolor="none")
        axes[1, 1].set_xlabel("SSIM Score")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Distribution of SSIM")

        axes[1, 2].scatter(er_means, ssims, alpha=0.3, s=10, color="purple")
        axes[1, 2].set_xlabel("Edit-Region Mean Flow (px)")
        axes[1, 2].set_ylabel("SSIM")
        axes[1, 2].set_title("Edit-Region Flow vs SSIM")

        fig.suptitle(f"Aggregate Warping Analysis ({len(all_metrics)} images)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, "aggregate_warping_stats.png"), dpi=150)
        print(f"\nAggregate plot saved to: {os.path.join(args.output_dir, 'aggregate_warping_stats.png')}")
    finally:
        plt.close(fig)
    print(f"\nDone! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
