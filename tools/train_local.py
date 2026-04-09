#!/usr/bin/env python3
"""
Local (single-device) training script for the double chin blend-map model.

Works on MPS (Apple Silicon), CUDA (single GPU), or CPU — no DDP required.
Designed to be a drop-in replacement for train_blend_map.py when training
on a single machine without distributed infrastructure.

Usage:
    python tools/train_local.py                          # uses blend_map.yaml
    python tools/train_local.py --config local.yaml      # custom config
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
import time

import numpy as np
try:
    import mlflow
except ImportError:
    mlflow = None
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from tqdm import tqdm

# ── Project imports ──
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.unet import BaseUNetHalf, BaseUNetHalfLite
from src.losses.losses import CombinedLoss
from src.blend.blend_map import apply_blend_formula
from src.utils.utils_blend import load_checkpoint, save_visualization_batch, compute_metrics, save_full_checkpoint, load_full_checkpoint
from src.data.dataset import BlendMapDataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms may be slower but ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(requested: str = "auto") -> torch.device:
    """Pick the best available device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_data_loaders(args: dict, test: bool = False):
    """Create train/val DataLoaders (no DDP, no DistributedSampler)."""
    dataset = BlendMapDataset(
        args["data_root"],
        args["data_json"],
        transform=ToTensor(),
        resize_dim=(args["img_size"], args["img_size"]),
        test=test,
    )

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    split_gen = torch.Generator().manual_seed(args.get("seed", 42))
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=split_gen)

    common = dict(
        batch_size=args.get("batch_size", 2),
        num_workers=args.get("num_workers", 0),
        pin_memory=args.get("pin_memory", False),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train / Validate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: CombinedLoss,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> dict:
    model.train()
    running = {k: 0.0 for k in ("total_loss", "blend_map_loss", "image_mse_loss", "perc_loss", "tv_loss")}

    bar = tqdm(loader, desc=f"Train [{epoch+1}/{num_epochs}]", leave=False)
    for batch in bar:
        images = batch["image"].to(device)
        target_bm = batch["blend_map"].to(device)
        gt = batch["gt"].to(device)

        optimizer.zero_grad()
        pred_bm = model(images)
        retouched = apply_blend_formula(images, pred_bm)
        loss, losses = criterion(pred_bm, target_bm, retouched, gt)
        loss.backward()
        optimizer.step()

        for k in running:
            running[k] += losses.get(k, 0.0)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    save_dir: str | None = None,
) -> dict:
    model.eval()
    running = {k: 0.0 for k in ("total_loss", "blend_map_loss", "image_mse_loss", "perc_loss", "tv_loss")}
    running_psnr = 0.0

    bar = tqdm(loader, desc=f"Val   [{epoch+1}/{num_epochs}]", leave=False)
    for batch_idx, batch in enumerate(bar):
        images = batch["image"].to(device)
        target_bm = batch["blend_map"].to(device)
        gt = batch["gt"].to(device)

        pred_bm = model(images)
        retouched = apply_blend_formula(images, pred_bm)
        loss, losses = criterion(pred_bm, target_bm, retouched, gt)
        metrics = compute_metrics(retouched, gt)

        for k in running:
            running[k] += losses.get(k, 0.0)
        running_psnr += metrics.get("psnr", 0.0)
        bar.set_postfix(loss=f"{loss.item():.4f}")

        # Save visualizations for first few batches
        if save_dir and batch_idx < 3:
            vis_dir = os.path.join(save_dir, "results", f"epoch_{epoch+1}")
            os.makedirs(vis_dir, exist_ok=True)
            batch_vis = {
                "image": images,
                "blend_map": target_bm,
                "gt": gt,
                "filename": batch["filename"],
            }
            save_visualization_batch(batch_vis, pred_bm, vis_dir, prefix=f"val_{batch_idx}", max_samples=4)

    n = len(loader)
    result = {k: v / n for k, v in running.items()}
    result["psnr"] = running_psnr / n
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: dict):
    seed = args.get("seed", 42)
    seed_everything(seed)
    print(f"Seed: {seed}")

    device = get_device(args.get("device", "auto"))
    print(f"Device: {device}")

    # ── Data ──
    test_mode = args.get("test", False)
    train_loader, val_loader = create_data_loaders(args, test=test_mode)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ── Model ──
    ModelClass = BaseUNetHalfLite if args.get('model_variant', 'default') == 'lite' else BaseUNetHalf
    model = ModelClass(
        n_channels=3, n_classes=3,
        last_layer_activation=args.get('last_layer_activation', 'sigmoid'),
        blend_scale=args.get('blend_scale', 0.5),
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")

    if args.get("pretrained_path") and os.path.isfile(args["pretrained_path"]):
        print(f"Loading pretrained: {args['pretrained_path']}")
        model = load_checkpoint(model, args["pretrained_path"])
        model = model.to(device)

    # ── Loss / Optim / Scheduler ──
    criterion = CombinedLoss(
        lambda_blend_mse=args.get("lambda_blend_mse", 1.0),
        lambda_image_mse=args.get("lambda_image_mse", 1.0),
        lambda_perc=args.get("lambda_perc", 0.1),
        lambda_tv=args.get("lambda_tv", 0.1),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.get("learning_rate", 1e-4))
    scheduler_type = args.get("lr_scheduler", "step")
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.get("num_epochs", 10),
            eta_min=args.get("lr_min", 1e-6),
        )
    else:
        scheduler = StepLR(
            optimizer,
            step_size=args.get("lr_step_size", 50),
            gamma=args.get("lr_gamma", 0.1),
        )

    # ── Resume from full checkpoint ──
    best_val_loss = float("inf")
    best_epoch = -1
    start_epoch = args.get("start_epoch", 0)
    resume_path = args.get("resume_path", "")
    if resume_path and os.path.isfile(resume_path):
        ckpt = load_full_checkpoint(resume_path, model, optimizer, scheduler, device=str(device))
        model = ckpt["model"].to(device)
        start_epoch = ckpt["start_epoch"]
        best_val_loss = ckpt["best_val_loss"]
        best_epoch = start_epoch  # approximate
        print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Dirs ──
    save_dir = args.get("save_dir", "weights-local")
    # When resuming, keep the same directory; otherwise create a timestamped sub-dir
    if not resume_path:
        run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, run_tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)
    # Save the config used for this run
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(args, f, default_flow_style=False, sort_keys=False)
    print(f"Save dir: {save_dir}")

    # ── MLflow ──
    use_mlflow = args.get("use_mlflow", False) and mlflow is not None
    if use_mlflow:
        mlflow.set_tracking_uri(args.get("mlflow_tracking_uri", "mlruns"))
        mlflow.set_experiment(args.get("project_name", "local-train"))
        mlflow.start_run(run_name=args.get("run_name"))
        mlflow.log_params({k: v for k, v in args.items() if isinstance(v, (str, int, float, bool))})

    # ── Training loop ──
    num_epochs = args.get("num_epochs", 10)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        val_results = validate(model, val_loader, criterion, device, epoch, num_epochs, save_dir)
        scheduler.step()

        elapsed = time.time() - t0
        eta = (num_epochs - epoch - 1) * elapsed

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_losses['total_loss']:.4f} | "
            f"val_loss={val_results['total_loss']:.4f} | "
            f"psnr={val_results['psnr']:.1f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.0f}s | ETA {datetime.timedelta(seconds=int(eta))}"
        )

        # MLflow logging
        if use_mlflow:
            mlflow.log_metrics(
                {f"train_{k}": v for k, v in train_losses.items()},
                step=epoch + 1,
            )
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val_results.items()},
                step=epoch + 1,
            )
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch + 1)

        # Save best
        if val_results["total_loss"] < best_val_loss:
            best_val_loss = val_results["total_loss"]
            best_epoch = epoch + 1
            # Save inference-ready weights
            torch.save(model.state_dict(), os.path.join(save_dir, "double_chin_bmap_best.pth"))
            # Save full checkpoint for resuming
            save_full_checkpoint(
                os.path.join(save_dir, "checkpoint_best.pth"),
                model, optimizer, scheduler, epoch, best_val_loss,
            )
            print(f"  ✓ New best (epoch {best_epoch}, loss {best_val_loss:.4f})")

        # Periodic save
        if (epoch + 1) % args.get("save_interval", 5) == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"double_chin_bmap_epoch_{epoch+1}.pth"))
            save_full_checkpoint(
                os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                model, optimizer, scheduler, epoch, best_val_loss,
            )

    # Final save
    torch.save(model.state_dict(), os.path.join(save_dir, "double_chin_bmap_final.pth"))
    save_full_checkpoint(
        os.path.join(save_dir, "checkpoint_latest.pth"),
        model, optimizer, scheduler, num_epochs - 1, best_val_loss,
    )
    print(f"\nDone. Best epoch {best_epoch}, val_loss {best_val_loss:.4f}")
    print(f"Saved to {save_dir}/")

    if use_mlflow:
        mlflow.log_artifact(os.path.join(save_dir, "double_chin_bmap_best.pth"))
        mlflow.end_run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local training (single GPU / MPS / CPU)")
    parser.add_argument("--config", "-c", default="blend_map.yaml", help="YAML config path")
    cli = parser.parse_args()

    with open(cli.config) as f:
        args = yaml.safe_load(f)

    # Test mode overrides
    if args.get("test", False):
        args["use_mlflow"] = False
        args["project_name"] = args.get("project_name", "local") + "_test"
        args["save_dir"] = args.get("save_dir", "weights-local") + "_test"

    train(args)


if __name__ == "__main__":
    main()
