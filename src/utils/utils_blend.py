import csv
import os
import subprocess
import torch
import cv2
import numpy as np


class CSVMetricsLogger:
    """Append-friendly CSV logger for per-epoch training metrics.

    Creates (or reopens) a ``metrics.csv`` file in *save_dir*.  A header row
    is written only when the file is first created.  Each call to
    :meth:`log_epoch` appends a single data row and flushes immediately so
    nothing is lost on a crash.

    The column order is fixed so every run produces an identical schema that
    is easy to ``pd.read_csv()`` later.
    """

    COLUMNS = [
        "epoch",
        "train_total_loss",
        "train_blend_map_loss",
        "train_blend_masked_loss",
        "train_blend_unmasked_loss",
        "train_image_mse_loss",
        "train_perc_loss",
        "train_tv_loss",
        "val_total_loss",
        "val_blend_map_loss",
        "val_blend_masked_loss",
        "val_blend_unmasked_loss",
        "val_image_mse_loss",
        "val_perc_loss",
        "val_tv_loss",
        "val_psnr",
        "val_ssim",
        "val_roi_psnr",
        "val_roi_ssim",
        "lr",
        "grad_norm_mean",
        "grad_norm_max",
        "grad_clip_ratio",
        "epoch_time_s",
    ]

    def __init__(self, save_dir: str, filename: str = "metrics.csv"):
        self.path = os.path.join(save_dir, filename)
        write_header = not os.path.exists(self.path)

        # On resume: detect column schema change and migrate gracefully
        if not write_header:
            try:
                with open(self.path, "r", newline="") as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, None)
                if existing_header is not None and existing_header != self.COLUMNS:
                    # Schema changed — back up old CSV and start fresh
                    import shutil
                    backup = self.path + ".bak"
                    shutil.copy2(self.path, backup)
                    write_header = True  # will create new file with updated header
                    # Log to stderr so it's visible even without a logger
                    import sys
                    print(
                        f"[CSVMetricsLogger] Column schema changed. "
                        f"Backed up old CSV to {backup} and starting fresh.",
                        file=sys.stderr,
                    )
            except Exception:
                pass  # if we can't read the header, just append as before

        mode = "w" if write_header and os.path.exists(self.path) else "a"
        self._fp = open(self.path, mode, newline="")
        self._writer = csv.writer(self._fp)
        if write_header:
            self._writer.writerow(self.COLUMNS)
            self._fp.flush()

    # ── public API ──

    def log_epoch(
        self,
        epoch: int,
        train_losses: dict,
        val_results: dict,
        lr: float,
        epoch_time: float,
    ) -> None:
        """Write one row of metrics for the completed *epoch* (1-based)."""
        row = [
            epoch,
            train_losses.get("total_loss", 0.0),
            train_losses.get("blend_map_loss", 0.0),
            train_losses.get("blend_masked_loss", 0.0),
            train_losses.get("blend_unmasked_loss", 0.0),
            train_losses.get("image_mse_loss", 0.0),
            train_losses.get("perc_loss", 0.0),
            train_losses.get("tv_loss", 0.0),
            val_results.get("total_loss", 0.0),
            val_results.get("blend_map_loss", 0.0),
            val_results.get("blend_masked_loss", 0.0),
            val_results.get("blend_unmasked_loss", 0.0),
            val_results.get("image_mse_loss", 0.0),
            val_results.get("perc_loss", 0.0),
            val_results.get("tv_loss", 0.0),
            val_results.get("psnr", 0.0),
            val_results.get("ssim", 0.0),
            val_results.get("roi_psnr", 0.0),
            val_results.get("roi_ssim", 0.0),
            lr,
            round(train_losses.get("grad_norm_mean", 0.0), 4),
            round(train_losses.get("grad_norm_max", 0.0), 4),
            round(train_losses.get("grad_clip_ratio", 0.0), 4),
            round(epoch_time, 1),
        ]
        self._writer.writerow(row)
        self._fp.flush()

    def close(self):
        self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def get_git_sha():
    """Return the current git short SHA, or 'unknown' if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        ).decode('ascii').strip()
        # Append '-dirty' if there are uncommitted changes
        dirty = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
        ).decode('ascii').strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return 'unknown'


def save_full_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, is_ddp=False, git_sha=None, model_config=None, best_psnr=None, best_ssim=None):
    """
    Save a full training checkpoint for resumable training.

    Args:
        path: File path to save the checkpoint
        model: PyTorch model (or DDP-wrapped model)
        optimizer: Optimizer
        scheduler: LR scheduler
        epoch: Current epoch (0-based, the epoch that just finished)
        best_val_loss: Best validation loss so far
        is_ddp: Whether the model is wrapped in DDP (will save module.state_dict)
        git_sha: Git commit hash (auto-detected if None)
        model_config: Dict with model architecture info (model_variant, last_layer_activation, blend_scale)
        best_psnr: Best validation PSNR so far (optional)
        best_ssim: Best validation SSIM so far (optional)
    """
    state_dict = model.module.state_dict() if is_ddp else model.state_dict()
    tmp_path = path + '.tmp'
    payload = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'git_sha': git_sha or get_git_sha(),
    }
    if model_config is not None:
        payload['model_config'] = model_config
    if best_psnr is not None:
        payload['best_psnr'] = best_psnr
    if best_ssim is not None:
        payload['best_ssim'] = best_ssim
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)  # atomic on Linux/macOS — crash-safe


def load_full_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load a full training checkpoint for resuming training.

    Args:
        path: File path to the checkpoint
        model: PyTorch model (must be un-wrapped, i.e. before DDP wrapping)
        optimizer: Optimizer (optional — skipped if None)
        scheduler: LR scheduler (optional — skipped if None)
        device: Device to map tensors to

    Returns:
        dict with keys: model, optimizer, scheduler, epoch (next epoch to run),
        best_val_loss
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Load model weights (strip DDP 'module.' prefix if present)
    state_dict = ckpt['model_state_dict']
    clean = {}
    for k, v in state_dict.items():
        clean[k.removeprefix('module.')] = v
    model.load_state_dict(clean)

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    start_epoch = ckpt.get('epoch', -1) + 1  # next epoch to run
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    best_psnr = ckpt.get('best_psnr', 0.0)
    best_ssim = ckpt.get('best_ssim', 0.0)

    print(f"Resumed from {path} (epoch {ckpt.get('epoch', '?')}, best_val_loss={best_val_loss:.4f}, best_psnr={best_psnr:.2f}, best_ssim={best_ssim:.4f})")
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'start_epoch': start_epoch,
        'best_val_loss': best_val_loss,
        'best_psnr': best_psnr,
        'best_ssim': best_ssim,
    }


def load_checkpoint(model, checkpoint_path):
    """
    Load model weights from checkpoint file.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        
    Returns:
        model: Model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract raw state_dict from various checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and any(k.endswith('.weight') for k in checkpoint):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Strip DDP 'module.' prefix if present
        clean = {k.removeprefix('module.'): v for k, v in state_dict.items()}
        model.load_state_dict(clean)
            
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    return model


def save_training_data_montage(data_loader, save_dir, num_montages=5, samples_per_montage=8, thumb_size=256):
    """
    Save YOLO-style montage grids of training data (with augmentations applied).

    Each montage is a grid of ``samples_per_montage`` rows × 3 columns:
        Input | Blend Map | Ground Truth

    Column headers are burned into the top of each montage.

    Args:
        data_loader: Training DataLoader (yields augmented batches)
        save_dir: Directory to save montage images
        num_montages: Number of montage images to generate (default 5)
        samples_per_montage: Number of samples (rows) per montage (default 8)
        thumb_size: Height/width of each thumbnail cell in pixels
    """
    montage_dir = os.path.join(save_dir, 'train_data_montage')
    os.makedirs(montage_dir, exist_ok=True)

    headers = ['Input', 'Blend Map', 'Ground Truth']
    num_cols = len(headers)
    header_h = 36  # pixels for the header row
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2

    collected = []  # list of (image, blend_map, gt) numpy arrays
    total_needed = num_montages * samples_per_montage

    # Use a temporary single-process DataLoader to avoid worker cleanup issues
    # (the montage only needs a handful of batches — no benefit from multiprocessing)
    from torch.utils.data import DataLoader
    tmp_loader = DataLoader(
        data_loader.dataset,
        batch_size=data_loader.batch_size,
        shuffle=True,
        num_workers=0,  # no worker processes → no messy cleanup on early break
    )
    for batch in tmp_loader:
        images = batch['image']       # [B, 3, H, W]
        blend_maps = batch['blend_map']
        gts = batch['gt']
        for i in range(images.size(0)):
            if len(collected) >= total_needed:
                break
            img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            bm_np = (blend_maps[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_np = (gts[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            collected.append((img_np, bm_np, gt_np))
        if len(collected) >= total_needed:
            break

    # Build montages
    for m_idx in range(num_montages):
        start = m_idx * samples_per_montage
        samples = collected[start:start + samples_per_montage]
        if not samples:
            break

        grid_w = num_cols * thumb_size
        grid_h = header_h + len(samples) * thumb_size
        canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark grey bg

        # Draw column headers
        for col_idx, label in enumerate(headers):
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            x = col_idx * thumb_size + (thumb_size - text_size[0]) // 2
            y = header_h - 10
            cv2.putText(canvas, label, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Fill grid cells
        for row_idx, (img, bm, gt) in enumerate(samples):
            y_off = header_h + row_idx * thumb_size
            for col_idx, cell in enumerate([img, bm, gt]):
                x_off = col_idx * thumb_size
                thumb = cv2.resize(cell, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
                # RGB → BGR for cv2.imwrite
                thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
                canvas[y_off:y_off + thumb_size, x_off:x_off + thumb_size] = thumb_bgr

        path = os.path.join(montage_dir, f'train_montage_{m_idx + 1}.jpg')
        cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"Saved {min(num_montages, len(collected) // max(samples_per_montage, 1))} "
          f"training data montages to {montage_dir}")


def save_visualization_batch(batch, outputs, save_dir, prefix="", max_samples=4):
    """
    Save visualization of model outputs.
    
    Args:
        batch: Batch of input data (dict with 'image', 'blend_map', 'gt')
        outputs: Model outputs (predicted blend maps)
        save_dir: Directory to save visualizations
        prefix: Prefix for saved filenames (e.g., epoch number)
        max_samples: Maximum number of samples to save
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Get tensors from batch and move everything to CPU for visualization
    images = batch['image'].detach().cpu()
    gt_blend_maps = batch['blend_map'].detach().cpu()
    gt_images = batch['gt'].detach().cpu()
    filenames = batch['filename']
    
    # Move outputs to CPU
    outputs_cpu = outputs.detach().cpu()
    
    # Compute retouched images from predictions (on CPU)
    from src.blend.blend_map import apply_blend_formula
    retouched_images = apply_blend_formula(images, outputs_cpu)
    
    # Process each sample in batch (up to max_samples)
    for i in range(min(max_samples, images.size(0))):
        # Original image
        orig_img = images[i].permute(1, 2, 0).numpy() * 255
        orig_img = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Ground truth
        gt_img = gt_images[i].permute(1, 2, 0).numpy() * 255
        gt_img = cv2.cvtColor(gt_img.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Ground truth blend map
        gt_blend = gt_blend_maps[i].permute(1, 2, 0).numpy() * 255
        gt_blend = cv2.cvtColor(gt_blend.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Predicted blend map
        pred_blend = outputs_cpu[i].permute(1, 2, 0).numpy() * 255
        pred_blend = cv2.cvtColor(pred_blend.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Retouched image
        retouched = retouched_images[i].permute(1, 2, 0).numpy() * 255
        retouched = cv2.cvtColor(retouched.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Get base filename without extension
        base_name = os.path.splitext(filenames[i])[0]

        # Grid: top row = [original, gt_blend, gt], bottom row = [original, pred_blend, retouched]
        grid = np.concatenate([
            np.concatenate([orig_img, gt_blend, gt_img], axis=1),
            np.concatenate([orig_img, pred_blend, retouched], axis=1)
        ], axis=0)
        cv2.imwrite(f"{save_dir}/{prefix}_{base_name}_grid.jpg", grid)


def compute_metrics(pred_images, gt_images, roi_crop_fraction=0.0):
    """
    Compute evaluation metrics between predicted and ground truth images.
    
    Args:
        pred_images: Predicted images tensor [B, C, H, W]
        gt_images: Ground truth images tensor [B, C, H, W]
        roi_crop_fraction: If > 0, also compute PSNR/SSIM on the bottom
            fraction of the image (the chin/edit region).  E.g. 0.5 means
            bottom 50%.  Set to 0 to skip ROI metrics.
        
    Returns:
        dict: Dictionary of metrics (psnr, ssim, and optionally
              roi_psnr, roi_ssim)
    """
    from skimage.metrics import structural_similarity as ssim_fn

    # Make sure tensors are on CPU and detached from the computation graph
    pred_np = pred_images.detach().cpu().numpy()
    gt_np = gt_images.detach().cpu().numpy()
    
    batch_size = pred_np.shape[0]
    psnr_values = []
    ssim_values = []
    roi_psnr_values = []
    roi_ssim_values = []

    compute_roi = roi_crop_fraction > 0.0
    
    for i in range(batch_size):
        pred = pred_np[i].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        gt = gt_np[i].transpose(1, 2, 0)
        
        # ── Full-image metrics ──
        mse = np.mean((pred - gt) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(1.0 / mse)
        psnr_values.append(psnr)
        ssim_values.append(ssim_fn(gt, pred, data_range=1.0, channel_axis=2))

        # ── ROI (bottom crop) metrics ──
        if compute_roi:
            h = pred.shape[0]
            crop_start = int(h * (1.0 - roi_crop_fraction))
            pred_roi = pred[crop_start:, :, :]
            gt_roi = gt[crop_start:, :, :]

            roi_mse = np.mean((pred_roi - gt_roi) ** 2)
            if roi_mse == 0:
                roi_psnr = float('inf')
            else:
                roi_psnr = 10 * np.log10(1.0 / roi_mse)
            roi_psnr_values.append(roi_psnr)
            roi_ssim_values.append(
                ssim_fn(gt_roi, pred_roi, data_range=1.0, channel_axis=2)
            )
    
    result = {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
    }
    if compute_roi:
        result['roi_psnr'] = np.mean(roi_psnr_values)
        result['roi_ssim'] = np.mean(roi_ssim_values)
    return result