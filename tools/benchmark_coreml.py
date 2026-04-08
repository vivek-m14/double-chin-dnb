#!/usr/bin/env python3
"""
CoreML Benchmark for Double Chin Removal Model.

Converts the PyTorch BaseUNetHalf model to CoreML (float16) and benchmarks
inference latency on the Apple Neural Engine / GPU / CPU.

Steps:
    1. Load PyTorch checkpoint → export to CoreML via coremltools
    2. Run warmup iterations
    3. Benchmark N iterations and report latency stats

Requirements:
    pip install coremltools torch torchvision

Usage:
    python tools/benchmark_coreml.py
    python tools/benchmark_coreml.py --model weights-blend-maps/.../best.pth
    python tools/benchmark_coreml.py --img-size 512 --iterations 100
    python tools/benchmark_coreml.py --compute-units all      # ANE + GPU + CPU
    python tools/benchmark_coreml.py --compute-units cpu_and_gpu
    python tools/benchmark_coreml.py --save-mlmodel model.mlpackage
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time

import numpy as np
import torch

log = logging.getLogger(__name__)

# ── Project imports ──
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))
from models.unet import BaseUNetHalf

try:
    import coremltools as ct
except ImportError:
    print("ERROR: coremltools not installed. Run: pip install coremltools")
    sys.exit(1)

# Default model path (same as inference.py)
DEFAULT_MODEL = "weights-blend-maps/blend_maps_v3_2/double_chin_bmap_best.pth"


# ---------------------------------------------------------------------------
# Model loading (same logic as inference.py)
# ---------------------------------------------------------------------------

def load_pytorch_model(checkpoint_path: str) -> torch.nn.Module:
    """Load the BaseUNetHalf model from a checkpoint."""
    model = BaseUNetHalf(
        n_channels=3,
        n_classes=3,
        deep_supervision=False,
        init_weights=False,
        last_layer_activation="sigmoid",
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch versions don't support weights_only
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip DDP 'module.' prefix
    clean = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        log.warning("Missing keys in checkpoint: %s", missing)
    if unexpected:
        log.warning("Unexpected keys in checkpoint: %s", unexpected)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# CoreML conversion
# ---------------------------------------------------------------------------

def convert_to_coreml(
    model: torch.nn.Module,
    img_size: int = 1024,
    compute_units: str = "all",
    save_path: str | None = None,
) -> "ct.models.MLModel":
    """
    Convert PyTorch model to CoreML float16 via torch.export.

    Args:
        model: PyTorch model in eval mode.
        img_size: Spatial input size (square).
        compute_units: 'all' (ANE+GPU+CPU), 'cpu_and_gpu', or 'cpu_only'.
        save_path: If set, save the .mlpackage to this path.

    Returns:
        CoreML MLModel ready for prediction.
    """
    # Map string to coremltools enum
    cu_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }
    cu = cu_map.get(compute_units, ct.ComputeUnit.ALL)

    log.info("Exporting model via torch.export (input [1, 3, %d, %d])...", img_size, img_size)
    # Work on a CPU copy to avoid mutating the caller's model
    import copy
    model_cpu = copy.deepcopy(model).cpu()
    dummy = torch.randn(1, 3, img_size, img_size)

    try:
        exported = torch.export.export(model_cpu, (dummy,), strict=False)
        exported = exported.run_decompositions({})
    except Exception as exc:
        log.error("torch.export failed: %s", exc)
        raise SystemExit(1) from exc

    log.info("Converting to CoreML (float16)...")
    try:
        mlmodel = ct.convert(
            exported,
            inputs=[ct.TensorType(name="input_image", shape=(1, 3, img_size, img_size))],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS13,
            compute_precision=ct.precision.FLOAT16,
            compute_units=cu,
        )
    except Exception as exc:
        log.error("CoreML conversion failed: %s", exc)
        raise SystemExit(1) from exc

    if save_path:
        mlmodel.save(save_path)
        log.info("Saved CoreML model to %s", save_path)

    return mlmodel


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_coreml(
    mlmodel: "ct.models.MLModel",
    img_size: int = 1024,
    warmup: int = 5,
    iterations: int = 50,
) -> dict:
    """
    Benchmark CoreML model inference latency.

    Returns dict with latency stats in milliseconds.
    """
    # Create a random input tensor (float32, [0, 1] range, matching model input)
    rng = np.random.default_rng(42)
    input_data = rng.random((1, 3, img_size, img_size), dtype=np.float32)

    # Discover the model's input name (torch.export uses 'x' from forward(self, x))
    input_name = list(mlmodel.input_description)[0]

    # Warmup
    log.info("CoreML warmup (%d iterations)...", warmup)
    for _ in range(warmup):
        mlmodel.predict({input_name: input_data})

    # Benchmark
    log.info("CoreML benchmark (%d iterations)...", iterations)
    latencies = []
    for i in range(iterations):
        t0 = time.perf_counter()
        mlmodel.predict({input_name: input_data})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

        if (i + 1) % 10 == 0:
            log.info("  [%d/%d] last = %.1f ms", i + 1, iterations, latencies[-1])

    latencies = np.array(latencies)
    stats = {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "fps": float(1000.0 / latencies.mean()),
        "iterations": iterations,
    }
    return stats


def benchmark_pytorch(
    model: torch.nn.Module,
    img_size: int = 1024,
    device: str = "cpu",
    warmup: int = 5,
    iterations: int = 50,
) -> dict:
    """Benchmark PyTorch model for comparison."""
    import copy
    model_dev = copy.deepcopy(model).to(device)
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    log.info("PyTorch warmup on %s (%d iterations)...", device, warmup)
    with torch.inference_mode():
        for _ in range(warmup):
            model_dev(dummy)
    if device == "mps":
        torch.mps.synchronize()

    log.info("PyTorch benchmark on %s (%d iterations)...", device, iterations)
    latencies = []
    with torch.inference_mode():
        for i in range(iterations):
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            model_dev(dummy)
            if device == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            if (i + 1) % 10 == 0:
                log.info("  [%d/%d] last = %.1f ms", i + 1, iterations, latencies[-1])

    latencies = np.array(latencies)
    return {
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "min_ms": float(latencies.min()),
        "max_ms": float(latencies.max()),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "fps": float(1000.0 / latencies.mean()),
        "iterations": iterations,
    }


def print_stats(label: str, stats: dict):
    """Pretty-print latency stats."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Mean:   {stats['mean_ms']:8.1f} ms  ({stats['fps']:.1f} FPS)")
    print(f"  Median: {stats['median_ms']:8.1f} ms")
    print(f"  Std:    {stats['std_ms']:8.1f} ms")
    print(f"  Min:    {stats['min_ms']:8.1f} ms")
    print(f"  Max:    {stats['max_ms']:8.1f} ms")
    print(f"  P95:    {stats['p95_ms']:8.1f} ms")
    print(f"  P99:    {stats['p99_ms']:8.1f} ms")
    print(f"  Iters:  {stats['iterations']}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def validate_numerical_accuracy(
    pt_model: torch.nn.Module,
    mlmodel: "ct.models.MLModel",
    img_size: int = 1024,
    atol: float = 1e-2,
) -> dict:
    """Compare PyTorch float32 and CoreML float16 outputs for numerical drift."""
    rng = np.random.default_rng(123)
    input_np = rng.random((1, 3, img_size, img_size), dtype=np.float32)

    # PyTorch reference
    with torch.inference_mode():
        pt_out = pt_model.cpu()(torch.from_numpy(input_np)).numpy()

    # CoreML
    input_name = list(mlmodel.input_description)[0]
    coreml_out = mlmodel.predict({input_name: input_np})
    output_name = list(mlmodel.output_description)[0]
    cm_out = np.array(coreml_out[output_name])

    max_abs_err = float(np.max(np.abs(pt_out - cm_out)))
    mean_abs_err = float(np.mean(np.abs(pt_out - cm_out)))
    passed = max_abs_err < atol

    status = "PASS" if passed else "FAIL"
    log.info(
        "Numerical validation: %s  (max_err=%.4f, mean_err=%.6f, atol=%.4f)",
        status, max_abs_err, mean_abs_err, atol,
    )
    return {
        "status": status,
        "max_abs_error": max_abs_err,
        "mean_abs_error": mean_abs_err,
        "atol": atol,
    }


def print_env_info() -> dict:
    """Print and return environment info for reproducibility."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "os": f"{platform.system()} {platform.mac_ver()[0]}" if platform.system() == "Darwin" else platform.platform(),
        "chip": platform.processor() or platform.machine(),
    }
    log.info("Environment:")
    for k, v in info.items():
        log.info("  %-12s %s", k, v)
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark double chin model on CoreML (float16)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default=DEFAULT_MODEL,
        help="Path to PyTorch checkpoint (.pth)",
    )
    parser.add_argument(
        "--img-size", type=int, default=1024,
        help="Input image size (default: 1024)",
    )
    parser.add_argument(
        "--compute-units", type=str, default="all",
        choices=["all", "cpu_and_gpu", "cpu_only", "cpu_and_ne"],
        help="CoreML compute units (default: all = ANE+GPU+CPU)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=50,
        help="Number of benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--save-mlmodel", type=str, default=None,
        help="Save CoreML model to this path (.mlpackage)",
    )
    parser.add_argument(
        "--skip-pytorch", action="store_true",
        help="Skip PyTorch baseline benchmark",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip numerical accuracy validation",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Write results to JSON file for CI/automation",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress logs (only print final stats)",
    )
    args = parser.parse_args()

    # ── Logging setup ──
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(message)s",
    )

    if not os.path.isfile(args.model):
        log.error("Model not found: %s", args.model)
        sys.exit(1)

    # ── Environment info ──
    env_info = print_env_info()

    # ── Load PyTorch model ──
    log.info("Loading PyTorch model from %s...", args.model)
    pt_model = load_pytorch_model(args.model)
    param_count = sum(p.numel() for p in pt_model.parameters())
    log.info("  Parameters: %s (%.1fM)", f"{param_count:,}", param_count / 1e6)

    results: dict = {
        "environment": env_info,
        "config": {
            "model": args.model,
            "img_size": args.img_size,
            "compute_units": args.compute_units,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "parameters": param_count,
        },
    }

    # ── PyTorch baseline ──
    pt_stats = None
    if not args.skip_pytorch:
        pt_device = "mps" if torch.backends.mps.is_available() else "cpu"
        pt_stats = benchmark_pytorch(
            pt_model, args.img_size, pt_device,
            warmup=args.warmup, iterations=args.iterations,
        )
        print_stats(f"PyTorch ({pt_device}) — float32 — {args.img_size}×{args.img_size}", pt_stats)
        results["pytorch"] = {"device": pt_device, "dtype": "float32", **pt_stats}

    # ── Convert to CoreML ──
    mlmodel = convert_to_coreml(
        pt_model,
        img_size=args.img_size,
        compute_units=args.compute_units,
        save_path=args.save_mlmodel,
    )

    # ── Numerical validation ──
    if not args.skip_validation:
        val = validate_numerical_accuracy(pt_model, mlmodel, args.img_size)
        results["validation"] = val

    # ── CoreML benchmark ──
    coreml_stats = benchmark_coreml(
        mlmodel, args.img_size,
        warmup=args.warmup, iterations=args.iterations,
    )
    print_stats(
        f"CoreML (float16, {args.compute_units}) — {args.img_size}×{args.img_size}",
        coreml_stats,
    )
    results["coreml"] = {"compute_units": args.compute_units, "dtype": "float16", **coreml_stats}

    # ── Comparison ──
    if pt_stats is not None:
        speedup = pt_stats["mean_ms"] / coreml_stats["mean_ms"]
        results["speedup"] = round(speedup, 3)
        print(f"\n  CoreML speedup vs PyTorch: {speedup:.2f}x")
        if speedup > 1:
            print(f"  CoreML is {speedup:.1f}x faster")
        else:
            print(f"  PyTorch is {1/speedup:.1f}x faster")

    # ── JSON output ──
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        log.info("Results written to %s", args.output_json)


if __name__ == "__main__":
    main()
