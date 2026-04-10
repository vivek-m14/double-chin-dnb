#!/usr/bin/env python3
"""
Face Identity Leakage Checker
==============================
Ensures no person (face identity) in the **validation** set also appears in
the **training** set.

Uses **InsightFace** (ArcFace, buffalo_l) 512-d face embeddings to:
  1. Reproduce the exact train / val split used by ``create_data_loaders``
     (seeded ``torch.randperm``).
  2. Extract face embeddings from every image in both splits.
  3. Compare all val embeddings against all train embeddings.
  4. Report any matches above a cosine-similarity threshold.

Usage
-----
    python tools/check_face_leakage.py \\
        --data_root /path/to/dataset_root \\
        --data_json /path/to/full_dataset.json \\
        --threshold 0.4 \\
        --workers 4

Install
-------
    pip install insightface onnxruntime numpy torch

Outputs
-------
  - Console report of leaked identities (if any)
  - Optional CSV with all pairwise matches above threshold
"""

import argparse
import csv
import json
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Reproduce train / val split (mirrors src/data/dataset.py)
# ---------------------------------------------------------------------------


def load_split_indices(
    data_json: str,
    seed: int = 42,
    train_ratio: float = 0.9,
    exclude_source: str = "aadan_double_chin",
) -> Tuple[List[dict], List[int], List[int]]:
    """
    Load the dataset JSON and reproduce the train/val index split used by
    ``create_data_loaders`` in ``src/data/dataset.py``.

    Returns
    -------
    data_map : list[dict]
        Filtered data entries.
    train_indices, val_indices : list[int]
        Index lists into *data_map*.
    """
    import torch

    with open(data_json, "r") as f:
        data_map = json.load(f)

    # Same filter as BlendMapDataset.__init__
    if exclude_source:
        data_map = [d for d in data_map if d.get("source") != exclude_source]

    n = len(data_map)
    train_size = int(n * train_ratio)
    split_gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=split_gen).tolist()

    train_indices = perm[:train_size]
    val_indices = perm[train_size:]
    return data_map, train_indices, val_indices


# ---------------------------------------------------------------------------
# Face embedding extraction  (InsightFace / ArcFace)
# ---------------------------------------------------------------------------


def _build_face_app(det_size: int = 640):
    """Build and return an InsightFace FaceAnalysis app (CPU)."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def extract_embedding_insightface(
    image_path: str,
    app,
) -> Optional[np.ndarray]:
    """
    Extract a 512-d ArcFace embedding from *image_path*.

    Returns the embedding of the **largest** detected face, or ``None``
    if no face is found.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    faces = app.get(img)
    if not faces:
        return None

    # Pick largest face by bounding-box area
    def _area(face):
        bbox = face.bbox  # [x1, y1, x2, y2]
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    best = max(faces, key=_area)
    emb = best.normed_embedding  # already L2-normalized, 512-d
    return emb


# Global app for worker processes
_WORKER_APP = None


def _worker_init(det_size: int = 640):
    """Initialize InsightFace app in each worker process."""
    global _WORKER_APP
    # Suppress onnxruntime logging in workers
    os.environ["ORT_LOG_LEVEL"] = "3"
    _WORKER_APP = _build_face_app(det_size)


def _worker_extract(args_tuple):
    """Picklable wrapper for multiprocessing."""
    idx, total, image_path, split_name = args_tuple
    global _WORKER_APP
    try:
        emb = extract_embedding_insightface(image_path, _WORKER_APP)
        return {
            "status": "ok",
            "idx": idx,
            "total": total,
            "image_path": image_path,
            "split": split_name,
            "embedding": emb,
        }
    except Exception as e:
        return {
            "status": "error",
            "idx": idx,
            "total": total,
            "image_path": image_path,
            "split": split_name,
            "error": str(e),
        }


def extract_all_embeddings(
    image_paths: List[str],
    split_name: str,
    workers: int = 4,
    det_size: int = 640,
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for all images. Returns dict mapping path → embedding.
    Images where no face is detected are skipped (logged).
    """
    results: Dict[str, np.ndarray] = {}
    no_face: List[str] = []
    errors: List[str] = []

    work_items = [
        (i, len(image_paths), p, split_name)
        for i, p in enumerate(image_paths)
    ]

    t0 = time.time()

    if workers > 1 and len(image_paths) > 1:
        with multiprocessing.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(det_size,),
        ) as pool:
            for res in pool.imap_unordered(_worker_extract, work_items, chunksize=4):
                _i = res["idx"] + 1
                _t = res["total"]
                path = res["image_path"]
                basename = os.path.basename(path)
                if res["status"] == "ok":
                    if res["embedding"] is not None:
                        results[path] = res["embedding"]
                        if _i % 50 == 0 or _i == _t:
                            elapsed = time.time() - t0
                            rate = _i / elapsed if elapsed > 0 else 0
                            print(
                                f"  [{split_name}] [{_i:4d}/{_t}]  "
                                f"{len(results)} faces  ({rate:.1f} img/s)"
                            )
                    else:
                        no_face.append(path)
                else:
                    errors.append(path)
                    print(f"  [{split_name}] [{_i:4d}/{_t}] ✗ Error: {basename} — {res['error']}")
    else:
        # Sequential — single app instance
        app = _build_face_app(det_size)
        for i, (_, _, path, _) in enumerate(work_items):
            basename = os.path.basename(path)
            try:
                emb = extract_embedding_insightface(path, app)
                if emb is not None:
                    results[path] = emb
                else:
                    no_face.append(path)
            except Exception as e:
                errors.append(path)
                print(f"  [{split_name}] [{i+1:4d}/{len(image_paths)}] ✗ Error: {basename} — {e}")

            if (i + 1) % 50 == 0 or i + 1 == len(image_paths):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"  [{split_name}] [{i+1:4d}/{len(image_paths)}]  "
                    f"{len(results)} faces  ({rate:.1f} img/s)"
                )

    elapsed = time.time() - t0
    print(
        f"\n  [{split_name}] Done: {len(results)} embeddings, "
        f"{len(no_face)} no-face, {len(errors)} errors  ({elapsed:.1f}s)"
    )
    if no_face:
        print(f"  [{split_name}] No-face examples: {[os.path.basename(p) for p in no_face[:5]]}")
    return results


# ---------------------------------------------------------------------------
# UUID-based identity grouping (fast pre-check)
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)


def extract_uuid(path_or_name: str) -> Optional[str]:
    """Extract UUID from a file path / name, if present."""
    m = _UUID_RE.search(os.path.basename(path_or_name))
    return m.group(1) if m else None


def check_uuid_leakage(
    data_map: List[dict],
    train_indices: List[int],
    val_indices: List[int],
    image_key: str = "original_image",
) -> Tuple[set, set, set]:
    """
    Fast pre-check: do any UUIDs appear in both train and val?

    Returns (leaked_uuids, train_uuids, val_uuids).
    """
    train_uuids = set()
    for idx in train_indices:
        u = extract_uuid(data_map[idx].get(image_key, ""))
        if u:
            train_uuids.add(u)

    val_uuids = set()
    for idx in val_indices:
        u = extract_uuid(data_map[idx].get(image_key, ""))
        if u:
            val_uuids.add(u)

    leaked = train_uuids & val_uuids
    return leaked, train_uuids, val_uuids


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


def find_leaks(
    train_embeddings: Dict[str, np.ndarray],
    val_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.4,
) -> List[dict]:
    """
    Compare every val embedding against every train embedding using cosine
    similarity (embeddings are already L2-normalized by InsightFace).

    Returns list of matches with similarity ≥ threshold, sorted by
    descending similarity.
    """
    matches = []

    train_paths = list(train_embeddings.keys())
    train_embs = np.array([train_embeddings[p] for p in train_paths], dtype=np.float64)  # (N_train, 512)

    # Normalize rows to guard against any non-unit embeddings
    norms = np.linalg.norm(train_embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    train_embs = train_embs / norms

    val_paths = list(val_embeddings.keys())

    print(
        f"\nComparing {len(val_paths)} val embeddings "
        f"against {len(train_paths)} train embeddings..."
    )
    t0 = time.time()

    for vi, vpath in enumerate(val_paths):
        v_emb = val_embeddings[vpath].astype(np.float64)
        v_norm = np.linalg.norm(v_emb)
        if v_norm < 1e-10:
            continue
        v_emb = v_emb / v_norm

        # Cosine similarity = dot product (both are L2-normalized)
        sims = train_embs @ v_emb  # (N_train,)

        above = np.where(sims >= threshold)[0]
        for ti in above:
            matches.append(
                {
                    "val_image": vpath,
                    "train_image": train_paths[ti],
                    "similarity": float(sims[ti]),
                }
            )

        if (vi + 1) % 50 == 0 or vi == len(val_paths) - 1:
            elapsed = time.time() - t0
            print(
                f"  Compared [{vi+1}/{len(val_paths)}]  "
                f"matches so far: {len(matches)}  ({elapsed:.1f}s)"
            )

    matches.sort(key=lambda m: m["similarity"], reverse=True)
    return matches


# ---------------------------------------------------------------------------
# Clustering matched identities
# ---------------------------------------------------------------------------


def cluster_leaked_identities(matches: List[dict]) -> List[dict]:
    """
    Group pairwise matches into identity clusters using Union-Find.

    Each cluster represents one person who appears in both train and val.

    Returns
    -------
    list[dict]
        Each dict has: train_images, val_images, max_similarity, num_pairs.
    """
    parent: Dict[str, str] = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for m in matches:
        union(m["val_image"], m["train_image"])

    # Group by root
    all_nodes = set()
    for m in matches:
        all_nodes.add(m["val_image"])
        all_nodes.add(m["train_image"])

    groups: Dict[str, set] = {}
    for node in all_nodes:
        root = find(node)
        groups.setdefault(root, set()).add(node)

    # Build cluster info
    clusters = []
    for root, members in groups.items():
        train_imgs = set()
        val_imgs = set()
        max_sim = 0.0
        for m in matches:
            if m["val_image"] in members or m["train_image"] in members:
                train_imgs.add(m["train_image"])
                val_imgs.add(m["val_image"])
                max_sim = max(max_sim, m["similarity"])
        clusters.append(
            {
                "train_images": sorted(train_imgs),
                "val_images": sorted(val_imgs),
                "max_similarity": max_sim,
                "num_pairs": len(train_imgs) * len(val_imgs),
            }
        )

    clusters.sort(key=lambda c: c["max_similarity"], reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Check for face identity leakage between train and val splits."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of the dataset (parent of double_chin_data_v3/).",
    )
    parser.add_argument(
        "--data_json",
        type=str,
        required=True,
        help="Path to the dataset JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (must match training).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Fraction of data used for training (must match training).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help=(
            "Cosine similarity threshold to flag a match. "
            "InsightFace ArcFace: >0.4 = likely same person, "
            ">0.5 = very likely, >0.6 = almost certain."
        ),
    )
    parser.add_argument(
        "--image_key",
        type=str,
        default="original_image",
        help="JSON key for the image path (default: original_image).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for embedding extraction.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Max images per split to process (0 = all). Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Path to save detailed match CSV (optional).",
    )
    parser.add_argument(
        "--exclude_source",
        type=str,
        default="aadan_double_chin",
        help="Source to exclude from the dataset (same as training filter).",
    )
    parser.add_argument(
        "--det_size",
        type=int,
        default=640,
        help="InsightFace detection input size (default: 640).",
    )
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  FACE IDENTITY LEAKAGE CHECK")
    print(f"  data_root:  {args.data_root}")
    print(f"  data_json:  {args.data_json}")
    print(f"  seed:       {args.seed}")
    print(f"  threshold:  {args.threshold}")
    print(f"{'='*70}\n")

    # ── 1. Reproduce the train/val split ──────────────────────────────────
    print("Step 1: Reproducing train/val split...")
    data_map, train_indices, val_indices = load_split_indices(
        args.data_json,
        seed=args.seed,
        train_ratio=args.train_ratio,
        exclude_source=args.exclude_source,
    )
    print(f"  Total samples (after filtering): {len(data_map)}")
    print(f"  Train: {len(train_indices)},  Val: {len(val_indices)}\n")

    # ── 1b. Fast UUID-based pre-check ─────────────────────────────────────
    print("Step 1b: UUID-based pre-check (same person = same UUID prefix)...")
    leaked_uuids, train_uuids, val_uuids = check_uuid_leakage(
        data_map, train_indices, val_indices, args.image_key,
    )
    print(f"  Unique UUIDs — train: {len(train_uuids)}, val: {len(val_uuids)}")
    if leaked_uuids:
        print(f"  🔴 {len(leaked_uuids)} UUID(s) appear in BOTH train and val!")
        for u in sorted(leaked_uuids)[:20]:
            print(f"     {u}")
        if len(leaked_uuids) > 20:
            print(f"     ... and {len(leaked_uuids) - 20} more")
    else:
        print(f"  ✅ No UUID overlap between train and val.")
    print()

    # ── 2. Resolve image paths ────────────────────────────────────────────
    def resolve_paths(indices, label):
        paths = []
        missing = 0
        for idx in indices:
            rel = data_map[idx].get(args.image_key, "")
            full = os.path.join(args.data_root, rel)
            if os.path.isfile(full):
                paths.append(full)
            else:
                missing += 1
        if missing:
            print(f"  ⚠ {label}: {missing}/{len(indices)} images not found on disk (skipped).")
        return paths

    print("Step 2: Resolving image paths...")
    train_paths = resolve_paths(train_indices, "Train")
    val_paths = resolve_paths(val_indices, "Val")

    if args.max_images > 0:
        rng = np.random.RandomState(args.seed)
        if len(train_paths) > args.max_images:
            sel = rng.choice(len(train_paths), args.max_images, replace=False)
            train_paths = [train_paths[i] for i in sorted(sel)]
        if len(val_paths) > args.max_images:
            sel = rng.choice(len(val_paths), args.max_images, replace=False)
            val_paths = [val_paths[i] for i in sorted(sel)]

    print(f"  Will process: {len(train_paths)} train, {len(val_paths)} val images.\n")

    # ── 3. Extract face embeddings ────────────────────────────────────────
    print("Step 3: Extracting face embeddings (train)...")
    train_embs = extract_all_embeddings(
        train_paths, "train", workers=args.workers, det_size=args.det_size,
    )

    print("\nStep 4: Extracting face embeddings (val)...")
    val_embs = extract_all_embeddings(
        val_paths, "val", workers=args.workers, det_size=args.det_size,
    )

    if not train_embs or not val_embs:
        print("\n⚠ Not enough embeddings to compare. Exiting.")
        return

    # ── 4. Find leaks ────────────────────────────────────────────────────
    print(f"\nStep 5: Comparing embeddings (threshold={args.threshold})...")
    matches = find_leaks(train_embs, val_embs, threshold=args.threshold)

    # ── 5. Report ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")

    if not matches:
        print(f"\n  ✅ NO LEAKAGE DETECTED at threshold {args.threshold}!")
        print(
            f"     All {len(val_embs)} val identities are distinct from "
            f"{len(train_embs)} train identities."
        )
    else:
        clusters = cluster_leaked_identities(matches)
        unique_val = set()
        unique_train = set()
        for m in matches:
            unique_val.add(m["val_image"])
            unique_train.add(m["train_image"])

        print(f"\n  🔴 POTENTIAL LEAKAGE DETECTED!")
        print(f"     {len(matches)} pairwise matches above threshold {args.threshold}")
        print(f"     {len(clusters)} leaked identity cluster(s)")
        print(
            f"     {len(unique_val)} val images matched to "
            f"{len(unique_train)} train images"
        )

        print(f"\n  ── Top matches ──")
        for i, m in enumerate(matches[:20]):
            val_name = os.path.basename(m["val_image"])
            train_name = os.path.basename(m["train_image"])
            print(
                f"    {i+1:3d}. sim={m['similarity']:.4f}  "
                f"val={val_name}  ↔  train={train_name}"
            )

        if len(clusters) <= 30:
            print(f"\n  ── Identity clusters ──")
            for ci, c in enumerate(clusters):
                print(f"\n    Cluster {ci+1} (max_sim={c['max_similarity']:.4f}):")
                for vp in c["val_images"][:5]:
                    print(f"      [VAL]   {os.path.basename(vp)}")
                for tp in c["train_images"][:5]:
                    print(f"      [TRAIN] {os.path.basename(tp)}")
                n_extra = len(c["val_images"]) + len(c["train_images"]) - 10
                if n_extra > 0:
                    print(f"      ... +{n_extra} more images")

    # ── 6. Save CSV (optional) ────────────────────────────────────────────
    if args.output_csv and matches:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["val_image", "train_image", "similarity"]
            )
            writer.writeheader()
            for m in matches:
                writer.writerow(
                    {
                        "val_image": os.path.basename(m["val_image"]),
                        "train_image": os.path.basename(m["train_image"]),
                        "similarity": f"{m['similarity']:.4f}",
                    }
                )
        print(f"\n  Match details saved to: {args.output_csv}")

    # ── Summary stats ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Train embeddings extracted: {len(train_embs)}/{len(train_paths)}")
    print(f"  Val embeddings extracted:   {len(val_embs)}/{len(val_paths)}")
    print(f"  Similarity threshold:       {args.threshold}")
    print(f"  Pairwise matches found:     {len(matches)}")
    if matches:
        sims = [m["similarity"] for m in matches]
        print(f"  Max similarity:             {max(sims):.4f}")
        print(f"  Mean match similarity:      {np.mean(sims):.4f}")
    if leaked_uuids:
        print(f"  UUID-level leaks:           {len(leaked_uuids)}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
