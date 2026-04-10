#!/usr/bin/env python3
"""
Build Face Identity Map
========================
Scans **all** images in the dataset, extracts face embeddings, and clusters
them into unique identities.  The resulting ``identity_map.json`` is consumed
by ``create_data_loaders`` to produce a **leak-free** train/val split.

Pipeline
--------
1.  Load dataset JSON (same filter as training).
2.  Extract 512-d ArcFace embeddings for every image (InsightFace buffalo_l).
3.  Pairwise cosine similarity across **all** image embeddings → merge
    pairs whose similarity exceeds ``--merge_threshold`` using Union-Find.
    Pairs are processed in descending similarity order.
    Guarded by ``--max_cluster_size`` to prevent transitive chain explosions.
4.  Build identity clusters from Union-Find groups.
    Images with no detected face become singleton identities.
5.  Integrity verification (every image mapped to exactly one identity).
6.  Emit ``identity_map.json`` ready for identity-aware splitting.

Usage
-----
    python tools/build_identity_map.py \\
        --data_root /path/to/dataset_root \\
        --data_json /path/to/full_dataset.json \\
        --output   identity_map.json \\
        --workers 4

Install
-------
    pip install insightface onnxruntime numpy torch opencv-python
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# InsightFace helpers (identical logic to check_face_leakage.py)
# ---------------------------------------------------------------------------


def _build_face_app(det_size: int = 640):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def _pad_image(img: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """Add a neutral-gray border around *img* (each side = factor × dim).

    Neutral gray (128,128,128) avoids strong contrast edges that confuse the
    detector more than black/white padding would.
    """
    h, w = img.shape[:2]
    pad_h, pad_w = int(h * factor), int(w * factor)
    return cv2.copyMakeBorder(
        img, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT, value=(128, 128, 128),
    )


# Padding multipliers tried in sequence when the detector fails on the
# original image.  Each value is the per-side fraction of the image size.
_PAD_FACTORS = (0.5, 1.0)


def _extract_embedding(image_path: str, app) -> Optional[np.ndarray]:
    """Return 512-d ArcFace embedding for the largest detected face.

    If the detector fails on the original image (common for tight face crops),
    retry with progressively larger neutral-gray padding.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    # --- first try: original image ---
    faces = app.get(img)
    if faces:
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best.normed_embedding  # L2-normalized, float32, (512,)

    # --- retry with padding (tight-crop recovery) ---
    for factor in _PAD_FACTORS:
        padded = _pad_image(img, factor)
        faces = app.get(padded)
        if faces:
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return best.normed_embedding

    return None


# -- multiprocessing plumbing --

_WORKER_APP = None


def _worker_init(det_size: int = 640):
    global _WORKER_APP
    os.environ["ORT_LOG_LEVEL"] = "3"
    _WORKER_APP = _build_face_app(det_size)


def _worker_extract(image_path: str):
    global _WORKER_APP
    try:
        emb = _extract_embedding(image_path, _WORKER_APP)
        return {"ok": True, "path": image_path, "emb": emb}
    except Exception as e:
        return {"ok": False, "path": image_path, "error": str(e)}


def extract_all_embeddings(
    image_paths: List[str],
    workers: int = 4,
    det_size: int = 640,
) -> Dict[str, np.ndarray]:
    """Extract embeddings for all *image_paths*. Returns path → emb dict."""
    results: Dict[str, np.ndarray] = {}
    no_face: List[str] = []
    errors: List[str] = []
    total = len(image_paths)
    t0 = time.time()

    if workers > 1 and total > 1:
        processed = 0
        # Use 'spawn' context to avoid fork+ONNX Runtime deadlocks on macOS
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, _worker_init, (det_size,)) as pool:
            for res in pool.imap_unordered(_worker_extract, image_paths, chunksize=8):
                processed += 1
                if res["ok"]:
                    if res["emb"] is not None:
                        results[res["path"]] = res["emb"]
                    else:
                        no_face.append(res["path"])
                else:
                    errors.append(res["path"])
                if processed % 200 == 0 or processed == total:
                    elapsed = time.time() - t0
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"  [{processed:5d}/{total}]  {len(results)} faces  ({rate:.1f} img/s)")
    else:
        app = _build_face_app(det_size)
        for i, path in enumerate(image_paths):
            try:
                emb = _extract_embedding(path, app)
                if emb is not None:
                    results[path] = emb
                else:
                    no_face.append(path)
            except Exception as e:
                errors.append(path)
                print(f"  ✗ Error: {os.path.basename(path)} — {e}")
            if (i + 1) % 200 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1:5d}/{total}]  {len(results)} faces  ({rate:.1f} img/s)")

    elapsed = time.time() - t0
    print(
        f"\n  Done: {len(results)} embeddings, "
        f"{len(no_face)} no-face, {len(errors)} errors  ({elapsed:.1f}s)\n"
    )
    return results


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class UnionFind:
    """Lightweight Union-Find with path compression and union by rank."""

    def __init__(self):
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1


# ---------------------------------------------------------------------------
# Core: build identity map
# ---------------------------------------------------------------------------


def build_identity_map(
    data_map: List[dict],
    data_root: str,
    image_key: str = "original_image",
    merge_threshold: float = 0.55,
    workers: int = 4,
    det_size: int = 640,
    max_images: int = 0,
    exclude_source: str = "",
    max_cluster_size: int = 20,
    data_json_path: str = "",
) -> dict:
    """
    Build an identity map from the dataset.

    Clustering is based **purely on face-embedding similarity** — no
    filename heuristics (e.g. UUIDs) are used.

    Parameters
    ----------
    merge_threshold : float
        Cosine similarity above which two images are considered the same
        person and merged.
    max_cluster_size : int
        Safety cap: refuse to merge two clusters if the resulting
        cluster would exceed this many images.  Prevents transitive
        chain reactions (A↔B↔C↔...) from creating mega-clusters of
        unrelated people.  Set to 0 to disable.

    Returns a dict with keys:
        metadata, identities, image_to_identity, diagnostics
    """
    # ------------------------------------------------------------------
    # 1. Resolve image paths
    # ------------------------------------------------------------------
    print("── Step 1: Resolving image paths ──")
    rel_paths: List[str] = []      # relative path (key in JSON)
    abs_paths: List[str] = []      # absolute path for loading
    missing = 0

    for entry in data_map:
        rel = entry.get(image_key, "")
        full = os.path.join(data_root, rel)
        if os.path.isfile(full):
            rel_paths.append(rel)
            abs_paths.append(full)
        else:
            missing += 1

    print(f"  Total entries: {len(data_map)}")
    print(f"  Images found:  {len(abs_paths)}")
    if missing:
        print(f"  ⚠ Missing:     {missing}")

    # Optional cap for smoke tests
    if max_images > 0 and len(abs_paths) > max_images:
        rng = np.random.RandomState(42)
        sel = sorted(rng.choice(len(abs_paths), max_images, replace=False))
        rel_paths = [rel_paths[i] for i in sel]
        abs_paths = [abs_paths[i] for i in sel]
        print(f"  (capped to {max_images} images for smoke test)")

    # ------------------------------------------------------------------
    # 2. Extract face embeddings
    # ------------------------------------------------------------------
    print("\n── Step 2: Extracting face embeddings ──")
    emb_map = extract_all_embeddings(abs_paths, workers=workers, det_size=det_size)

    # Build rel_path → embedding mapping  (use rel paths as canonical keys)
    abs_to_rel = {a: r for a, r in zip(abs_paths, rel_paths)}
    rel_emb: Dict[str, np.ndarray] = {}
    for abs_p, emb in emb_map.items():
        rel_emb[abs_to_rel[abs_p]] = emb

    images_with_face = sorted(rel_emb.keys())
    images_without_face = sorted(set(rel_paths) - set(images_with_face))
    print(f"  Images with face:    {len(images_with_face)}")
    print(f"  Images without face: {len(images_without_face)}")

    # ------------------------------------------------------------------
    # 3. Pairwise similarity → cluster all images with embeddings
    # ------------------------------------------------------------------
    print(f"\n── Step 3: Pairwise clustering (threshold={merge_threshold:.2f}) ──")

    uf = UnionFind()
    merge_count = 0
    blocked_merges = 0
    merge_pairs: List[Tuple[str, str, float]] = []  # for diagnostics

    n_emb = len(images_with_face)
    if n_emb > 1:
        # Build embedding matrix (N, 512)
        emb_matrix = np.array(
            [rel_emb[img].astype(np.float64) for img in images_with_face]
        )
        # Normalize (should already be, but be safe)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_matrix = emb_matrix / np.maximum(norms, 1e-10)

        # Cosine similarity matrix (N, N)
        print(f"  Computing {n_emb}×{n_emb} similarity matrix...")
        sim_matrix = emb_matrix @ emb_matrix.T

        # Find pairs above threshold (upper triangle only)
        rows, cols = np.where(np.triu(sim_matrix, k=1) >= merge_threshold)
        print(f"  Found {len(rows)} image pairs above threshold")

        # Sort by descending similarity so highest-confidence merges happen
        # first.  Prevents transitive chains from merging via weak links.
        pair_sims = [(float(sim_matrix[r, c]), r, c) for r, c in zip(rows, cols)]
        pair_sims.sort(key=lambda x: -x[0])

        # Track cluster sizes for the max_cluster_size guard
        _cluster_sizes: Dict[str, int] = {img: 1 for img in images_with_face}

        for sim_val, r, c in pair_sims:
            img_a = images_with_face[r]
            img_b = images_with_face[c]
            root_a, root_b = uf.find(img_a), uf.find(img_b)
            if root_a == root_b:
                continue  # already in same cluster

            # Transitive-chain safeguard
            combined_size = _cluster_sizes.get(root_a, 1) + _cluster_sizes.get(root_b, 1)
            if max_cluster_size > 0 and combined_size > max_cluster_size:
                blocked_merges += 1
                continue

            uf.union(img_a, img_b)
            new_root = uf.find(img_a)
            _cluster_sizes[new_root] = combined_size
            merge_count += 1
            merge_pairs.append((
                os.path.basename(img_a),
                os.path.basename(img_b),
                sim_val,
            ))

        if blocked_merges:
            print(f"  ⚠ {blocked_merges} merge(s) blocked by max_cluster_size={max_cluster_size}")

    print(f"  Merges performed: {merge_count}")

    # ------------------------------------------------------------------
    # 4. Build identity clusters
    # ------------------------------------------------------------------
    print("\n── Step 4: Building identity clusters ──")

    # Collect Union-Find clusters for images with embeddings
    cluster_to_images: Dict[str, List[str]] = {}
    for img in images_with_face:
        root = uf.find(img)
        cluster_to_images.setdefault(root, []).append(img)

    # Sort clusters by size (largest first)
    sorted_clusters = sorted(cluster_to_images.values(), key=lambda x: -len(x))

    # Assign identity IDs
    identities: List[dict] = []
    image_to_identity: Dict[str, str] = {}
    identity_counter = 0

    for cluster_images in sorted_clusters:
        identity_id = f"id_{identity_counter:05d}"
        identity_counter += 1

        identities.append({
            "identity_id": identity_id,
            "images": sorted(cluster_images),
            "image_count": len(cluster_images),
            "has_embedding": True,
        })

        for img in cluster_images:
            image_to_identity[img] = identity_id

    # Images without faces → each becomes its own singleton identity
    for img in images_without_face:
        identity_id = f"id_{identity_counter:05d}"
        identity_counter += 1
        identities.append({
            "identity_id": identity_id,
            "images": [img],
            "image_count": 1,
            "has_embedding": False,
        })
        image_to_identity[img] = identity_id

    # ------------------------------------------------------------------
    # 5. Compute diagnostics & verify integrity
    # ------------------------------------------------------------------
    print("\n── Step 5: Verification & diagnostics ──")

    cluster_sizes = [ident["image_count"] for ident in identities]
    multi_image_clusters = [i for i in identities if i["image_count"] > 1]
    large_clusters = [i for i in identities if i["image_count"] > 10]

    diagnostics = {
        "merge_pairs": [
            {"image_a": a, "image_b": b, "similarity": round(s, 4)}
            for a, b, s in sorted(merge_pairs, key=lambda x: -x[2])
        ],
        "large_clusters": [
            {"identity_id": i["identity_id"], "image_count": i["image_count"]}
            for i in large_clusters
        ],
    }

    print(f"  Total identities:             {len(identities)}")
    print(f"  Multi-image identities:       {len(multi_image_clusters)}")
    print(f"  Singleton identities (face):  {sum(1 for i in identities if i['image_count'] == 1 and i['has_embedding'])}")
    print(f"  Singleton identities (no face): {sum(1 for i in identities if not i['has_embedding'])}")
    if cluster_sizes:
        print(f"  Cluster size — min: {min(cluster_sizes)}, "
              f"max: {max(cluster_sizes)}, "
              f"mean: {np.mean(cluster_sizes):.1f}, "
              f"median: {np.median(cluster_sizes):.1f}")
    if large_clusters:
        print(f"  ⚠ {len(large_clusters)} cluster(s) with >10 images (review for false merges)")

    # Verification: every image must be in exactly one identity
    mapped_images = set(image_to_identity.keys())
    all_images = set(rel_paths)
    unmapped = all_images - mapped_images
    double_mapped = len(image_to_identity) - len(mapped_images)  # should be 0
    integrity_ok = True
    if unmapped:
        print(f"  🔴 BUG: {len(unmapped)} images not mapped to any identity!")
        integrity_ok = False
    if double_mapped:
        print(f"  🔴 BUG: {double_mapped} images mapped to multiple identities!")
        integrity_ok = False
    if integrity_ok:
        print(f"  ✅ All {len(all_images)} images mapped to exactly one identity.")

    # ------------------------------------------------------------------
    # 6. Assemble output
    # ------------------------------------------------------------------
    no_face_singletons = sum(
        1 for i in identities if not i["has_embedding"]
    )

    metadata = {
        "schema_version": 2,
        "total_images": len(rel_paths),
        "total_entries_in_json": len(data_map),
        "images_with_face": len(images_with_face),
        "images_without_face": len(images_without_face),
        "no_face_singletons": no_face_singletons,
        "unique_identities": len(identities),
        "multi_image_identities": len(multi_image_clusters),
        "merges_performed": merge_count,
        "merges_blocked": blocked_merges,
        "merge_threshold": merge_threshold,
        "max_cluster_size": max_cluster_size,
        "model": "buffalo_l (ArcFace 512-d)",
        "det_size": det_size,
        "image_key": image_key,
        "exclude_source": exclude_source,
        "data_root": data_root,
        "data_json": data_json_path,
        "partial_map": max_images > 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "metadata": metadata,
        "identities": identities,
        "image_to_identity": image_to_identity,
        "diagnostics": diagnostics,
        "integrity_ok": integrity_ok,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a face identity map for the dataset. "
            "Output is consumed by create_data_loaders for identity-aware splitting."
        )
    )
    parser.add_argument("--data_root", required=True, help="Dataset root directory.")
    parser.add_argument("--data_json", required=True, help="Dataset JSON file.")
    parser.add_argument(
        "--output", default="identity_map.json",
        help="Output JSON path (default: identity_map.json).",
    )
    parser.add_argument(
        "--merge_threshold", type=float, default=0.55,
        help=(
            "Cosine similarity threshold for merging image clusters. "
            "Higher = fewer merges, more conservative. "
            "0.55 is a good balance; 0.6+ is very conservative. (default: 0.55)"
        ),
    )
    parser.add_argument(
        "--image_key", default="original_image",
        help="JSON key for the image path (default: original_image).",
    )
    parser.add_argument(
        "--exclude_source", default="aadan_double_chin",
        help="Source to exclude (same filter as training).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--det_size", type=int, default=640, help="Face detection input size.")
    parser.add_argument(
        "--max_images", type=int, default=0,
        help="Cap images for smoke testing (0 = all).",
    )
    parser.add_argument(
        "--max_cluster_size", type=int, default=20,
        help=(
            "Safety cap: refuse to merge clusters if the resulting "
            "cluster exceeds this many images. Prevents transitive merge "
            "chains from creating mega-clusters. 0 = no limit. (default: 20)"
        ),
    )
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  BUILD FACE IDENTITY MAP")
    print(f"{'='*70}")
    print(f"  data_root:          {args.data_root}")
    print(f"  data_json:          {args.data_json}")
    print(f"  merge_threshold:    {args.merge_threshold}")
    print(f"  max_cluster_size:   {args.max_cluster_size}")
    print(f"  output:             {args.output}")
    print(f"{'='*70}\n")

    # Load & filter
    with open(args.data_json) as f:
        data_map = json.load(f)
    if args.exclude_source:
        before = len(data_map)
        data_map = [d for d in data_map if d.get("source") != args.exclude_source]
        print(f"Filtered '{args.exclude_source}': {before} → {len(data_map)} entries\n")

    result = build_identity_map(
        data_map=data_map,
        data_root=args.data_root,
        image_key=args.image_key,
        merge_threshold=args.merge_threshold,
        workers=args.workers,
        det_size=args.det_size,
        max_images=args.max_images,
        exclude_source=args.exclude_source,
        max_cluster_size=args.max_cluster_size,
        data_json_path=args.data_json,
    )

    if not result["integrity_ok"]:
        print("\n🔴 Integrity check FAILED — identity map NOT saved.")
        sys.exit(1)

    # Save (strip internal flag before writing)
    output_data = {k: v for k, v in result.items() if k != "integrity_ok"}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✅ Identity map saved to: {args.output}")

    # Print summary
    meta = result["metadata"]
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total images:           {meta['total_images']}")
    print(f"  Images with face:       {meta['images_with_face']}")
    print(f"  Images without face:    {meta['images_without_face']}")
    print(f"  Unique identities:      {meta['unique_identities']}")
    print(f"  Multi-image identities: {meta['multi_image_identities']}")
    print(f"  Merges performed:       {meta['merges_performed']}")
    print(f"  Merges blocked:         {meta['merges_blocked']}")
    print(f"  Merge threshold:        {meta['merge_threshold']}")
    print(f"{'='*70}\n")

    # Print top merge pairs for review
    diag = result["diagnostics"]
    if diag["merge_pairs"]:
        print("  ── Top merges (review for correctness) ──")
        for mp in diag["merge_pairs"][:20]:
            print(f"    sim={mp['similarity']:.4f}  {mp['image_a']}  ↔  {mp['image_b']}")
        if len(diag["merge_pairs"]) > 20:
            print(f"    ... and {len(diag['merge_pairs']) - 20} more")

    if diag["large_clusters"]:
        print(f"\n  ── Large clusters (>10 images, review for false merges) ──")
        for lc in diag["large_clusters"][:10]:
            print(f"    {lc['identity_id']}: {lc['image_count']} images")


if __name__ == "__main__":
    main()
