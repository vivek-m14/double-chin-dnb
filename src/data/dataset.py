import csv
import hashlib
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
import albumentations as A
import yaml
from src.blend.blend_map import process_target_blend_map, compute_target_blend_map

import json


class BlendMapDataset(Dataset):
    """
    Dataset for skin retouching with original images, blend maps, and ground truth images.

    Blend maps are computed on-the-fly from (original, gt) pairs using the
    Linear Light formula: blend = (gt - original + 1) / 2.  This enables
    color augmentations (jitter, grayscale) that would be inconsistent with
    pre-cached blend maps.

    Data **filtering** (source exclusion, SSIM-tier exclusion) is intentionally
    NOT done here — see :func:`create_data_loaders` which curates ``data_map``
    before handing it to this class.

    Args:
        data_root (str): Root directory containing image data
        data_json (str): Path to JSON manifest listing image pairs.
            Ignored when *data_map* is provided.
        data_map (list[dict]): Pre-filtered list of manifest entries.  When
            supplied, *data_json* is not loaded — this avoids duplicate
            JSON parsing and keeps all filtering in one place.
        resize_dim (tuple): Dimensions to resize images to (width, height)
        test (bool): If True, limit to first 100 samples
        augment (bool): If True, apply geometric + color augmentations
    """
    ORIGINAL_IMAGE = "original_image"
    GT_IMAGE = "edited_image"

    def __init__(self, data_root, data_json="", resize_dim=(1024, 1024), test=False, augment=True,
                 data_map=None, **kwargs):
        if kwargs:
            import warnings
            warnings.warn(
                f"BlendMapDataset received unknown kwargs: {sorted(kwargs)}. "
                f"Data filtering (ssim_csv, exclude_sources, etc.) has moved to "
                f"create_data_loaders — pass these in the config dict instead.",
                stacklevel=2,
            )
        self.data_root = data_root
        self.resize_dim = resize_dim
        self.augment = augment

        if data_map is not None:
            # Use pre-filtered data_map supplied by create_data_loaders
            data_map = list(data_map)
        elif data_json:
            with open(data_json) as f:
                data_map = json.load(f)
        else:
            raise ValueError(
                "BlendMapDataset requires either `data_map` (preferred, from "
                "create_data_loaders) or a valid `data_json` path."
            )

        if test:
            data_map = data_map[:100]

        print(f"Length of data_map: {len(data_map)}")
        self.data_map = data_map

        # Build Albumentations pipeline (applied identically to image & gt)
        if augment:
            self.aug = A.Compose([
                # ── Geometric ──
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.OneOf([
                    A.Rotate(limit=(90, 90), p=1),
                    A.Rotate(limit=(180, 180), p=1),
                    A.Rotate(limit=(45, 45), p=1),
                ], p=0.2),

                # ── Color / tone ──
                A.ColorJitter(
                    brightness=(0.6, 1.4),
                    contrast=(0.6, 1.4),
                    saturation=(0.6, 1.4),
                    hue=(-0.1, 0.1),
                    p=0.3,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.2,
                ),
                A.RandomToneCurve(scale=0.15, p=0.2),
                A.ToGray(p=0.1),

                # ── Degradation (simulate real-world compression) ──
                A.ImageCompression(quality_lower=40, quality_upper=95, p=0.3),
            ], additional_targets={'gt': 'image'})
        else:
            self.aug = None

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx, retry=0):
        img_path = gt_path = "unknown"
        try:
            data_item = self.data_map[idx]

            img_path = data_item[self.ORIGINAL_IMAGE]
            gt_path = data_item[self.GT_IMAGE]

            img_name = img_path.split('/')[-1]

            img_path = os.path.join(self.data_root, img_path)
            gt_path = os.path.join(self.data_root, gt_path)

            # Load images as uint8 HWC RGB (Albumentations expects this)
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.resize_dim)

            gt = cv2.imread(gt_path)
            if gt is None:
                raise FileNotFoundError(f"Failed to load image: {gt_path}")
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = cv2.resize(gt, self.resize_dim)

            # Augment (same random transform on both image & gt)
            if self.aug is not None:
                augmented = self.aug(image=image, gt=gt)
                image = augmented['image']
                gt = augmented['gt']

            # Normalize to [0, 1] float32 and convert to CHW tensors
            image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
            gt = torch.from_numpy(gt.astype(np.float32) / 255.0).permute(2, 0, 1)

            # Compute blend map on-the-fly from (augmented) pair
            blend_map = compute_target_blend_map(image, gt)

            return {
                'image': image,
                'blend_map': blend_map,
                'gt': gt,
                'filename': img_name
            }
        except Exception as e:
            if retry > 3:
                raise e
            with open('error_log1.txt', 'a') as f:
                f.write(f"Error in reading {img_path}, {gt_path}: {e}\n")
            print(f"Error in reading {img_path}, {gt_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)), retry=retry+1)


class TensorMapDataset(Dataset):
    """
    Dataset for tensor map generation.
    
    Args:
        data_root (str): Directory containing original images
        data_json (str): Path to the json file containing the data
        transform (callable, optional): Optional transform to be applied on a sample
    """
    def __init__(self, data_root, data_json, transform=None, resize_dim=(1024, 1024)):
        self.data_root = data_root
        data_json = data_json

        self.tensor_map_type = 'npy'

        with open(data_json) as f:
            self.data_map = json.load(f)

        self.transform = transform
        self.resize_dim = resize_dim
        
    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):

        try:
            data_item = self.data_map[idx]

            img_path = data_item['original_image']
            tensor_map_path = data_item['flow_map']
            gt_path = data_item['edited_image']

            img_name = img_path.split('/')[-2]
            
            img_path = os.path.join(self.data_root, img_path)
            tensor_map_path = os.path.join(self.data_root, tensor_map_path)
            gt_path = os.path.join(self.data_root, gt_path)

            # Load images
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.resize_dim)
            image = image.astype(np.float32) / 255.0

            # Load tensor map
            if self.tensor_map_type == 'npy':
                tensor_map = np.load(tensor_map_path)
                tensor_map = process_target_blend_map(tensor_map)
            elif self.tensor_map_type == 'png':
                tensor_map = cv2.imread(tensor_map_path)
                tensor_map = cv2.cvtColor(tensor_map, cv2.COLOR_BGR2RGB)
            tensor_map = cv2.resize(tensor_map, self.resize_dim)
            tensor_map = tensor_map.astype(np.float32)

            # Load ground truth image
            gt = cv2.imread(gt_path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = cv2.resize(gt, self.resize_dim)
            gt = gt.astype(np.float32) / 255.0

            # Convert to PyTorch tensors
            if self.transform:
                image = self.transform(image)
                tensor_map = self.transform(tensor_map)  
                gt = self.transform(gt)
            else:
                image = torch.from_numpy(image).permute(2, 0, 1)
                tensor_map = torch.from_numpy(tensor_map).permute(2, 0, 1)  # Now a 3-channel tensor
                gt = torch.from_numpy(gt).permute(2, 0, 1)

            # image, tensor_map, gt = self.apply_augmentation(image, tensor_map, gt)

            return {
                'image': image,
                'tensor_map': tensor_map,
                'gt': gt,
                'filename': img_name
            }
        except Exception as e:
            with open('error_log1.txt', 'a') as f:
                f.write(f"Error in reading {img_path}, {tensor_map_path}, {gt_path}: {e}\n")
            return self.__getitem__(np.random.randint(0, len(self)))

    def apply_augmentation(self, image, tensor_map, gt):
        """Apply the same augmentation to all three components"""
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            image = TF.hflip(image)
            tensor_map = TF.hflip(tensor_map)
            gt = TF.hflip(gt)

        static_angle = 0
        if torch.rand(1) < 0.2:
            angles = [90, 180, 45]
            static_angle = angles[torch.randint(0, len(angles), (1,)).item()]

        # Range rotation (e.g., -30 to 30 degrees)
        if torch.rand(1) < 0.5:
            rotation_range = 30 + static_angle
            angle = torch.randint(-rotation_range, rotation_range, (1,)).item()
            image = TF.rotate(image, angle, expand=False)
            gt = TF.rotate(gt, angle, expand=False)
            tensor_map = TF.rotate(tensor_map, angle, fill=(0, 0), expand=False)
            
        return image, tensor_map, gt


# ---------------------------------------------------------------------------
# Versioning / fingerprinting
# ---------------------------------------------------------------------------

def _file_sha256(path: str) -> str:
    """Return the hex SHA-256 digest of a file (first 12 chars for brevity)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def compute_data_fingerprint(
    args: dict,
    n_after_filter: int,
    n_train: int,
    n_val: int,
) -> dict:
    """Build a reproducibility fingerprint for the data pipeline.

    Captures the exact files and parameters used to construct the
    train/val split so that a future run (or a checkpoint resume) can
    verify it is using the same data.

    The fingerprint is stored in ``args['data_fingerprint']`` and
    persisted alongside the training config (``config.yaml``).
    """
    fp: dict = {
        "data_json_sha256": _file_sha256(args["data_json"]),
        "exclude_sources": sorted(args.get("exclude_sources") or []),
    }

    ssim_csv = args.get("ssim_csv", "")
    if ssim_csv and os.path.isfile(ssim_csv):
        fp["ssim_csv_sha256"] = _file_sha256(ssim_csv)
        fp["ssim_exclude_tiers"] = sorted(
            t.lower().strip()
            for t in (args.get("ssim_exclude_tiers") or [])
        )
    else:
        fp["ssim_csv_sha256"] = None
        fp["ssim_exclude_tiers"] = []

    id_map_path = args.get("identity_map", "")
    if id_map_path and os.path.isfile(id_map_path):
        fp["identity_map_sha256"] = _file_sha256(id_map_path)
    else:
        fp["identity_map_sha256"] = None

    fp["seed"] = args.get("seed", 42)
    fp["train_ratio"] = args.get("train_ratio", 0.9)
    fp["n_after_filter"] = n_after_filter
    fp["n_train"] = n_train
    fp["n_val"] = n_val

    return fp


def validate_data_fingerprint(args: dict) -> None:
    """Warn if the current data files differ from a previous fingerprint.

    Called on **resume** when ``args['data_fingerprint']`` already exists
    (loaded from the saved ``config.yaml``).  If the SHA-256 of a file
    has changed, a warning is printed so the user knows the data has
    drifted.
    """
    old = args.get("data_fingerprint")
    if not old:
        return  # no previous fingerprint → nothing to validate

    checks = [
        ("data_json", "data_json_sha256"),
        ("ssim_csv", "ssim_csv_sha256"),
        ("identity_map", "identity_map_sha256"),
    ]
    for path_key, hash_key in checks:
        path = args.get(path_key, "")
        old_hash = old.get(hash_key)
        if not path or not old_hash or not os.path.isfile(path):
            continue
        current_hash = _file_sha256(path)
        if current_hash != old_hash:
            print(
                f"  ⚠ DATA DRIFT: {path_key} hash changed! "
                f"saved={old_hash}  current={current_hash}  "
                f"(file: {path})"
            )


def _load_identity_map(path: str) -> dict:
    """Load identity_map.json produced by ``tools/build_identity_map.py``."""
    with open(path, "r") as f:
        data = json.load(f)
    # Validate schema version
    version = data.get("metadata", {}).get("schema_version", 0)
    if version < 2:
        raise ValueError(
            f"identity_map at {path} has schema_version={version} (expected ≥2). "
            f"Re-run tools/build_identity_map.py to regenerate."
        )
    return data


def _identity_aware_split(
    data_map: list,
    identity_map: dict,
    image_key: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple:
    """
    Split *data_map* indices into train / val such that **no identity
    appears in both splits**.

    Algorithm
    ---------
    1. Map every sample index → its identity_id (via *identity_map*).
    2. Group indices by identity_id.
    3. Permute identity groups (seeded).
    4. Greedily assign groups to train until we reach *train_ratio*.
    5. Remaining groups → val.

    Returns (train_indices, val_indices).
    """
    img2id = identity_map["image_to_identity"]

    # index → identity_id
    idx_to_identity: dict = {}
    unmapped = 0
    for idx, entry in enumerate(data_map):
        rel = entry.get(image_key, "")
        if rel in img2id:
            idx_to_identity[idx] = img2id[rel]
        else:
            # Not in identity map → assign unique placeholder
            idx_to_identity[idx] = f"__unmapped_{idx}"
            unmapped += 1

    if unmapped:
        print(f"  ⚠ {unmapped}/{len(data_map)} samples not in identity map (each assigned own identity)")

    # Group indices by identity
    identity_to_indices: dict = {}
    for idx, ident in idx_to_identity.items():
        identity_to_indices.setdefault(ident, []).append(idx)

    identity_ids = sorted(identity_to_indices.keys())
    n_identities = len(identity_ids)
    n_samples = len(data_map)
    target_train = int(n_samples * train_ratio)

    # Seeded permutation of identity groups
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_identities, generator=gen).tolist()

    train_indices = []
    val_indices = []
    for pi in perm:
        ident = identity_ids[pi]
        group = identity_to_indices[ident]
        if len(train_indices) + len(group) <= target_train:
            train_indices.extend(group)
        else:
            # If adding this group overshoots, still add if train is very
            # under-filled (within one group-size tolerance).  Otherwise → val.
            if len(train_indices) < target_train:
                train_indices.extend(group)
            else:
                val_indices.extend(group)

    actual_ratio = len(train_indices) / n_samples if n_samples else 0
    print(f"  Identity-aware split: {n_identities} identities → "
          f"{len(train_indices)} train ({actual_ratio:.1%}), "
          f"{len(val_indices)} val ({1-actual_ratio:.1%})")

    return train_indices, val_indices


def _load_ssim_exclusions(ssim_csv: str, exclude_tiers: set) -> set:
    """Return set of original-filename stems to exclude based on SSIM tier.

    The CSV is produced by ``tools/analyze_ssim.py`` and must contain
    ``original_filename`` and ``ssim_tier`` columns.  The join key is
    ``os.path.splitext(original_filename)[0]`` which matches the JSON
    manifest's ``os.path.splitext(os.path.basename(original_image))[0]``.
    """
    excluded: set = set()
    with open(ssim_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ssim_tier"].lower().strip() in exclude_tiers:
                stem = os.path.splitext(row["original_filename"])[0]
                excluded.add(stem)
    return excluded


def _filter_data_map(
    data_map: list,
    exclude_sources: list | None = None,
    ssim_csv: str = "",
    ssim_exclude_tiers: list | None = None,
    image_key: str = "original_image",
) -> list:
    """Apply all data-curation filters to *data_map* (in order).

    1. **Source exclusion** — drop entries whose ``source`` is in
       *exclude_sources*.
    2. **SSIM-tier exclusion** — drop entries whose SSIM tier (from
       ``tools/analyze_ssim.py``) is in *ssim_exclude_tiers*.

    Returns the filtered list (never mutates the input).
    """
    # ── 1. Source filter ──────────────────────────────────────────────
    if exclude_sources:
        src_set = set(exclude_sources)
        before = len(data_map)
        data_map = [e for e in data_map if e.get("source") not in src_set]
        print(f"Source filter: excluded {before - len(data_map)}/{before} "
              f"pairs (sources={sorted(src_set)})")

    # ── 2. SSIM tier filter ───────────────────────────────────────────
    if ssim_csv and ssim_exclude_tiers:
        if not os.path.isfile(ssim_csv):
            raise FileNotFoundError(
                f"ssim_csv not found: {ssim_csv}. "
                f"Run tools/analyze_ssim.py first, or set ssim_csv='' to disable."
            )
        tier_set = set(t.lower().strip() for t in ssim_exclude_tiers)
        excluded_stems = _load_ssim_exclusions(ssim_csv, tier_set)
        before = len(data_map)
        data_map = [
            e for e in data_map
            if os.path.splitext(os.path.basename(e[image_key]))[0]
               not in excluded_stems
        ]
        print(f"SSIM filter: excluded {before - len(data_map)}/{before} "
              f"pairs (tiers={sorted(tier_set)})")

    return data_map


def create_data_loaders(args, world_size=None, rank=None, dataset_type='blend_map', test=False):
    """
    Create data loaders for training and testing.

    Applies data-curation filters (source exclusion, SSIM-tier exclusion)
    **before** constructing datasets, then performs an identity-aware
    train/val split (if ``args['identity_map']`` is set).

    Filter pipeline (in order):
        1. Exclude sources listed in ``args['exclude_sources']``
        2. Exclude SSIM tiers listed in ``args['ssim_exclude_tiers']``
        3. Identity-aware (or random) train/val split
    
    Args:
        args (dict): Arguments containing dataset paths and parameters
        world_size (int, optional): Number of processes for distributed training
        rank (int, optional): Process rank for distributed training
        
    Returns:
        tuple: (train_loader, test_loader, train_sampler, test_sampler)
    """
    # ------------------------------------------------------------------
    # 1. Load JSON manifest and apply curation filters
    # ------------------------------------------------------------------
    with open(args['data_json']) as f:
        raw_data_map = json.load(f)
    print(f"JSON manifest loaded: {len(raw_data_map)} entries")

    data_map = _filter_data_map(
        raw_data_map,
        exclude_sources=args.get('exclude_sources'),
        ssim_csv=args.get('ssim_csv', ''),
        ssim_exclude_tiers=args.get('ssim_exclude_tiers', None),
        image_key=BlendMapDataset.ORIGINAL_IMAGE,
    )
    print(f"After filtering: {len(data_map)} entries")

    # ------------------------------------------------------------------
    # 2. Create dataset(s) with the curated data_map
    # ------------------------------------------------------------------
    common_kwargs = dict(
        data_root=args['data_root'],
        data_map=data_map,
        resize_dim=(args['img_size'], args['img_size']),
        test=test,
    )

    if dataset_type == 'blend_map':
        train_ds_full = BlendMapDataset(**common_kwargs, augment=True)
        # Share the same data_map — both datasets see the same entries,
        # only augmentation differs.  Avoids duplicating the manifest in memory.
        val_ds_full   = BlendMapDataset(**common_kwargs, augment=False)
        val_ds_full.data_map = train_ds_full.data_map
    elif dataset_type == 'tensor_map':
        # Legacy flow-based path — still uses the filtered data_map
        dataset = TensorMapDataset(
            args['data_root'], args['data_json'],
            transform=ToTensor(),
            resize_dim=(args['img_size'], args['img_size']),
        )
        dataset.data_map = data_map   # apply the same curation filters
        train_ds_full = dataset
        val_ds_full   = dataset
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    n = len(train_ds_full)
    train_ratio = args.get("train_ratio", 0.9)
    identity_map_path = args.get("identity_map", "")

    if identity_map_path and os.path.isfile(identity_map_path):
        # ── Identity-aware split (leak-free) ──────────────────────────
        print(f"Using identity-aware split from: {identity_map_path}")
        id_map = _load_identity_map(identity_map_path)

        # Validate the identity map isn't stale / partial
        meta = id_map.get("metadata", {})
        if meta.get("partial_map"):
            raise ValueError(
                f"identity_map was built with --max_images (partial). "
                f"Re-run tools/build_identity_map.py without --max_images for production."
            )
        map_total = meta.get("total_images", 0)
        if map_total > 0 and abs(map_total - n) > 0.05 * n:
            print(
                f"  ⚠ identity_map covers {map_total} images but dataset has {n}. "
                f"Consider rebuilding the identity map."
            )

        image_key = BlendMapDataset.ORIGINAL_IMAGE
        train_indices, val_indices = _identity_aware_split(
            data_map=train_ds_full.data_map,
            identity_map=id_map,
            image_key=image_key,
            train_ratio=train_ratio,
            seed=args.get("seed", 42),
        )
    else:
        # ── Legacy random split (kept for backward compat) ────────────
        if identity_map_path:
            print(f"⚠ identity_map path not found: {identity_map_path} — falling back to random split")
        else:
            print("⚠ No identity_map configured — using legacy random split (may have identity leakage)")

        train_size = int(n * train_ratio)
        split_gen = torch.Generator().manual_seed(args.get("seed", 42))
        perm = torch.randperm(n, generator=split_gen).tolist()
        train_indices = perm[:train_size]
        val_indices   = perm[train_size:]

    train_dataset = Subset(train_ds_full, train_indices)
    test_dataset  = Subset(val_ds_full,   val_indices)

    # ------------------------------------------------------------------
    # 4. Data fingerprint (reproducibility)
    # ------------------------------------------------------------------
    # On resume the saved config already has a fingerprint → validate
    validate_data_fingerprint(args)
    # Compute (or overwrite) the fingerprint for the current run
    args['data_fingerprint'] = compute_data_fingerprint(
        args,
        n_after_filter=len(data_map),
        n_train=len(train_indices),
        n_val=len(val_indices),
    )
    print(f"Data fingerprint: {args['data_fingerprint']}")

    # Create samplers for distributed training if world_size and rank are provided
    if world_size is not None and rank is not None:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    
    # Create data loaders
    num_workers = args.get('num_workers', 4)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args['batch_size'], 
        sampler=train_sampler, 
        num_workers=num_workers,
        shuffle=(train_sampler is None),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args['batch_size'], 
        sampler=test_sampler, 
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    
    return train_loader, test_loader, train_sampler, test_sampler

if __name__ == '__main__':
    config = yaml.load(open('blend_map.yaml'), Loader=yaml.FullLoader)
    train_loader, test_loader, train_sampler, test_sampler = create_data_loaders(config, dataset_type='blend_map')
    
    print(len(train_loader))
    for batch in train_loader:
        print(batch)
        break