"""Leakage-aware, reproducible MRI dataset splits.

Images, not file paths, are the unit of comparison. When available, explicit
patient IDs take precedence; perceptual matching additionally groups scans
that are near-identical across Kaggle datasets and file formats.

This is *not* proof of patient independence for sources without patient IDs.
The similarity rule is deliberately conservative and its thresholds must not
be tuned using downstream test performance.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from sklearn.model_selection import StratifiedGroupKFold


# Thresholds fixed without using any model's test accuracy.
PHASH_MAX_DISTANCE = 4
MIN_GRAY_CORRELATION = 0.98
SEED = 42


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _read_mri(path):
    """Match the preprocessing of the two training scripts before resizing."""
    path = Path(path)
    if path.suffix.lower() == ".mat":
        import h5py

        with h5py.File(path, "r") as file:
            arr = np.asarray(file["cjdata"]["image"], dtype=np.float32)
            pid_chars = np.asarray(file["cjdata"]["PID"]).reshape(-1)
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ValueError(f"Expected finite 2D MRI pixels: {path}")
        pid = "".join(chr(int(c)) for c in pid_chars if int(c)).strip()
        if not pid:
            raise ValueError(f"Missing Figshare patient identifier: {path}")
        low, high = float(arr.min()), float(arr.max())
        arr = (arr - low) / (high - low) if high > low else np.zeros_like(arr)
        image = Image.fromarray(np.uint8(np.clip(arr * 255.0, 0, 255))).convert("RGB")
        patient_id = "figshare:" + pid
    else:
        with Image.open(path) as file:
            image = file.convert("RGB")
        patient_id = None
    return image, patient_id


def _signature(path):
    image, patient_id = _read_mri(path)
    # Detect identical pixels even if JPG/PNG containers have different bytes.
    model_input = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR))
    pixel_digest = hashlib.sha256(model_input.tobytes()).hexdigest()

    gray = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32
    )
    coeff = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].ravel()[1:]
    median = np.median(coeff)
    phash = sum(int(value) << bit for bit, value in enumerate(coeff > median))

    normalized = np.asarray(
        image.convert("L").resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32
    ).ravel()
    normalized -= normalized.mean()
    norm = np.linalg.norm(normalized)
    normalized /= max(float(norm), 1e-6)
    return pixel_digest, phash, normalized, patient_id


def _group_near_duplicates(paths):
    """Connect exact/near-exact images and explicit Figshare patient IDs."""
    uf = UnionFind(len(paths))
    pixel_seen = {}
    patient_seen = {}
    bands = defaultdict(list)
    feature_vectors = []
    exact_links = 0
    near_links = 0
    patient_links = 0

    for idx, path in enumerate(paths):
        digest, phash, vector, patient_id = _signature(path)
        if digest in pixel_seen:
            uf.union(idx, pixel_seen[digest])
            exact_links += 1
        else:
            pixel_seen[digest] = idx

        if patient_id:
            if patient_id in patient_seen:
                uf.union(idx, patient_seen[patient_id])
                patient_links += 1
            else:
                patient_seen[patient_id] = idx

        # Eight eight-bit hash bands: for <=4 changed bits, at least one band
        # must match. Check the full hash and the pixels before grouping.
        possible = set()
        for band in range(8):
            possible.update(bands[(band, (phash >> (band * 8)) & 255)])
        for previous in possible:
            prev_hash, prev_vector = feature_vectors[previous]
            if (phash ^ prev_hash).bit_count() > PHASH_MAX_DISTANCE:
                continue
            if float(np.dot(vector, prev_vector)) >= MIN_GRAY_CORRELATION:
                uf.union(idx, previous)
                near_links += 1

        for band in range(8):
            bands[(band, (phash >> (band * 8)) & 255)].append(idx)
        feature_vectors.append((phash, vector))

    groups = np.array([uf.find(i) for i in range(len(paths))], dtype=np.int64)
    counts = Counter(groups.tolist())
    summary = {
        "total_images": len(paths),
        "image_groups": len(counts),
        "groups_with_multiple_images": sum(count > 1 for count in counts.values()),
        "largest_group": max(counts.values(), default=0),
        "exact_pixel_links": exact_links,
        "near_image_links": near_links,
        "explicit_patient_links": patient_links,
        "hash_max_hamming_distance": PHASH_MAX_DISTANCE,
        "minimum_gray_correlation": MIN_GRAY_CORRELATION,
    }
    return groups, summary


def grouped_train_val_test(
    paths, labels, *, task_name, output_dir, seed=SEED, precomputed_groups=None
):
    """Return disjoint group-aware train/validation/test arrays (about 60/20/20).

    All similarity detection occurs before partitioning. Importantly, this
    function does not rebalance the whole dataset: sampling / class weighting
    must be performed exclusively on its training result.
    """
    if len(paths) != len(labels) or len(paths) < 15:
        raise ValueError("Expected matching paths/labels with >=15 observations")

    original_labels = np.asarray(labels)
    if original_labels.ndim != 1 or original_labels.dtype.kind not in "iu":
        raise ValueError("Labels must be a one-dimensional integer array")

    paths = np.asarray(paths, dtype=str)
    labels = np.asarray(labels, dtype=np.int32)
    order = np.argsort(paths, kind="stable")
    paths, labels = paths[order], labels[order]
    if precomputed_groups is None:
        groups, summary = _group_near_duplicates(paths)
    else:
        if len(precomputed_groups) != len(paths):
            raise ValueError("Expected one precomputed allocation group per image")
        groups = np.asarray(precomputed_groups)[order]
        summary = {
            "total_images": len(paths),
            "image_groups": len(set(groups.tolist())),
            "exact_pixel_links": 0,
            "near_image_links": 0,
            "explicit_patient_links": 0,
            "group_origin": "global raw-source audit before deduplication; includes patient and crop constraints",
        }

    # Different labels inside a single visual/patient group are ambiguous.
    # Do not silently use them for either training or evaluation.
    by_group = defaultdict(set)
    for group, label in zip(groups, labels):
        by_group[int(group)].add(int(label))
    conflicting = {group for group, classes in by_group.items() if len(classes) > 1}
    valid = np.array([int(group) not in conflicting for group in groups], dtype=bool)
    excluded_items = [
        {
            "path": path,
            "label": int(label),
            "image_group": int(group),
            "reason": "conflicting_labels_in_group",
        }
        for path, label, group, is_valid in zip(paths, labels, groups, valid)
        if not is_valid
    ]
    summary["conflicting_label_groups_excluded"] = len(conflicting)
    summary["images_excluded_for_label_conflicts"] = len(excluded_items)
    paths, labels, groups = paths[valid], labels[valid], groups[valid]
    summary["images_after_exclusions"] = len(paths)
    if len(np.unique(labels)) < 2:
        raise ValueError("At least two classes must remain after conflict exclusion")
    if any(len(set(groups[labels == label].tolist())) < 5 for label in np.unique(labels)):
        raise ValueError("Need at least five independent allocation groups per class")

    # First reserve 20% as an untouched test set, then ~20% as validation.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val, test = next(splitter.split(paths, labels, groups))
    validation_splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
    train_sub, val_sub = next(
        validation_splitter.split(paths[train_val], labels[train_val], groups[train_val])
    )
    train, validation = train_val[train_sub], train_val[val_sub]

    split_ids = np.empty(len(paths), dtype=object)
    for split_name, indices in (("train", train), ("validation", validation), ("test", test)):
        split_ids[indices] = split_name

    seen_groups = []
    for indices in (train, validation, test):
        seen_groups.append(set(groups[indices].tolist()))
        if set(labels[indices].tolist()) != set(labels.tolist()):
            raise ValueError("Every partition must contain every declared class")
    if any(seen_groups[i] & seen_groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Image/patient group leaked between dataset partitions")

    summary["seed"] = seed
    summary["class_distribution"] = {
        name: dict(Counter(int(label) for label in labels[indices]))
        for name, indices in (("train", train), ("validation", validation), ("test", test))
    }
    summary["partition_sizes"] = {
        name: len(indices)
        for name, indices in (("train", train), ("validation", validation), ("test", test))
    }
    summary["patient_independence_guaranteed"] = False
    summary["limitation"] = (
        "Patient IDs are known only for Figshare; perceptual similarity cannot guarantee distinct patients elsewhere."
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "summary": summary,
        "excluded_items": excluded_items,
        "items": [
            {"path": path, "label": int(label), "image_group": int(group), "split": split_name}
            for path, label, group, split_name in zip(paths, labels, groups, split_ids)
        ],
    }
    manifest_path = out / f"split_{task_name}.json"
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(
            f"Refusing to replace a different split manifest: {manifest_path}. Choose a new run directory."
        )
    manifest_path.write_text(encoded, encoding="utf-8")

    print(f"Leakage-aware split {task_name}: {summary['partition_sizes']}")
    print(
        f"Group audit: {summary['image_groups']} groups, {summary['exact_pixel_links']} exact pixel links, "
        f"{summary['near_image_links']} near-image links, {summary['explicit_patient_links']} patient links"
    )
    return (
        paths[train],
        labels[train],
        paths[validation],
        labels[validation],
        paths[test],
        labels[test],
        summary,
    )
