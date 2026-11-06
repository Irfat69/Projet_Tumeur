"""Audit cached sources, provenance and historical splits without training.

Run: python -m brain_tumor.data_audit --download. Records stay in ignored outputs/.
Similarity edges constrain allocation; they never assert patient identity.
"""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from threadpoolctl import threadpool_limits

from brain_tumor.datasets import SOURCES, dataset_root, inventory
from brain_tumor.safe_split import UnionFind, _signature, _read_mri

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/audit_data"
# Five disjoint bands guarantee a candidate for every <=4-bit hash mismatch.
HASH_BANDS = [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]


def features(record):
    path = record["path"]
    with open(path, "rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    pixel, phash, vector, _ = _signature(path)
    image, _ = _read_mri(path)
    gray = np.asarray(image.convert("L"))
    foreground = np.argwhere(gray > 12)
    if len(foreground):
        y0, x0 = foreground.min(axis=0)
        y1, x1 = foreground.max(axis=0) + 1
        crop = image.crop((int(x0), int(y0), int(x1), int(y1)))
    else:
        crop = image
    a = np.asarray(crop.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coeff = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].ravel()[1:]
    crop_hash = sum(int(v) << b for b, v in enumerate(coeff > np.median(coeff)))
    a = a.ravel() - a.mean()
    a /= max(float(np.linalg.norm(a)), 1e-6)
    border = np.concatenate(
        [gray[:8].ravel(), gray[-8:].ravel(), gray[:, :8].ravel(), gray[:, -8:].ravel()]
    )
    stats = [
        image.width,
        image.height,
        float(gray.mean()),
        float(gray.std()),
        float(border.mean()),
        float(np.mean(gray < 8)),
    ]
    return sha, pixel, phash, vector, crop_hash, a, stats


def build_cache():
    OUT.mkdir(parents=True, exist_ok=True)
    if all((OUT / name).exists() for name in ("features.npz", "records.json", "cache_key.txt")):
        cached_records = json.loads((OUT / "records.json").read_text())
        try:
            cached_key = hashlib.sha256(
                json.dumps(
                    [
                        (
                            r["path"],
                            Path(r["path"]).stat().st_size,
                            Path(r["path"]).stat().st_mtime_ns,
                        )
                        for r in cached_records
                    ]
                ).encode()
            ).hexdigest()
        except FileNotFoundError:
            cached_key = None
        if cached_key == (OUT / "cache_key.txt").read_text():
            print(
                "Reusing verified size/mtime signature cache for pinned source versions", flush=True
            )
            return cached_records, np.load(OUT / "features.npz")
    records = inventory()
    key = hashlib.sha256(
        json.dumps(
            [
                (r["path"], Path(r["path"]).stat().st_size, Path(r["path"]).stat().st_mtime_ns)
                for r in records
            ]
        ).encode()
    ).hexdigest()
    if (OUT / "features.npz").exists() and (OUT / "cache_key.txt").read_text() == key:
        return records, np.load(OUT / "features.npz")
    print(f"Inspecting {len(records)} raw files", flush=True)
    results, valid, corrupt = [], [], []

    def guarded(record):
        try:
            return features(record), None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (record, (value, error)) in enumerate(zip(records, pool.map(guarded, records))):
            if error:
                corrupt.append({**record, "error": error})
            else:
                results.append(value)
                valid.append(record)
            if (i + 1) % 2000 == 0:
                print(f"Decoded {i + 1}/{len(records)}", flush=True)
    (OUT / "corrupt.json").write_text(json.dumps(corrupt, indent=2))
    if corrupt:
        raise RuntimeError(
            f"{len(corrupt)} files need quarantine; see outputs/audit_data/corrupt.json"
        )
    arrays = {
        "sha": np.array([x[0] for x in results]),
        "pixel": np.array([x[1] for x in results]),
        "phash": np.array([x[2] for x in results], dtype=np.uint64),
        "vector": np.array([x[3] for x in results]),
        "crop_hash": np.array([x[4] for x in results], dtype=np.uint64),
        "crop": np.array([x[5] for x in results]),
        "stats": np.array([x[6] for x in results]),
    }
    np.savez(OUT / "features.npz", **arrays)
    (OUT / "records.json").write_text(json.dumps(valid, indent=2))
    (OUT / "cache_key.txt").write_text(key)
    return valid, arrays


def connect_similar(hashes, vectors, uf, min_correlation):
    bands, edges = defaultdict(list), []
    for i, hash_value in enumerate(hashes):
        h = int(hash_value)
        candidates = set()
        for band, (shift, width) in enumerate(HASH_BANDS):
            candidates.update(bands[(band, (h >> shift) & ((1 << width) - 1))])
        for j in candidates:
            if (h ^ int(hashes[j])).bit_count() <= 4:
                correlation = float(np.dot(vectors[i], vectors[j]))
                if correlation >= min_correlation:
                    uf.union(i, j)
                    edges.append((i, j, correlation))
        for band, (shift, width) in enumerate(HASH_BANDS):
            bands[(band, (h >> shift) & ((1 << width) - 1))].append(i)
    return edges


def connect_figshare_orientations(records, hashes, vectors, uf):
    """Check seven alternative D4 orientations of Figshare against JPEG aliases.

    MATLAB/HDF5 array ordering and JPEG exports may transpose or rotate a scan.
    A match constrains allocation; it does not equate unrelated patient IDs.
    """
    bands, edges = defaultdict(list), []
    for i, value in enumerate(hashes):
        h = int(value)
        for band, (shift, width) in enumerate(HASH_BANDS):
            bands[(band, (h >> shift) & ((1 << width) - 1))].append(i)
    for i, record in enumerate(records):
        if not record["patient_id"]:
            continue
        image = vectors[i].reshape(32, 32)
        for flipped in (False, True):
            for rotation in range(4):
                if not flipped and not rotation:
                    continue
                array = np.rot90(np.fliplr(image) if flipped else image, rotation)
                coeff = dct(dct(array, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].ravel()[
                    1:
                ]
                h = sum(int(v) << bit for bit, v in enumerate(coeff > np.median(coeff)))
                possible = set()
                for band, (shift, width) in enumerate(HASH_BANDS):
                    possible.update(bands[(band, (h >> shift) & ((1 << width) - 1))])
                vector = array.ravel()
                for j in possible:
                    if records[j]["source"] == "figshare" or (h ^ int(hashes[j])).bit_count() > 4:
                        continue
                    correlation = float(np.dot(vector, vectors[j]))
                    if correlation >= 0.995:
                        uf.union(i, j)
                        edges.append((i, j, correlation))
    return edges


def run():
    records, a = build_cache()
    uf = UnionFind(len(records))
    for keys in (a["sha"], a["pixel"], [r["patient_id"] for r in records]):
        seen = {}
        for i, key in enumerate(keys):
            if not key:
                continue
            if key in seen:
                uf.union(i, seen[key])
            else:
                seen[key] = i
    print("Grouping full-frame perceptual copies", flush=True)
    exact_edges = connect_similar(a["phash"], a["vector"], uf, 0.98)
    # A stricter crop-normalized rule is an additional conservative constraint.
    print("Grouping crop-normalized copies", flush=True)
    crop_edges = connect_similar(a["crop_hash"], a["crop"], uf, 0.995)
    print("Checking Figshare export orientations", flush=True)
    orientation_edges = connect_figshare_orientations(records, a["crop_hash"], a["crop"], uf)
    groups = [uf.find(i) for i in range(len(records))]
    for r, sha, pixel, group in zip(records, a["sha"], a["pixel"], groups):
        r.update(sha256=str(sha), pixel_sha256=str(pixel), audit_group=int(group))
    (OUT / "records.json").write_text(json.dumps(records, indent=2))
    by_sha, by_group = defaultdict(list), defaultdict(list)
    for i, r in enumerate(records):
        by_sha[r["sha256"]].append(i)
        by_group[r["audit_group"]].append(i)
    conflicts = [
        indices
        for indices in by_sha.values()
        if len({records[i]["class_name"] for i in indices} - {"tumor_unspecified"}) > 1
    ]
    source_pairs = {}
    for left in sorted({r["source"] for r in records}):
        for right in sorted({r["source"] for r in records}):
            if left < right:
                ls = {r["sha256"] for r in records if r["source"] == left}
                rs = {r["sha256"] for r in records if r["source"] == right}
                source_pairs[f"{left} / {right}"] = len(ls & rs)
    summary = {
        "raw_files": len(records),
        "corrupt_files": 0,
        "unique_sha256": len(by_sha),
        "unique_resized_pixels": len(set(a["pixel"])),
        "strict_similarity_edges": len(exact_edges),
        "crop_similarity_edges": len(crop_edges),
        "figshare_orientation_edges": len(orientation_edges),
        "allocation_groups": len(by_group),
        "raw_sha_label_conflict_groups": len(conflicts),
        "cross_source_shared_sha256": source_pairs,
        "sources": {},
        "historical_splits": {},
        "criteria": {
            "phash_hamming": 4,
            "gray64_correlation": 0.98,
            "cropped32_correlation": 0.995,
            "interpretation": "Allocation constraints, not inferred patient identities; residual similarity remains possible.",
        },
    }
    for source in sorted({r["source"] for r in records}):
        subset = [r for r in records if r["source"] == source]
        original_train = {
            r["sha256"] for r in subset if r["original_split"] in ("train", "training")
        }
        original_test = {r["sha256"] for r in subset if r["original_split"] in ("test", "testing")}
        summary["sources"][source] = {
            "images": len(subset),
            "classes": dict(Counter(r["class_name"] for r in subset)),
            "original_folders": dict(Counter(r["original_split"] for r in subset)),
            "known_patients": len({r["patient_id"] for r in subset if r["patient_id"]}),
            "original_train_test_shared_sha256": len(original_train & original_test),
        }
    lookup = {r["path"]: i for i, r in enumerate(records)}
    for task in ("binary", "multiclass"):
        manifest_path = ROOT / f"outputs/split_{task}.json"
        if not manifest_path.exists():
            summary["historical_splits"][task] = {
                "status": "Historical local manifest unavailable; source audit still completed."
            }
            continue
        manifest = json.loads(manifest_path.read_text())
        partition_by_group = defaultdict(set)
        for item in manifest["items"]:
            partition_by_group[groups[lookup[item["path"]]]].add(item["split"])
        crossings = {
            g: sorted(splits) for g, splits in partition_by_group.items() if len(splits) > 1
        }
        exposed_test = [
            x["path"]
            for x in manifest["items"]
            if x["split"] == "test" and groups[lookup[x["path"]]] in crossings
        ]
        summary["historical_splits"][task] = {
            "cross_partition_audit_groups": len(crossings),
            "test_images_in_crossing_groups": len(exposed_test),
            "excluded_label_conflict_images": len(manifest["excluded_items"]),
        }
        (OUT / f"{task}_exposed_test.json").write_text(json.dumps(exposed_test, indent=2))
    (OUT / "orientation_edges.json").write_text(json.dumps(orientation_edges))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare pinned data and provenance groups")
    parser.add_argument(
        "--download", action="store_true", help="Download only missing pinned sources"
    )
    args = parser.parse_args()
    if args.download:
        for source in SOURCES:
            print(source, dataset_root(source, download=True), flush=True)
    with threadpool_limits(limits=2):
        run()
