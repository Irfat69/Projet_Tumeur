"""Pinned public dataset inventory. Raw files and patient identifiers stay local."""

from pathlib import Path
import hashlib

import h5py
import numpy as np

SOURCES = {
    "sartaj": ("sartajbhuvaji/brain-tumor-classification-mri", 3),
    "mri_7000": ("masoudnickparvar/brain-tumor-mri-dataset", 2),
    "negative_boost": ("praneet0327/brain-tumor-dataset", 1),
    "figshare": ("ashkhagan/figshare-brain-tumor-dataset", 1),
}
TYPE_CLASSES = ["glioma", "meningioma", "pituitary"]
FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def dataset_root(source, download=False):
    handle, version = SOURCES[source]
    root = Path.home() / ".cache/kagglehub/datasets" / handle / "versions" / str(version)
    if not root.is_dir() and download:
        import kagglehub

        root = Path(kagglehub.dataset_download(f"{handle}/versions/{version}"))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing pinned dataset {handle}/versions/{version}. Use --download."
        )
    return root


def figshare_patient(path):
    with h5py.File(path, "r") as file:
        values = np.asarray(file["cjdata/PID"]).reshape(-1)
    pid = "".join(chr(int(c)) for c in values if int(c)).strip()
    if not pid:
        raise ValueError(f"Empty Figshare patient identifier: {path}")
    # A stable anonymous key is enough to enforce grouping.
    return "figshare:" + hashlib.sha256(pid.encode()).hexdigest()


def infer_class(path):
    for part in reversed(Path(path).parts[:-1]):
        token = part.lower().replace("-", "_").replace(" ", "_")
        if token in {"no", "normal", "healthy", "negative", "no_tumor", "notumor"}:
            return "no_tumor"
        for name in TYPE_CLASSES:
            if name in token:
                return name
        if token in {"yes", "tumor", "tumour", "positive"}:
            return "tumor_unspecified"
    return None


def inventory(download=False):
    records = []
    for source, (handle, version) in SOURCES.items():
        root = dataset_root(source, download)
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".mat", ".jpg", ".jpeg", ".png"}:
                continue
            if path.suffix.lower() == ".mat":
                with h5py.File(path, "r") as f:
                    if "cjdata/label" not in f:
                        continue
                    name = FIGSHARE_LABELS[int(np.asarray(f["cjdata/label"]).reshape(-1)[0])]
                patient = figshare_patient(path)
            else:
                name, patient = infer_class(path), None
            if name is None:
                continue
            relative = path.relative_to(root).as_posix()
            original_split = next(
                (
                    s.lower()
                    for s in path.relative_to(root).parts
                    if s.lower() in {"training", "testing", "train", "test"}
                ),
                "unspecified",
            )
            records.append(
                {
                    "path": str(path),
                    "relative_path": relative,
                    "source": source,
                    "dataset": handle,
                    "version": version,
                    "class_name": name,
                    "patient_id": patient,
                    "original_split": original_split,
                }
            )
    return records
