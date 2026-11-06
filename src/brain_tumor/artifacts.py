"""Pinned demo artifacts, cached locally and verified before loading.

Importing this module never downloads anything. The release contains the
unchanged v3 weights and metrics, selected using internal validation only.
"""

import hashlib
from pathlib import Path
import tempfile
import urllib.error
import urllib.request

RELEASE_URL = "https://github.com/Irfat69/Projet_Tumeur/releases/download/gradio-demo-v3"
ARTIFACTS = {
    "binary": {
        "model_sha256": "90c10e0a7ecd45fc18536ea7638640f2ea17a644ef9fcc96289ef721c8673dee",
        "metrics_sha256": "4216f83445d8c4f2a7c0cb24929de43fe2c91dd370c7918aef273936461138e0",
    },
    "multiclass": {
        "model_sha256": "80240035c826c5e2e39183835c2dc627f97cfd5b0e72a1839e724f8bbb193ad2",
        "metrics_sha256": "10e128ea10a914c9c26ec2150672786daa9b5a3286ec0e5a38292e2029962b87",
    },
}


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def demo_registry(root):
    """Reuse complete local audit runs, otherwise use the release cache."""
    root = Path(root)
    registry = {}
    for task, hashes in ARTIFACTS.items():
        run = Path(f"outputs/runs/audit_v3_{task}")
        if (root / run / "model.keras").is_file() and (root / run / "metrics.json").is_file():
            model, metrics = run / "model.keras", run / "metrics.json"
        else:
            model = Path(f"models/gradio-demo-v3/{task}.keras")
            metrics = Path(f"models/gradio-demo-v3/{task}_metrics.json")
        registry[task] = {
            **hashes,
            "model_file": model.as_posix(),
            "metrics_file": metrics.as_posix(),
            "model_url": f"{RELEASE_URL}/{task}.keras",
            "metrics_url": f"{RELEASE_URL}/{task}_metrics.json",
        }
    return registry


def ensure_artifact(url, destination, expected_sha256):
    """Download to a temporary file; publish it only after a valid SHA-256."""
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError(f"Artifact integrity check failed; preserve and inspect {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        print(f"Downloading verified demo artifact: {destination.name}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "BrainTumorResearchDemo/0.1"})
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".", suffix=".partial", delete=False
        ) as file:
            temporary = Path(file.name)
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
        if sha256(temporary) != expected_sha256:
            raise ValueError(f"Downloaded artifact checksum mismatch: {destination.name}")
        temporary.replace(destination)
        return destination
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Cannot download the demo models. Check the network connection and retry python app.py."
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_demo_models(root, registry, tasks=("binary", "multiclass")):
    for task in tasks:
        record = registry[task]
        for kind in ("metrics", "model"):
            ensure_artifact(
                record[f"{kind}_url"],
                Path(root) / record[f"{kind}_file"],
                record[f"{kind}_sha256"],
            )
