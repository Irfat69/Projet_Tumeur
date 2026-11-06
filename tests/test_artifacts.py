import json
import hashlib
import io
import numpy as np
import pytest
from brain_tumor.artifacts import demo_registry, ensure_artifact, ensure_demo_models
from brain_tumor import artifacts
from brain_tumor.gpu import ensure_link
from brain_tumor.model_manager import ModelManager
from brain_tumor.utils import evaluate_model


def test_metrics_preserve_history_and_reject_nonfinite_values(tmp_path):
    manager = ModelManager(tmp_path / "models")
    manager.save_metrics({"accuracy": 0.8}, output_dir=tmp_path)
    with pytest.raises(FileExistsError):
        manager.save_metrics({"accuracy": 0.9}, output_dir=tmp_path)
    assert json.loads((tmp_path / "results.json").read_text())["accuracy"] == 0.8
    with pytest.raises(ValueError):
        manager.save_metrics({"accuracy": float("nan")}, output_dir=tmp_path, filename="bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_gpu_links_are_idempotent_and_never_replace_a_file(tmp_path):
    source, link = tmp_path / "library.so", tmp_path / "alias.so"
    source.write_text("library")
    assert ensure_link(source, link) == 1
    assert ensure_link(source, link) == 0
    other = tmp_path / "existing.so"
    other.write_text("preserve")
    with pytest.raises(SystemExit):
        ensure_link(source, other)
    assert other.read_text() == "preserve"


def test_legacy_evaluator_handles_absent_class():
    class Model:
        def predict(self, images):
            return np.array([[0.9], [0.2]])

    result = evaluate_model(Model(), None, np.array([1, 1]))
    assert result["confusion_matrix"] == [[0, 0], [1, 1]]


def test_demo_download_checks_bytes_and_reuses_valid_cache(tmp_path, monkeypatch):
    content = b"verified model bytes"
    digest = hashlib.sha256(content).hexdigest()
    requests = []

    def fetch(request, timeout):
        requests.append(request.full_url)
        return io.BytesIO(content)

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", fetch)
    destination = tmp_path / "models/model.keras"
    for _ in range(2):
        assert ensure_artifact("https://example.test/model", destination, digest) == destination
    assert destination.read_bytes() == content and len(requests) == 1
    assert not list(destination.parent.glob("*.partial"))


def test_corrupt_download_is_not_published_or_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        artifacts.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong bytes")
    )
    destination = tmp_path / "model.keras"
    with pytest.raises(ValueError, match="checksum mismatch"):
        ensure_artifact("https://example.test/model", destination, "0" * 64)
    assert not destination.exists() and not list(tmp_path.glob("*.partial"))


def test_corrupt_existing_file_is_preserved_without_network(tmp_path, monkeypatch):
    destination = tmp_path / "model.keras"
    destination.write_bytes(b"preserve this file")

    def no_network(*args, **kwargs):
        pytest.fail("Existing mismatched artifacts must not trigger a download")

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", no_network)
    with pytest.raises(ValueError, match="integrity check failed"):
        ensure_artifact("https://example.test/model", destination, "0" * 64)
    assert destination.read_bytes() == b"preserve this file"


def test_fresh_demo_registry_downloads_both_models_and_metadata(tmp_path, monkeypatch):
    registry = demo_registry(tmp_path)
    calls = []
    monkeypatch.setattr(artifacts, "ensure_artifact", lambda *args: calls.append(args))
    ensure_demo_models(tmp_path, registry)
    assert len(calls) == 4
    assert all(str(path).startswith(str(tmp_path / "models")) for _, path, _ in calls)
    assert {url.rsplit("/", 1)[1] for url, _, _ in calls} == {
        "binary.keras",
        "binary_metrics.json",
        "multiclass.keras",
        "multiclass_metrics.json",
    }
