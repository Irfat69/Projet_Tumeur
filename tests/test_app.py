import numpy as np
from PIL import Image
import app


def test_demo_uses_evaluated_threshold_and_conditional_labels(monkeypatch):
    monkeypatch.setattr(app, "load_models", lambda: None)
    monkeypatch.setattr(app, "binary_model", object())
    monkeypatch.setattr(app, "tumor_type_model", object())
    monkeypatch.setattr(app, "CALIBRATED_THRESHOLD", 0.21)
    monkeypatch.setattr(app, "BINARY_METRICS", {"calibrated_threshold": 0.21})
    monkeypatch.setattr(
        app,
        "predict_probabilities",
        lambda image, model, classes: (
            np.array([0.8]) if classes == 2 else np.array([0.1, 0.7, 0.2])
        ),
    )
    monkeypatch.setattr(app, "generate_gradcam", lambda image, *args, **kwargs: image)
    image = Image.new("RGB", (32, 32))
    result, binary, types, cam = app.predict_irm(image)
    assert "0.21" in result and "meningioma" in result
    assert np.isclose(sum(binary.values()), 1) and np.isclose(sum(types.values()), 1)
    assert list(types) == app.TYPE_CLASS_NAMES and cam is image


def test_invalid_image_and_threshold_are_reported_without_fake_probabilities(monkeypatch):
    result, binary, types, cam = app.predict_irm(None)
    assert "Ajoute une image" in result and binary is None and types is None and cam is None
    monkeypatch.setattr(app, "BINARY_METRICS", {"calibrated_threshold": float("nan")})
    assert app.load_threshold() is None


def test_prediction_failure_is_html_escaped(monkeypatch):
    monkeypatch.setattr(app, "BINARY_METRICS", {"calibrated_threshold": 0.21})
    monkeypatch.setattr(app, "CALIBRATED_THRESHOLD", 0.21)

    def fail():
        raise ValueError("<script>alert(1)</script>")

    monkeypatch.setattr(app, "load_models", fail)
    result, *_ = app.predict_irm(Image.new("RGB", (32, 32)))
    assert "<script>" not in result and "&lt;script&gt;" in result
