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


def test_new_input_invalidates_previous_scores_and_explanation():
    result, binary, types, cam, analyze, clear = app.reset_analysis(Image.new("RGB", (32, 32)))
    assert "Image importée" in result
    assert binary is None
    assert types is None
    assert cam is None
    assert analyze["interactive"] and clear["interactive"]


def test_empty_and_invalid_inputs_cannot_start_an_analysis():
    empty = app.reset_analysis(None)
    assert "En attente" in empty[0]
    assert not empty[-2]["interactive"] and not empty[-1]["interactive"]
    invalid = app.reset_analysis(Image.new("RGB", (8, 8)))
    assert "16" in invalid[0] and "Analyse impossible" in invalid[0]
    assert not invalid[-2]["interactive"] and invalid[-1]["interactive"]


def test_ui_preserves_scores_when_translating_labels(monkeypatch):
    image = Image.new("RGB", (32, 32))
    binary = {"Aucune tumeur": 0.2, "Tumeur": 0.8}
    types = {"glioma": 0.1, "meningioma": 0.7, "pituitary": 0.2}
    monkeypatch.setattr(app, "predict_irm", lambda value: ("result", binary, types, image))
    finished = app.analyze_interface(image)
    assert finished[-2]["interactive"]
    assert finished[1] == {"Motif non tumoral": 0.2, "Motif tumoral": 0.8}
    assert finished[2] == {"Gliome": 0.1, "Méningiome": 0.7, "Tumeur hypophysaire": 0.2}
    assert finished[3] is image


def test_ui_recovers_from_inference_failure_without_stale_outputs(monkeypatch):
    monkeypatch.setattr(app, "predict_irm", lambda image: ("error", None, None, None))
    finished = app.analyze_interface(Image.new("RGB", (32, 32)))
    assert finished[-2]["interactive"]
    assert finished[1] is None and finished[2] is None
    assert finished[3] is None


def test_model_status_reflects_loaded_models(monkeypatch):
    monkeypatch.setattr(app, "binary_model", None)
    monkeypatch.setattr(app, "tumor_type_model", None)
    assert "Modèles prêts" not in app.model_status_html()
    monkeypatch.setattr(app, "binary_model", object())
    monkeypatch.setattr(app, "tumor_type_model", object())
    assert "Modèles prêts" in app.model_status_html()


def test_replacing_image_during_inference_discards_old_results(monkeypatch):
    session = {"revision": 0}
    image = Image.new("RGB", (32, 32))

    def replaced_while_running(value):
        app.reset_analysis(Image.new("RGB", (40, 40)), session)
        return "old result", {"Aucune tumeur": 0.2, "Tumeur": 0.8}, {}, image

    monkeypatch.setattr(app, "predict_irm", replaced_while_running)
    outputs = app.analyze_interface(image, session)
    assert session["revision"] == 1
    assert all(value == {"__type__": "update"} for value in outputs)


def test_nontumor_prediction_skips_type_classifier_and_type_chart(monkeypatch):
    monkeypatch.setattr(app, "load_models", lambda: None)
    monkeypatch.setattr(app, "binary_model", object())
    monkeypatch.setattr(app, "CALIBRATED_THRESHOLD", 0.22)
    monkeypatch.setattr(app, "BINARY_METRICS", {"decision_threshold": 0.22})
    image = Image.new("RGB", (32, 32))

    def probabilities(value, model, classes):
        assert classes == 2, "Type classification must remain conditional"
        return np.array([0.05])

    monkeypatch.setattr(app, "predict_probabilities", probabilities)
    monkeypatch.setattr(app, "generate_gradcam", lambda *args, **kwargs: image)
    result, binary, types, cam = app.predict_irm(image)
    assert "Motif non tumoral prédit" in result and "predicted-type" not in result
    assert types == {} and cam is image
    ui = app.interface_updates(result, binary, types, cam)
    assert ui[2] is None
