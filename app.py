"""Interface Gradio pour la detection de tumeurs cerebrales sur IRM."""

from pathlib import Path
import json
import os
import html
import logging
import hashlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

from brain_tumor.inference import generate_gradcam, predict_probabilities
from brain_tumor.artifacts import demo_registry, ensure_demo_models


BASE_DIR = Path(__file__).resolve().parent


def load_json(path: Path):
    if not path.exists():
        return {}

    try:
        with open(path, "r") as file:
            return json.load(file)
    except Exception:
        return {}


MODEL_REGISTRY = demo_registry(BASE_DIR)
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
BINARY_RUN = (
    Path(os.environ["MRI_BINARY_RUN"]).resolve() if os.environ.get("MRI_BINARY_RUN") else None
)
TYPE_RUN = (
    Path(os.environ["MRI_MULTICLASS_RUN"]).resolve()
    if os.environ.get("MRI_MULTICLASS_RUN")
    else None
)
BINARY_METRICS_FILE = (
    BINARY_RUN / "metrics.json"
    if BINARY_RUN
    else BASE_DIR
    / MODEL_REGISTRY.get("binary", {}).get("metrics_file", "outputs/transfer_results_grouped.json")
)

# Group-aware evaluations (Nov 2026): prefer models trained on leakage-aware
# partitions, while leaving legacy model files intact for historical comparison.
BINARY_MODEL_NAME = "brain_tumor_transfer_grouped"
TYPE_MODEL_NAME = "brain_tumor_type_grouped"
TYPE_CLASS_NAMES = ["glioma", "meningioma", "pituitary"]

GRADCAM_ALPHA = 0.52


binary_model = None
tumor_type_model = None


def load_models():
    global binary_model, tumor_type_model
    if binary_model is None:
        binary_model = load_verified_model("binary", BINARY_RUN)
    if tumor_type_model is None:
        try:
            tumor_type_model = load_verified_model("multiclass", TYPE_RUN)
            metadata = load_json(
                TYPE_RUN / "metrics.json"
                if TYPE_RUN
                else BASE_DIR
                / MODEL_REGISTRY.get("multiclass", {}).get(
                    "metrics_file", "outputs/tumor_type_results_grouped.json"
                )
            )
            if metadata.get("class_names") != TYPE_CLASS_NAMES:
                tumor_type_model = None
                raise ValueError(
                    "Ordre des classes absent ou incompatible avec le modèle multiclasses."
                )
        except FileNotFoundError:
            pass


BINARY_METRICS = load_json(BINARY_METRICS_FILE)


def load_verified_model(task, run):
    if run:
        path = run / "model.keras"
        metadata = load_json(run / "metrics.json")
        expected = metadata.get("model_sha256")
    else:
        record = MODEL_REGISTRY.get(task, {})
        path = BASE_DIR / record.get("model_file", "models/missing.keras")
        metric_path = BASE_DIR / record.get("metrics_file", "outputs/missing.json")
        if not metric_path.is_file():
            raise FileNotFoundError("Métadonnées du modèle absentes. Consulter le README.")
        if hashlib.sha256(metric_path.read_bytes()).hexdigest() != record.get("metrics_sha256"):
            raise ValueError("Les métriques ne correspondent plus au registre du modèle.")
        expected = record.get("model_sha256")
    if not path.is_file():
        raise FileNotFoundError(
            "Modèle absent. Entraîner le modèle et configurer MRI_BINARY_RUN / MRI_MULTICLASS_RUN (README)."
        )
    with path.open("rb") as file:
        actual = hashlib.file_digest(file, "sha256").hexdigest()
    if not expected or actual != expected:
        raise ValueError(
            "Empreinte du modèle incompatible avec ses métadonnées. Vérifier les artefacts."
        )
    return tf.keras.models.load_model(path, compile=False)


def pct(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%".replace(".", ",")


def load_threshold(default=None):
    value = BINARY_METRICS.get(
        "decision_threshold", BINARY_METRICS.get("calibrated_threshold", default)
    )
    try:
        value = float(value)
        return value if np.isfinite(value) and 0 <= value <= 1 else None
    except (TypeError, ValueError):
        return None


CALIBRATED_THRESHOLD = load_threshold()


def initialize_models():
    """Fetch missing pinned demo assets only when explicitly launching the app."""
    global BINARY_METRICS, CALIBRATED_THRESHOLD
    default_tasks = [
        task for task, run in (("binary", BINARY_RUN), ("multiclass", TYPE_RUN)) if run is None
    ]
    ensure_demo_models(BASE_DIR, MODEL_REGISTRY, tasks=default_tasks)
    BINARY_METRICS = load_json(BINARY_METRICS_FILE)
    CALIBRATED_THRESHOLD = load_threshold()
    if CALIBRATED_THRESHOLD is None:
        raise ValueError("Seuil de validation absent ou invalide dans les métadonnées du modèle.")
    load_models()


def predict_tumor_type(image: Image.Image):
    if tumor_type_model is None:
        return None, None, None, {}

    probabilities = predict_probabilities(image, tumor_type_model, 3)
    best_idx = int(np.argmax(probabilities))
    best_label = TYPE_CLASS_NAMES[best_idx]
    best_confidence = float(probabilities[best_idx])
    distribution = {
        class_name: float(probabilities[index]) for index, class_name in enumerate(TYPE_CLASS_NAMES)
    }

    return best_label, best_confidence, best_idx, distribution


def build_result_html(
    has_tumor,
    probability,
    confidence,
    threshold,
    tumor_type_label=None,
    tumor_type_confidence=None,
    gradcam_target="modele binaire",
):
    status_class = "danger" if has_tumor else "safe"
    badge = "CLASSE DU MODELE : TUMEUR" if has_tumor else "CLASSE DU MODELE : SANS TUMEUR"
    title = "Motif tumoral prédit" if has_tumor else "Motif non tumoral prédit"

    type_line = ""
    if tumor_type_label is not None:
        type_line = (
            "<div class='result-row emphasis'>"
            "<span>Type prédit, conditionnel à la classe tumeur</span>"
            f"<strong>{tumor_type_label} · {tumor_type_confidence * 100:.2f}%</strong>"
            "</div>"
        )
    elif has_tumor:
        type_line = "<p>Modèle des types indisponible : consulter la configuration du README.</p>"

    return (
        f"<section class='result-card {status_class}'>"
        f"<div class='badge'>{badge}</div>"
        f"<h2>{title}</h2>"
        "<div class='result-row'>"
        "<span>Score de la classe (non calibré)</span>"
        f"<strong>{confidence * 100:.2f}%</strong>"
        "</div>"
        "<div class='result-row'>"
        "<span>Score tumeur</span>"
        f"<strong>{probability:.4f}</strong>"
        "</div>"
        "<div class='result-row'>"
        "<span>Seuil utilise</span>"
        f"<strong>{threshold:.2f}</strong>"
        "</div>"
        f"{type_line}"
        f"<p>Grad-CAM: {gradcam_target}</p>"
        "</section>"
    )


def predict_irm(image: Image.Image):
    try:
        if image is None:
            raise ValueError("Ajoute une image IRM avant de lancer la démonstration.")
        if not BINARY_METRICS or CALIBRATED_THRESHOLD is None:
            raise ValueError(
                "Métadonnées absentes : entraîne les modèles ou configure leurs chemins."
            )
        load_models()
        threshold = CALIBRATED_THRESHOLD

        tumor_probability = float(predict_probabilities(image, binary_model, 2)[0])
        has_tumor = tumor_probability >= threshold
        confidence = tumor_probability if has_tumor else 1.0 - tumor_probability

        binary_distribution = {
            "Aucune tumeur": 1.0 - tumor_probability,
            "Tumeur": tumor_probability,
        }
        type_distribution = {}
        type_label = None
        type_confidence = None
        gradcam_model = binary_model
        gradcam_class_index = int(has_tumor)
        gradcam_target = "classe prédite du modèle binaire"

        if has_tumor:
            type_label, type_confidence, type_index, type_distribution = predict_tumor_type(image)
            if type_label is not None:
                gradcam_model = tumor_type_model
                gradcam_class_index = type_index
                gradcam_target = f"classe {type_label}"

        gradcam_image = generate_gradcam(
            image,
            gradcam_model,
            class_index=gradcam_class_index,
        )
        result_html = build_result_html(
            has_tumor=has_tumor,
            probability=tumor_probability,
            confidence=confidence,
            threshold=threshold,
            tumor_type_label=type_label,
            tumor_type_confidence=type_confidence,
            gradcam_target=gradcam_target,
        )

        return result_html, binary_distribution, type_distribution, gradcam_image

    except Exception as error:
        if isinstance(error, (ValueError, FileNotFoundError)):
            logging.info("MRI demo input/model validation failed: %s", error)
        else:
            logging.exception("MRI demo inference failed")
        message = (
            str(error)
            if isinstance(error, (ValueError, FileNotFoundError))
            else "Prédiction impossible. Vérifie les modèles et consulte les journaux du serveur."
        )
        return (
            "<section class='result-card danger'>"
            "<div class='badge'>ERREUR</div>"
            "<h2>Analyse impossible</h2>"
            f"<p>{html.escape(message)}</p>"
            "</section>",
            None,
            None,
            None,
        )


custom_css = """
:root {
    --bg-start: #0b1220;
    --bg-end: #0f2f2d;
    --panel: rgba(10, 15, 26, 0.72);
    --panel-border: rgba(255, 255, 255, 0.15);
    --text: #f4f7fb;
    --muted: #b8c2cf;
    --accent: #21c78a;
    --danger: #ff7b6b;
    --danger-bg: rgba(255, 123, 107, 0.15);
    --safe-bg: rgba(33, 199, 138, 0.15);
}

.gradio-container {
    background: radial-gradient(circle at 10% 10%, #16304c, transparent 35%),
                radial-gradient(circle at 90% 15%, #1b4a47, transparent 30%),
                linear-gradient(140deg, var(--bg-start), var(--bg-end));
    color: var(--text);
    font-family: "Segoe UI", system-ui, sans-serif;
}

.hero {
    background: linear-gradient(120deg, rgba(20, 29, 45, 0.86), rgba(14, 57, 54, 0.86));
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
}

.hero h1 {
    margin: 0 0 8px 0;
    color: var(--text);
    font-size: 1.7rem;
    letter-spacing: 0;
}

.hero p {
    margin: 4px 0;
    color: var(--muted);
}

.panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 14px;
    backdrop-filter: blur(4px);
}

.result-card {
    border: 1px solid var(--panel-border);
    border-radius: 14px;
    padding: 14px;
}

.result-card.safe {
    background: var(--safe-bg);
}

.result-card.danger {
    background: var(--danger-bg);
}

.result-card h2 {
    margin: 10px 0 12px 0;
    color: var(--text);
    font-size: 1.2rem;
    letter-spacing: 0;
}

.result-card p {
    margin: 10px 0 0 0;
    color: var(--muted);
}

.badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.18);
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0;
}

.result-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 7px 0;
    border-top: 1px solid rgba(255, 255, 255, 0.13);
    color: var(--muted);
}

.result-row strong {
    color: var(--text);
    text-align: right;
}

.result-row.emphasis strong {
    color: var(--accent);
}

#predict-btn {
    border: none !important;
    background: linear-gradient(120deg, #20bb83, #17a49d) !important;
    color: #ffffff !important;
    font-weight: 700;
}

#predict-btn:hover {
    filter: brightness(1.08);
}

.result-card .result-row span, .mri-note p, .mri-note strong {
    color: #c4d2db;
}

.result-card .badge {
    color: #f4f7fb;
}

.footer-note {
    font-size: 0.9rem;
    color: var(--muted);
}

"""


with gr.Blocks(title="Détection de tumeur cérébrale - Portfolio") as demo:
    gr.Markdown(
        """
        <div class='hero'>
            <h1>Détection IRM - Tumeur Cérébrale</h1>
            <p>Démonstration de classification binaire, de type tumoral et de Grad-CAM.</p>
            <p class='footer-note'>Prototype pédagogique, sans validation clinique. Les scores ne sont pas des probabilités cliniques ; Grad-CAM ne localise pas une tumeur de manière validée.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(elem_classes=["panel"], scale=6):
            image_input = gr.Image(type="pil", label="Image IRM", sources=["upload"], height=380)
            gradcam_output = gr.Image(
                type="pil",
                label="Grad-CAM — attribution exploratoire, pas une segmentation",
                height=300,
            )
            gr.Markdown(
                f"Seuil fixé sur validation : **{CALIBRATED_THRESHOLD:.2f}**. Aucun réglage sur le test."
                if CALIBRATED_THRESHOLD is not None
                else "Modèles ou métadonnées absents : consulter les instructions du README.",
                elem_classes=["mri-note"],
            )
            gr.Markdown(
                "Importer une IRM anonymisée. L'application ne vérifie pas automatiquement le domaine de l'image et ne rejette pas les images hors distribution.",
                elem_classes=["mri-note"],
            )

        with gr.Column(elem_classes=["panel"], scale=5):
            result_output = gr.HTML(label="Résultat")
            probability_output = gr.Label(label="Scores du modèle binaire (non calibrés)")
            tumor_type_output = gr.Label(label="Scores des types (parmi les tumeurs seulement)")

    with gr.Row():
        predict_button = gr.Button("Analyser l'image", elem_id="predict-btn", variant="primary")
        clear_button = gr.Button("Réinitialiser", variant="secondary")

    predict_button.click(
        fn=predict_irm,
        inputs=[image_input],
        outputs=[result_output, probability_output, tumor_type_output, gradcam_output],
        concurrency_limit=1,
    )
    clear_button.click(
        fn=lambda: (None, "", None, None, None),
        inputs=None,
        outputs=[image_input, result_output, probability_output, tumor_type_output, gradcam_output],
    )


if __name__ == "__main__":
    initialize_models()
    server_port = os.environ.get("GRADIO_SERVER_PORT")
    launch_kwargs = {"theme": gr.themes.Soft(), "css": custom_css}
    if server_port:
        launch_kwargs["server_port"] = int(server_port)
    demo.launch(**launch_kwargs)
