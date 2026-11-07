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
from brain_tumor.preprocessing import validate_image


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
TYPE_DISPLAY_NAMES = {
    "glioma": "Gliome",
    "meningioma": "Méningiome",
    "pituitary": "Tumeur hypophysaire",
}

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
    gradcam_target="modèle binaire",
):
    title = "Motif tumoral prédit" if has_tumor else "Motif non tumoral prédit"
    type_line = ""
    if tumor_type_label is not None:
        label = html.escape(TYPE_DISPLAY_NAMES.get(tumor_type_label, tumor_type_label))
        raw_label = html.escape(tumor_type_label, quote=True)
        score = f"{tumor_type_confidence * 100:.1f}".replace(".", ",")
        type_line = (
            f"<div class='predicted-type' data-class='{raw_label}'>"
            f"<span>Type prédit</span><strong>{label}</strong>"
            f"<span>Score du type : {score} % · parmi les tumeurs</span></div>"
        )
    elif has_tumor:
        type_line = (
            "<p>Classification du type indisponible. La prédiction binaire reste affichée.</p>"
        )
    else:
        type_line = "<p>La classification du type n’est pas appliquée à cette prédiction.</p>"
    target = gradcam_target
    for raw, display in TYPE_DISPLAY_NAMES.items():
        target = target.replace("classe " + raw, "classe " + display.lower())
    score = f"{confidence * 100:.1f}".replace(".", ",")
    return (
        "<section class='result-card' aria-label='Résultat du modèle'>"
        "<p class='result-status'><span class='status-dot' aria-hidden='true'></span>Analyse terminée</p>"
        f"<h2>{title}</h2>"
        f"{type_line}"
        "<div class='model-score'><span>Score de la prédiction binaire</span>"
        f"<strong>{score} %</strong></div>"
        "<p class='score-caption'>Score non calibré, sans interprétation clinique.</p>"
        "<details class='decision-details'><summary>Détails de la décision</summary>"
        f"<p>Score tumoral : {probability:.4f} · seuil binaire : {threshold:.2f}.</p>"
        f"<p>Grad-CAM : {html.escape(target)}.</p></details>"
        "</section>"
    )


def error_html(message):
    if message == "Image dimensions must be at least 16 px and at most 25 megapixels.":
        message = "Utilisez une image d’au moins 16 × 16 pixels et de 25 mégapixels maximum."
    return (
        "<section class='result-card result-error' role='alert'>"
        "<p class='result-status'>Vérification nécessaire</p>"
        "<h2>Analyse impossible</h2>"
        f"<p>{html.escape(message)}</p></section>"
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
        return error_html(message), None, None, None


EMPTY_RESULT = (
    "<section class='empty-result' role='status'>"
    "<p class='result-status'>En attente d’une image</p>"
    "<h2>Une image, deux étapes d’analyse.</h2>"
    "<p>Importez une IRM puis lancez l’analyse.</p>"
    "<dl class='analysis-outline'><div><dt>Détection binaire</dt>"
    "<dd>Motif tumoral ou non tumoral</dd></div><div><dt>Classification du type</dt>"
    "<dd>Gliome, méningiome ou tumeur hypophysaire, si un motif tumoral est prédit.</dd></div></dl>"
    "</section>"
)
READY_RESULT = (
    "<section class='empty-result' role='status'>"
    "<p class='result-status'>Image importée</p>"
    "<h2>Prête à être analysée.</h2>"
    "<p>Le modèle affichera sa prédiction et une visualisation Grad-CAM.</p>"
    "</section>"
)


def interface_updates(result, binary=None, types=None, cam=None, *, ready=False, clear=False):
    """Adapt model outputs to the UI without changing scientific class keys."""
    display_types = {
        TYPE_DISPLAY_NAMES.get(key, key): value for key, value in (types or {}).items()
    }
    display_binary = (
        {"Motif non tumoral": binary["Aucune tumeur"], "Motif tumoral": binary["Tumeur"]}
        if binary is not None
        else None
    )
    return (
        result,
        display_binary,
        display_types or None,
        cam,
        gr.update(
            value="Analyser l’image",
            interactive=ready,
        ),
        gr.update(interactive=clear),
    )


def reset_analysis(image, session=None):
    """Invalidate previous results whenever the selected input changes."""
    if session is not None:
        session["revision"] += 1
    if image is None:
        return interface_updates(EMPTY_RESULT)
    try:
        validate_image(image)
    except ValueError as error:
        return interface_updates(error_html(str(error)), clear=True)
    return interface_updates(READY_RESULT, ready=True, clear=True)


def clear_interface(session=None):
    return (gr.update(value=None), *reset_analysis(None, session))


def analyze_interface(image, session=None):
    """Never attach a completed prediction to an input replaced during inference."""
    revision = session["revision"] if session is not None else None
    result, binary, types, cam = predict_irm(image)
    if session is not None and session["revision"] != revision:
        return tuple(gr.skip() for _ in range(6))
    return interface_updates(
        result, binary, types, cam, ready=image is not None, clear=image is not None
    )


def model_status_html():
    if binary_model is not None and tumor_type_model is not None:
        message = "Modèles prêts"
    elif binary_model is not None:
        message = "Détection binaire disponible"
    else:
        message = "Modèles à initialiser au lancement"
    dot = "<span class='status-dot' aria-hidden='true'></span>" if binary_model is not None else ""
    return f"<div class='runtime-status' role='status'>{dot}{message}</div>"


def page_header_html():
    return (
        "<header class='app-header'><div class='brand-group'>"
        "<svg class='brand-mark' viewBox='0 0 32 32' fill='none' aria-hidden='true'>"
        "<rect x='4' y='4' width='24' height='24' rx='8' stroke='currentColor'/>"
        "<ellipse cx='16' cy='16' rx='7' ry='10' stroke='currentColor'/>"
        "<path d='M6 16h20M16 6v20' stroke='currentColor'/></svg>"
        "<span class='app-brand'>Brain MRI</span>"
        "<span class='brand-description'>Computer vision · MobileNetV2</span></div>"
        "<div class='header-links'>"
        + model_status_html()
        + "<a class='source-link' href='https://github.com/Irfat69/Projet_Tumeur' "
        "target='_blank' rel='noopener noreferrer'>Code sur GitHub ↗</a></div></header>"
        "<div class='page-intro'><div><h1>Détection de tumeurs cérébrales</h1>"
        "<p>Importez une IRM pour consulter la prédiction du modèle et son attribution Grad-CAM.</p></div>"
        "<a class='evaluation-link' "
        "href='https://github.com/Irfat69/Projet_Tumeur/releases/tag/gradio-demo-v3' "
        "target='_blank' rel='noopener noreferrer'>Consulter l’évaluation ↗</a></div>"
        "<p class='research-notice'><strong>Démonstration pédagogique.</strong> "
        "Sans validation clinique ; ne fournit aucun diagnostic médical.</p>"
    )


def interface_theme():
    tokens = {
        "body_background_fill": "#111516",
        "body_text_color": "#edf1ef",
        "body_text_color_subdued": "#aab5b1",
        "background_fill_primary": "#1a2021",
        "background_fill_secondary": "#151a1b",
        "border_color_primary": "#35403c",
        "border_color_accent": "#9bd6c1",
        "block_background_fill": "#1a2021",
        "block_border_color": "#35403c",
        "block_label_background_fill": "#1a2021",
        "block_label_text_color": "#aab5b1",
        "input_background_fill": "#0c1011",
        "input_border_color": "#35403c",
        "input_placeholder_color": "#aab5b1",
        "button_primary_background_fill": "#9bd6c1",
        "button_primary_background_fill_hover": "#b7e4d3",
        "button_primary_text_color": "#142d24",
        "button_primary_text_color_hover": "#142d24",
        "button_secondary_background_fill": "#262e2c",
        "button_secondary_background_fill_hover": "#333e39",
        "button_secondary_text_color": "#edf1ef",
        "button_secondary_text_color_hover": "#edf1ef",
    }
    return gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Manrope", weights=[400, 500, 600]), "Segoe UI", "sans-serif"],
        font_mono=["ui-monospace", "monospace"],
    ).set(**tokens, **{key + "_dark": value for key, value in tokens.items()})


# Desktop workbench: compare the input and attribution at the same scale, beside
# a fixed results rail. Charcoal surfaces, restrained mint, native Gradio controls.
# Intended for PC demonstration; no mobile breakpoint or alternate mobile layout.
custom_css = """
.gradio-container {
    --page: #111516;
    --surface: #1a2021;
    --line: #35403c;
    --ink: #edf1ef;
    --quiet: #aab5b1;
    --mint: #9bd6c1;
    width: calc(100% - 64px) !important;
    min-width: 1120px !important;
    max-width: 1536px !important;
    margin: 0 auto !important;
    padding: 20px 0 24px !important;
    background: var(--page);
    color: var(--ink);
    font-variant-numeric: tabular-nums;
}
.gradio-container > .main { padding: 0 !important; }
.app-html, .app-html .html-container { padding: 0 !important; background: transparent !important; border: 0 !important; }
.app-header { display: flex; align-items: center; justify-content: space-between; gap: 24px; padding: 0 0 16px; border-bottom: 1px solid var(--line); }
.brand-group, .header-links { display: flex; align-items: center; gap: 24px; }
.brand-group { gap: 12px; }
.brand-mark { width: 32px; height: 32px; color: var(--mint); }
.app-brand { color: var(--ink); font-size: 18px; font-weight: 600; white-space: nowrap; }
.brand-description { color: var(--quiet); font-size: 13px; padding-left: 16px; margin-left: 4px; border-left: 1px solid var(--line); }
.page-intro { display: flex; align-items: center; justify-content: space-between; gap: 24px; padding: 20px 0 12px; }
.page-intro h1 { color: var(--ink); font-size: 30px; font-weight: 600; line-height: 1.25; letter-spacing: -0.025em; margin: 0 0 8px; }
.page-intro p { margin: 0; color: var(--quiet); font-size: 14px; line-height: 1.65; }
.source-link, .evaluation-link { flex-shrink: 0; color: var(--quiet) !important; font-size: 13px; text-decoration: none; text-underline-offset: 5px; transition: color 160ms ease; }
.source-link:hover, .evaluation-link:hover { color: var(--mint) !important; text-decoration: underline; }
.research-notice { color: var(--quiet); font-size: 13px; line-height: 1.65; margin: 0 0 4px; }
.research-notice strong { color: var(--ink); font-weight: 500; }
.runtime-status { display: flex; align-items: center; gap: 8px; color: var(--quiet); font-size: 13px; }
.status-dot { display: inline-block; flex: 0 0 6px; width: 6px; height: 6px; border-radius: 50%; background: var(--mint); }
#workspace { display: grid !important; grid-template-columns: minmax(0, 1fr) 336px; gap: 24px; align-items: stretch; flex-wrap: nowrap !important; margin-top: 12px; }
#viewer, #results-panel { min-width: 0 !important; width: 100%; }
#viewer { gap: 0; }
#image-grid { display: grid !important; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; flex-wrap: nowrap !important; }
.image-column { min-width: 0 !important; gap: 12px !important; }
.panel-heading { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.panel-heading h2 { color: var(--ink); font-size: 15px; line-height: 1.5; font-weight: 500; margin: 0; }
.panel-heading span { color: var(--quiet); font-size: 13px; }
#mri-input, #cam-output { border: 1px solid var(--line); border-radius: 12px; background: #0c1011; }
#mri-input:not(:has(img)) { border-style: dashed; }
#mri-input img, #cam-output img { object-fit: contain; }
#mri-input .upload-container, #cam-output .upload-container { color: var(--quiet); }
#input-actions { display: grid !important; grid-template-columns: 232px 112px minmax(0, 1fr); gap: 12px; align-items: center; flex-wrap: nowrap !important; padding: 20px 0; }
#input-actions button { min-height: 46px; min-width: 0 !important; border-radius: 8px; font-size: 14px; font-weight: 600; transition: background-color 160ms ease, transform 160ms ease; }
#input-actions button:not(:disabled):hover { transform: translateY(-1px); }
#input-actions button:not(:disabled):active { transform: translateY(1px); }
#predict-btn { background: var(--mint); color: #142d24; }
#predict-btn:hover:not(:disabled) { background: #b7e4d3; }
#predict-btn:disabled { opacity: 1; background: #28332f; color: #b1c1b9; border-color: #40534a; cursor: not-allowed; }
#clear-btn:disabled { opacity: 0.55; }
.gradio-container button:focus-visible, .gradio-container a:focus-visible, .gradio-container summary:focus-visible { outline: 2px solid var(--mint); outline-offset: 3px; }
.input-note { justify-self: end; }
.input-note p { margin: 0; color: var(--quiet); font-size: 13px; line-height: 1.6; text-align: right; }
.attribution-note { margin: 0; color: var(--quiet); font-size: 13px; line-height: 1.7; border-top: 1px solid var(--line); padding-top: 16px; }
.attribution-note strong { color: var(--ink); font-weight: 500; }
#results-panel { border-radius: 12px; background: var(--surface); padding: 24px; gap: 20px; }
.results-heading { padding-bottom: 16px; border-bottom: 1px solid var(--line); }
.results-heading h2 { margin: 0; font-size: 15px; font-weight: 500; color: var(--ink); }
#result-region { border: 0; background: transparent; padding: 0; }
.result-card, .empty-result { padding: 0; color: var(--ink); }
.result-status { display: flex; align-items: center; gap: 8px; margin: 0 0 12px; color: var(--quiet); font-size: 13px; }
.result-card h2, .empty-result h2 { color: var(--ink); font-size: 22px; font-weight: 500; letter-spacing: -0.02em; line-height: 1.4; margin: 0 0 16px; text-wrap: pretty; }
.empty-result p:not(.result-status), .result-card > p:not(.result-status) { color: var(--quiet); font-size: 13px; line-height: 1.7; }
.analysis-outline { margin: 28px 0 0; }
.analysis-outline div { padding: 16px 0; border-top: 1px solid var(--line); }
.analysis-outline dt { color: var(--ink); font-size: 14px; font-weight: 500; margin-bottom: 4px; }
.analysis-outline dd { margin: 0; color: var(--quiet); font-size: 13px; line-height: 1.7; }
.predicted-type { display: flex; flex-direction: column; gap: 4px; margin: 0 0 20px; color: var(--quiet); font-size: 13px; }
.predicted-type strong { color: var(--mint); font-size: 28px; line-height: 1.3; font-weight: 500; letter-spacing: -0.02em; }
.model-score { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; border-top: 1px solid var(--line); padding-top: 16px; font-size: 13px; }
.model-score span { color: var(--quiet); max-width: 19ch; }
.model-score strong { color: var(--ink); font-size: 20px; font-weight: 500; white-space: nowrap; }
.score-caption { margin: 6px 0 0 !important; font-size: 13px !important; }
.decision-details { margin-top: 12px; color: var(--quiet); font-size: 13px; line-height: 1.7; }
.decision-details summary { cursor: pointer; padding: 4px 0; }
.decision-details p { margin: 8px 0 0; }
.result-error .result-status { color: #efba9c; }
.score-panel { display: none !important; border: 0 !important; background: transparent !important; box-shadow: none !important; padding: 0 !important; }
#workspace:has(.result-card:not(.result-error)) #binary-scores, #workspace:has(.predicted-type) #type-scores { display: block !important; }
#score-details { display: none !important; border: 1px solid var(--line); border-radius: 8px; background: transparent; }
#workspace:has(.result-card:not(.result-error)) #score-details { display: block !important; }
#score-details > div:not(.label-wrap) { gap: 20px; }
.score-panel label { color: var(--quiet) !important; font-size: 13px !important; background: transparent !important; border: 0 !important; padding-left: 0 !important; }
.score-panel { --font-mono: var(--font); }
#reading-guide { margin-top: 12px; border: 1px solid var(--line); border-radius: 8px; background: transparent; }
#reading-guide > .label-wrap { min-height: 42px; }
#reading-guide .prose { color: var(--quiet); font-size: 13px; line-height: 1.8; max-width: 110ch; }
#reading-guide .prose strong { color: var(--ink); }
@media (prefers-reduced-motion: reduce) {
    #input-actions button { transition: none; transform: none !important; }
    .gradio-container *, .gradio-container *::before, .gradio-container *::after { scroll-behavior: auto !important; }
}
"""

with gr.Blocks(title="Brain MRI — Démonstration de classification") as demo:
    page_header = gr.HTML(
        value=page_header_html, apply_default_css=False, elem_classes=["app-html"]
    )
    with gr.Row(elem_id="workspace"):
        with gr.Column(elem_id="viewer"):
            with gr.Row(elem_id="image-grid"):
                with gr.Column(elem_classes=["image-column"]):
                    gr.HTML(
                        "<div class='panel-heading'><h2>Image IRM</h2><span>Image source</span></div>",
                        apply_default_css=False,
                        elem_classes=["app-html"],
                    )
                    image_input = gr.Image(
                        type="pil",
                        format="png",
                        label="Image IRM à analyser",
                        show_label=False,
                        sources=["upload"],
                        height=360,
                        elem_id="mri-input",
                        placeholder="Déposez une IRM ou cliquez pour l’importer",
                        buttons=["fullscreen"],
                    )
                with gr.Column(elem_classes=["image-column"]):
                    gr.HTML(
                        "<div class='panel-heading'><h2>Grad-CAM</h2><span>Attribution du modèle</span></div>",
                        apply_default_css=False,
                        elem_classes=["app-html"],
                    )
                    gradcam_output = gr.Image(
                        type="pil",
                        format="png",
                        label="Attribution exploratoire Grad-CAM",
                        show_label=False,
                        interactive=False,
                        height=360,
                        elem_id="cam-output",
                        buttons=["fullscreen", "download"],
                        placeholder="La carte apparaît après une analyse",
                    )
            with gr.Row(elem_id="input-actions"):
                predict_button = gr.Button(
                    "Analyser l’image",
                    variant="primary",
                    interactive=False,
                    elem_id="predict-btn",
                    min_width=0,
                )
                clear_button = gr.Button(
                    "Effacer", interactive=False, elem_id="clear-btn", min_width=0
                )
                gr.Markdown(
                    "Image anonymisée\n\nPNG ou JPEG conseillés",
                    elem_classes=["input-note"],
                )
            gr.HTML(
                "<p class='attribution-note'><strong>À propos de Grad-CAM.</strong> "
                "La carte met en évidence les régions qui contribuent à la prédiction. "
                "Elle ne délimite pas une tumeur et n’est pas une localisation médicale validée.</p>",
                apply_default_css=False,
                elem_classes=["app-html"],
            )
        with gr.Column(elem_id="results-panel"):
            gr.HTML(
                "<div class='results-heading'><h2>Résultats du modèle</h2></div>",
                apply_default_css=False,
                elem_classes=["app-html"],
            )
            result_output = gr.HTML(
                value=EMPTY_RESULT,
                elem_id="result-region",
                apply_default_css=False,
                elem_classes=["app-html"],
            )
            with gr.Accordion("Distributions des scores", open=False, elem_id="score-details"):
                probability_output = gr.Label(
                    label="Scores binaires · non calibrés",
                    show_heading=False,
                    num_top_classes=2,
                    elem_id="binary-scores",
                    color="#9bd6c1",
                    elem_classes=["score-panel"],
                )
                tumor_type_output = gr.Label(
                    label="Scores par type · parmi les tumeurs",
                    show_heading=False,
                    num_top_classes=3,
                    elem_id="type-scores",
                    color="#9bd6c1",
                    elem_classes=["score-panel"],
                )
    with gr.Accordion(
        "Comprendre les résultats et les limites", open=False, elem_id="reading-guide"
    ):
        gr.Markdown(
            "Le modèle prédit **tumeur / sans tumeur**, puis un type (**gliome, méningiome ou tumeur hypophysaire**) "
            "uniquement si la classe tumeur est retenue. Le seuil binaire est choisi sur validation et reste fixe.\n\n"
            "Un score élevé ne garantit ni une prédiction correcte ni une probabilité clinique fiable. "
            "L’application ne vérifie pas qu’un fichier représente une IRM et ne rejette pas toutes les images hors distribution. "
            "Grad-CAM reste une attribution exploratoire. **Cette démonstration ne fournit aucun diagnostic médical.**\n\n"
            "[Modèles et évaluation scientifique](https://github.com/Irfat69/Projet_Tumeur/releases/tag/gradio-demo-v3)."
        )
    analysis_session = gr.State({"revision": 0})
    ui_outputs = [
        result_output,
        probability_output,
        tumor_type_output,
        gradcam_output,
        predict_button,
        clear_button,
    ]
    analysis_event = predict_button.click(
        fn=analyze_interface,
        inputs=[image_input, analysis_session],
        outputs=ui_outputs,
        concurrency_limit=1,
        show_progress="full",
        trigger_mode="once",
    )
    image_input.upload(
        fn=reset_analysis,
        inputs=[image_input, analysis_session],
        outputs=ui_outputs,
        queue=False,
        cancels=[analysis_event],
    )
    image_input.clear(
        fn=reset_analysis,
        inputs=[image_input, analysis_session],
        outputs=ui_outputs,
        queue=False,
        cancels=[analysis_event],
    )
    clear_button.click(
        fn=clear_interface,
        inputs=[analysis_session],
        outputs=[image_input, *ui_outputs],
        queue=False,
        cancels=[analysis_event],
    )
    demo.load(fn=page_header_html, outputs=[page_header], queue=False)


if __name__ == "__main__":
    initialize_models()
    server_port = os.environ.get("GRADIO_SERVER_PORT")
    launch_kwargs = {"theme": interface_theme(), "css": custom_css, "footer_links": []}
    if server_port:
        launch_kwargs["server_port"] = int(server_port)
    demo.launch(**launch_kwargs)
