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
    "<h2>Votre analyse apparaîtra ici.</h2>"
    "<p>Importez une image, puis lancez l’analyse pour explorer la prédiction du modèle.</p>"
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
    return (
        "<div class='runtime-status' role='status'>"
        f"{dot}{message}<span class='model-name'>MobileNetV2</span></div>"
    )


def page_header_html():
    return (
        "<header class='app-header'><div>"
        "<div class='app-brand'>Brain MRI</div>"
        "<h1>Explorer une IRM cérébrale.</h1>"
        "<p>Détection de tumeurs, classification et Grad-CAM.</p></div>"
        "<a class='source-link' href='https://github.com/Irfat69/Projet_Tumeur' "
        "target='_blank' rel='noopener noreferrer'>Voir sur GitHub ↗</a></header>"
        "<p class='research-notice'><strong>Démonstration pédagogique, sans validation clinique.</strong> "
        "Les scores ne sont pas des probabilités cliniques ; Grad-CAM n’est pas une localisation médicale validée.</p>"
        + model_status_html()
    )


def interface_theme():
    tokens = {
        "body_background_fill": "#0c161c",
        "body_text_color": "#edf3f5",
        "body_text_color_subdued": "#acbdc7",
        "background_fill_primary": "#14232c",
        "background_fill_secondary": "#101d25",
        "border_color_primary": "#304651",
        "border_color_accent": "#74d4b0",
        "block_background_fill": "#14232c",
        "block_border_color": "#304651",
        "block_label_background_fill": "#14232c",
        "block_label_text_color": "#b8c9d1",
        "input_background_fill": "#0c171e",
        "input_border_color": "#3a535e",
        "input_placeholder_color": "#b4c4ce",
        "button_primary_background_fill": "#74d4b0",
        "button_primary_background_fill_hover": "#96e2c6",
        "button_primary_text_color": "#0b261d",
        "button_primary_text_color_hover": "#0b261d",
        "button_secondary_background_fill": "#1a2c36",
        "button_secondary_background_fill_hover": "#263e49",
        "button_secondary_text_color": "#edf3f5",
        "button_secondary_text_color_hover": "#edf3f5",
    }
    return gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
        font=["system-ui", "Segoe UI", "sans-serif"],
        font_mono=["ui-monospace", "monospace"],
    ).set(**tokens, **{key + "_dark": value for key, value in tokens.items()})


custom_css = """
.gradio-container {
    --page: #0c161c;
    --surface: #14232c;
    --line: #304651;
    --ink: #edf3f5;
    --quiet: #acbdc7;
    --mint: #74d4b0;
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: clamp(20px, 3vw, 32px) clamp(16px, 2.5vw, 28px) 24px !important;
    background: var(--page);
    color: var(--ink);
}
.gradio-container > .main { padding: 0 !important; }
.app-html, .app-html .html-container { padding: 0 !important; background: transparent !important; border: 0 !important; }
.app-header { display: flex; align-items: start; justify-content: space-between; gap: 24px; padding-bottom: 18px; }
.app-brand { margin: 0 0 12px; font-size: 16px; font-weight: 650; color: var(--mint); }
.app-header h1 { margin: 0 0 10px; color: var(--ink); font-size: clamp(26px, 3.2vw, 36px); font-weight: 600; line-height: 1.2; letter-spacing: -0.025em; }
.app-header p { margin: 0; color: var(--quiet); font-size: 15px; line-height: 1.6; }
.source-link { flex-shrink: 0; padding: 10px 0; color: var(--quiet) !important; font-size: 14px; text-decoration: none; }
.source-link:hover { color: var(--mint) !important; text-decoration: underline; }
.research-notice { color: var(--quiet); font-size: 13px; line-height: 1.7; margin: 0 0 16px; }
.research-notice strong { color: var(--ink); font-weight: 600; }
.runtime-status { display: flex; align-items: center; gap: 9px; color: var(--quiet); font-size: 13px; padding: 0 0 4px; }
.status-dot { display: inline-block; flex: 0 0 6px; width: 6px; height: 6px; border-radius: 50%; background: var(--mint); }
.model-name { margin-left: auto; }
#workspace { border: 1px solid var(--line); border-radius: 16px; background: var(--surface); padding: 24px; gap: 28px; align-items: flex-start; }
#viewer { gap: 14px; }
#viewer-tabs { border: 0; }
#viewer-tabs [role="tab"] { min-height: 44px; padding: 10px 14px; font-size: 14px; }
#viewer-tabs [role="tabpanel"] { padding: 14px 0 0; border: 0; }
#mri-input, #cam-output { border: 1px solid var(--line); border-radius: 12px; background: #080f14; }
#mri-input img, #cam-output img { object-fit: contain; }
#input-actions { gap: 10px; }
#input-actions button { min-height: 46px; border-radius: 10px; font-size: 14px; font-weight: 600; }
#predict-btn { background: var(--mint); color: #0b261d; }
#predict-btn:hover:not(:disabled) { background: #96e2c6; }
#predict-btn:disabled { opacity: 1; background: #263b44; color: #b8cbd3; border-color: #36505b; cursor: not-allowed; }
#clear-btn:disabled { opacity: 0.65; }
.gradio-container button:focus-visible, .gradio-container a:focus-visible, .gradio-container summary:focus-visible { outline: 2px solid var(--mint); outline-offset: 3px; }
.input-note p, .explanation-note p { margin: 0; color: var(--quiet); font-size: 12px; line-height: 1.7; }
#results-panel { border-left: 1px solid var(--line); padding-left: 28px; gap: 22px; }
#result-region { border: 0; background: transparent; padding: 0; }
.result-card, .empty-result { padding: 0; color: var(--ink); }
.result-status { display: flex; align-items: center; gap: 9px; margin: 4px 0 14px; color: var(--quiet); font-size: 13px; }
.result-card h2, .empty-result h2 { color: var(--ink); font-size: 25px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.25; margin: 0 0 16px; }
.empty-result { min-height: 230px; padding-top: 10px; }
.empty-result p:not(.result-status), .result-card > p:not(.result-status) { color: var(--quiet); font-size: 14px; line-height: 1.7; max-width: 45ch; }
.predicted-type { display: flex; flex-direction: column; gap: 5px; margin: 0 0 22px; color: var(--quiet); font-size: 13px; }
.predicted-type strong { color: var(--mint); font-size: 22px; font-weight: 600; }
.model-score { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; border-top: 1px solid var(--line); padding-top: 16px; font-size: 14px; }
.model-score span { color: var(--quiet); }
.model-score strong { color: var(--ink); font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.score-caption { margin: 6px 0 0 !important; font-size: 12px !important; }
.decision-details { margin-top: 16px; color: var(--quiet); font-size: 12px; line-height: 1.7; }
.decision-details summary { cursor: pointer; padding: 5px 0; }
.decision-details p { margin: 6px 0 0; }
.result-error .result-status { color: #f4bc8a; }
.score-panel { display: none !important; border: 0 !important; background: transparent !important; box-shadow: none !important; padding: 0 !important; }
#workspace:has(.result-card:not(.result-error)) #binary-scores, #workspace:has(.predicted-type) #type-scores { display: block !important; }
.score-panel label { color: var(--quiet) !important; font-size: 13px !important; }
#reading-guide { margin-top: 12px; border-color: var(--line); border-radius: 12px; background: transparent; }
#reading-guide .prose { color: var(--quiet); font-size: 13px; line-height: 1.8; }
#reading-guide .prose strong { color: var(--ink); }
@media (max-width: 760px) {
    .app-header { gap: 12px; padding-bottom: 18px; }
    .app-header h1 { max-width: 18ch; }
    .source-link { font-size: 12px; }
    #workspace { padding: 16px; gap: 22px; flex-direction: column; }
    #viewer, #results-panel { flex: none !important; min-width: 0 !important; width: 100% !important; max-width: 100%; }
    #results-panel { border-left: 0; border-top: 1px solid var(--line); padding-left: 0; padding-top: 22px; }
    #mri-input, #cam-output { height: 300px !important; }
    .empty-result { min-height: 130px; }
    .result-card h2, .empty-result h2 { font-size: 23px; }
}
@media (prefers-reduced-motion: reduce) {
    .gradio-container *, .gradio-container *::before, .gradio-container *::after { scroll-behavior: auto !important; }
}
"""


with gr.Blocks(title="Brain MRI — Démonstration de classification") as demo:
    page_header = gr.HTML(
        value=page_header_html, apply_default_css=False, elem_classes=["app-html"]
    )
    with gr.Row(elem_id="workspace"):
        with gr.Column(scale=6, min_width=280, elem_id="viewer"):
            with gr.Tabs(selected="source", elem_id="viewer-tabs") as viewer_tabs:
                with gr.Tab("Image IRM", id="source"):
                    image_input = gr.Image(
                        type="pil",
                        format="png",
                        label="Image IRM à analyser",
                        show_label=False,
                        sources=["upload"],
                        height=360,
                        elem_id="mri-input",
                        placeholder="Déposez une image ou cliquez pour l’importer",
                        buttons=["fullscreen"],
                    )
                with gr.Tab("Grad-CAM", id="explanation", render_children=True):
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
                    gr.Markdown(
                        "Cette carte illustre l’attribution du modèle. Elle ne délimite pas une tumeur.",
                        elem_classes=["explanation-note"],
                    )
            with gr.Row(elem_id="input-actions"):
                predict_button = gr.Button(
                    "Analyser l’image",
                    variant="primary",
                    interactive=False,
                    elem_id="predict-btn",
                    scale=3,
                    min_width=160,
                )
                clear_button = gr.Button(
                    "Effacer", interactive=False, elem_id="clear-btn", scale=1, min_width=80
                )
            gr.Markdown(
                "Utilisez une image anonymisée. PNG ou JPEG conseillés.",
                elem_classes=["input-note"],
            )
        with gr.Column(scale=5, min_width=280, elem_id="results-panel"):
            result_output = gr.HTML(
                value=EMPTY_RESULT,
                elem_id="result-region",
                apply_default_css=False,
                elem_classes=["app-html"],
            )
            probability_output = gr.Label(
                label="Scores binaires · non calibrés",
                show_heading=False,
                num_top_classes=2,
                elem_id="binary-scores",
                color="#74d4b0",
                elem_classes=["score-panel"],
            )
            tumor_type_output = gr.Label(
                label="Scores par type · parmi les tumeurs",
                show_heading=False,
                num_top_classes=3,
                elem_id="type-scores",
                color="#74d4b0",
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
        js="(...args) => { document.querySelector('#viewer-tabs [role=tab]')?.click(); return args; }",
    )
    demo.load(fn=page_header_html, outputs=[page_header], queue=False)


if __name__ == "__main__":
    initialize_models()
    server_port = os.environ.get("GRADIO_SERVER_PORT")
    launch_kwargs = {"theme": interface_theme(), "css": custom_css, "footer_links": []}
    if server_port:
        launch_kwargs["server_port"] = int(server_port)
    demo.launch(**launch_kwargs)
