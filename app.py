"""Interface Gradio pour la detection de tumeurs cerebrales sur IRM."""

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

from model_manager import ModelManager


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
BINARY_METRICS_FILE = OUTPUT_DIR / "transfer_results.json"

BINARY_MODEL_NAME = "brain_tumor_transfer"
TYPE_MODEL_NAME = "brain_tumor_type"
TYPE_CLASS_NAMES = ["glioma", "meningioma", "pituitary"]

GRADCAM_ALPHA = 0.52


manager = ModelManager(models_dir=MODELS_DIR)
binary_model = manager.load_model(BINARY_MODEL_NAME)
try:
    tumor_type_model = manager.load_model(TYPE_MODEL_NAME)
except FileNotFoundError:
    tumor_type_model = None


def load_json(path: Path):
    if not path.exists():
        return {}

    try:
        with open(path, "r") as file:
            return json.load(file)
    except Exception:
        return {}


BINARY_METRICS = load_json(BINARY_METRICS_FILE)


def pct(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%".replace(".", ",")


def load_threshold(default=0.5):
    threshold = BINARY_METRICS.get("calibrated_threshold", default)
    try:
        return float(threshold)
    except Exception:
        return float(default)


CALIBRATED_THRESHOLD = load_threshold(default=0.5)
_gradcam_models = {}


def model_input_spec(loaded_model):
    shape = loaded_model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _, height, width, channels = shape
    return int(height), int(width), int(channels)


def preprocess_image_for_model(image: Image.Image, loaded_model) -> np.ndarray:
    if image is None:
        raise ValueError("Ajoute une image IRM avant de lancer l'analyse.")

    height, width, channels = model_input_spec(loaded_model)
    image = image.convert("L" if channels == 1 else "RGB")
    image = image.resize((width, height))

    image_array = np.array(image, dtype=np.float32) / 255.0
    if channels == 1:
        image_array = image_array.reshape(1, height, width, 1)
    else:
        image_array = image_array.reshape(1, height, width, channels)

    return image_array


def call_layer_inference(layer, tensor):
    try:
        return layer(tensor, training=False)
    except TypeError:
        return layer(tensor)


def build_gradcam_model(loaded_model):
    model_key = id(loaded_model)
    if model_key in _gradcam_models:
        return _gradcam_models[model_key]

    input_shape = loaded_model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    grad_input = tf.keras.Input(shape=input_shape[1:])
    x = grad_input
    feature_output = None

    for layer in loaded_model.layers[1:]:
        x = call_layer_inference(layer, x)
        if len(x.shape) == 4:
            feature_output = x

    if feature_output is None:
        raise RuntimeError("Aucune couche convolutive exploitable pour Grad-CAM.")

    gradcam_model = tf.keras.Model(grad_input, [feature_output, x])
    _gradcam_models[model_key] = gradcam_model
    return gradcam_model


def heatmap_to_rgb(heatmap):
    heatmap = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(2.0 * heatmap, 0.0, 1.0)
    green = np.clip(2.0 * heatmap - 1.0, 0.0, 1.0)
    blue = np.zeros_like(heatmap)
    return np.stack([red, green, blue], axis=-1)


def generate_gradcam(image: Image.Image, loaded_model, class_index=None):
    image_array = preprocess_image_for_model(image, loaded_model)
    gradcam_model = build_gradcam_model(loaded_model)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(image_array, training=False)
        if predictions.shape[-1] == 1:
            target_score = predictions[:, 0]
        else:
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]).numpy())
            target_score = predictions[:, int(class_index)]

    gradients = tape.gradient(target_score, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap).numpy()

    source_image = image.convert("RGB")
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255), mode="L")
    heatmap_image = heatmap_image.resize(source_image.size, Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_image, dtype=np.float32) / 255.0

    source_array = np.array(source_image, dtype=np.float32) / 255.0
    heatmap_rgb = heatmap_to_rgb(heatmap_resized)
    alpha_mask = (GRADCAM_ALPHA * heatmap_resized)[..., np.newaxis]
    overlay = source_array * (1.0 - alpha_mask) + heatmap_rgb * alpha_mask

    return Image.fromarray(np.uint8(np.clip(overlay, 0.0, 1.0) * 255))


def predict_tumor_type(image: Image.Image):
    if tumor_type_model is None:
        return None, None, None, {"Non disponible": 1.0}

    image_array = preprocess_image_for_model(image, tumor_type_model)
    probabilities = tumor_type_model.predict(image_array, verbose=0)[0]
    best_idx = int(np.argmax(probabilities))
    best_label = TYPE_CLASS_NAMES[best_idx]
    best_confidence = float(probabilities[best_idx])
    distribution = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(TYPE_CLASS_NAMES)
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
    badge = "TUMEUR DETECTEE" if has_tumor else "AUCUNE TUMEUR DETECTEE"
    title = "Cas suspect" if has_tumor else "IRM classee normale"

    type_line = ""
    if tumor_type_label is not None:
        type_line = (
            "<div class='result-row emphasis'>"
            "<span>Type suspect</span>"
            f"<strong>{tumor_type_label} · {tumor_type_confidence * 100:.2f}%</strong>"
            "</div>"
        )

    return (
        f"<section class='result-card {status_class}'>"
        f"<div class='badge'>{badge}</div>"
        f"<h2>{title}</h2>"
        "<div class='result-row'>"
        "<span>Confiance</span>"
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


def predict_irm(image: Image.Image, sensitive_mode: bool):
    try:
        threshold = max(0.20, CALIBRATED_THRESHOLD - 0.10) if sensitive_mode else CALIBRATED_THRESHOLD

        binary_input = preprocess_image_for_model(image, binary_model)
        tumor_probability = float(binary_model.predict(binary_input, verbose=0)[0][0])
        has_tumor = tumor_probability >= threshold
        confidence = tumor_probability if has_tumor else 1.0 - tumor_probability

        binary_distribution = {
            "Aucune tumeur": 1.0 - tumor_probability,
            "Tumeur": tumor_probability,
        }
        type_distribution = {"Non evalue": 1.0}
        type_label = None
        type_confidence = None
        gradcam_model = binary_model
        gradcam_class_index = None
        gradcam_target = "score tumeur du modele binaire"

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
        return (
            "<section class='result-card danger'>"
            "<div class='badge'>ERREUR</div>"
            "<h2>Analyse impossible</h2>"
            f"<p>{error}</p>"
            "</section>",
            {"Erreur": 1.0},
            {"Erreur": 1.0},
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
    font-family: "Space Grotesk", "Segoe UI", sans-serif;
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
            <p>Analyse binaire, estimation du type tumoral et visualisation Grad-CAM.</p>
            <p class='footer-note'>Prototype pédagogique: ne remplace pas un diagnostic médical.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(elem_classes=["panel"], scale=6):
            image_input = gr.Image(type="pil", label="Image IRM", sources=["upload"], height=380)
            gradcam_output = gr.Image(type="pil", label="Grad-CAM", height=300)
            sensitive_mode_input = gr.Checkbox(
                value=True,
                label="Mode sensible",
                info="Utilise un seuil plus bas pour privilégier le rappel.",
            )

        with gr.Column(elem_classes=["panel"], scale=5):
            result_output = gr.HTML(label="Résultat")
            probability_output = gr.Label(label="Détection binaire")
            tumor_type_output = gr.Label(label="Type tumoral")

    with gr.Row():
        predict_button = gr.Button("Analyser l'image", elem_id="predict-btn", variant="primary")
        clear_button = gr.Button("Réinitialiser", variant="secondary")

    predict_button.click(
        fn=predict_irm,
        inputs=[image_input, sensitive_mode_input],
        outputs=[result_output, probability_output, tumor_type_output, gradcam_output],
    )
    clear_button.click(
        fn=lambda: (None, "", None, None, None),
        inputs=None,
        outputs=[image_input, result_output, probability_output, tumor_type_output, gradcam_output],
    )


if __name__ == "__main__":
    server_port = os.environ.get("GRADIO_SERVER_PORT")
    launch_kwargs = {"theme": gr.themes.Soft(), "css": custom_css}
    if server_port:
        launch_kwargs["server_port"] = int(server_port)
    demo.launch(**launch_kwargs)
