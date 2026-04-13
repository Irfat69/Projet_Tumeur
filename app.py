"""Interface web simple pour la détection binaire de tumeurs cérébrales."""

from pathlib import Path
import json

import gradio as gr
import numpy as np
from PIL import Image

from model_manager import ModelManager


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
METRICS_FILE = OUTPUT_DIR / "transfer_results.json"
MODEL_NAME = "brain_tumor_transfer"


manager = ModelManager(models_dir=MODELS_DIR)
model = manager.load_model(MODEL_NAME)


def _load_calibrated_threshold(default=0.5):
    """Charge le seuil calibré si disponible, sinon retourne le seuil par défaut."""
    if not METRICS_FILE.exists():
        return float(default)

    try:
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)
        threshold = metrics.get("calibrated_threshold", default)
        return float(threshold)
    except Exception:
        return float(default)


CALIBRATED_THRESHOLD = _load_calibrated_threshold(default=0.5)


def _get_model_input_spec(loaded_model):
    """Retourne (height, width, channels) attendus par le modèle."""
    shape = loaded_model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _, height, width, channels = shape
    return int(height), int(width), int(channels)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Prépare l'image pour le modèle Keras."""
    if image is None:
        raise ValueError("Aucune image fournie.")

    height, width, channels = _get_model_input_spec(model)

    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize((width, height))
    image_array = np.array(image, dtype=np.float32) / 255.0

    if channels == 1:
        image_array = image_array.reshape(1, height, width, 1)
    else:
        image_array = image_array.reshape(1, height, width, channels)

    return image_array


def predict_irm(image: Image.Image, sensitive_mode: bool):
    """Retourne un verdict binaire à partir d'une image IRM."""
    try:
        image_array = preprocess_image(image)
        probability = float(model.predict(image_array, verbose=0)[0][0])
        threshold = max(0.20, CALIBRATED_THRESHOLD - 0.10) if sensitive_mode else CALIBRATED_THRESHOLD
        has_tumor = probability >= threshold
        confidence = probability if has_tumor else 1.0 - probability

        result_text = "Tumeur détectée" if has_tumor else "Aucune tumeur détectée"
        confidence_text = f"Confiance: {confidence * 100:.2f}%"
        probability_text = f"Probabilité brute (classe tumeur): {probability:.4f}"
        threshold_text = f"Seuil de décision utilisé: {threshold:.2f}"
        status_badge = "ALERTE" if has_tumor else "RAS"

        result_html = (
            "<div class='result-card "
            f"{'danger' if has_tumor else 'safe'}'>"
            f"<div class='result-badge'>{status_badge}</div>"
            f"<h3>{result_text}</h3>"
            f"<p>{confidence_text}</p>"
            f"<p>{probability_text}</p>"
            f"<p>{threshold_text}</p>"
            "</div>"
        )

        return result_html, {"Aucune tumeur": 1.0 - probability, "Tumeur": probability}
    except Exception as error:
        return (
            f"<div style='padding: 16px; border-radius: 12px; background-color: #4a1f1f; color: white;'>"
            f"Erreur: {error}</div>",
            {"Erreur": 1.0},
        )


custom_css = """
:root {
    --bg-start: #0b1220;
    --bg-end: #0f2f2d;
    --panel: rgba(10, 15, 26, 0.65);
    --panel-border: rgba(255, 255, 255, 0.15);
    --txt: #f4f7fb;
    --muted: #b7c0cf;
    --accent: #21c78a;
    --warn: #ff7b6b;
    --warn-bg: rgba(255, 123, 107, 0.15);
    --ok-bg: rgba(33, 199, 138, 0.15);
}

.gradio-container {
    background: radial-gradient(circle at 10% 10%, #16304c, transparent 35%),
                radial-gradient(circle at 90% 15%, #1b4a47, transparent 30%),
                linear-gradient(140deg, var(--bg-start), var(--bg-end));
    color: var(--txt);
    font-family: "Space Grotesk", "Segoe UI", sans-serif;
}

.hero {
    background: linear-gradient(120deg, rgba(20, 29, 45, 0.82), rgba(14, 57, 54, 0.82));
    border: 1px solid var(--panel-border);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 12px;
    box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
}

.hero h1 {
    margin: 0 0 8px 0;
    font-size: 1.7rem;
    letter-spacing: 0.4px;
}

.hero p {
    margin: 4px 0;
    color: var(--muted);
}

.panel {
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 16px;
    padding: 14px;
    backdrop-filter: blur(4px);
}

.result-card {
    border-radius: 14px;
    border: 1px solid var(--panel-border);
    padding: 14px 14px 10px 14px;
}

.result-card h3 {
    margin: 0 0 8px 0;
}

.result-card p {
    margin: 5px 0;
    color: #e9edf4;
}

.result-card.safe {
    background: var(--ok-bg);
}

.result-card.danger {
    background: var(--warn-bg);
}

.result-badge {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    padding: 3px 9px;
    border-radius: 999px;
    margin-bottom: 10px;
    background: rgba(255, 255, 255, 0.18);
}

#predict-btn {
    border: none !important;
    background: linear-gradient(120deg, #20bb83, #17a49d) !important;
    color: #ffffff !important;
    font-weight: 700;
    letter-spacing: 0.2px;
}

#predict-btn:hover {
    filter: brightness(1.08);
}

.footer-note {
    font-size: 0.9rem;
    color: var(--muted);
}
"""


with gr.Blocks(title="Détection de tumeur cérébrale - Modèle avancé", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
        <div class='hero'>
            <h1>Détection IRM - Tumeur Cérébrale</h1>
            <p>Analyse binaire assistée par deep learning sur image IRM 2D.</p>
            <p class='footer-note'>Prototype pédagogique: ne remplace pas un diagnostic médical.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(elem_classes=["panel"], scale=6):
            image_input = gr.Image(type="pil", label="Image IRM", sources=["upload"], height=380)

        with gr.Column(elem_classes=["panel"], scale=5):
            sensitive_mode_input = gr.Checkbox(
                value=True,
                label="Mode sensible (réduit les faux négatifs)",
                info="Active un seuil plus bas pour détecter plus de cas suspects.",
            )
            result_output = gr.HTML(label="Résultat")
            probability_output = gr.Label(label="Répartition des classes")

    with gr.Row():
        predict_button = gr.Button("Analyser l'image", elem_id="predict-btn", variant="primary")
        clear_button = gr.Button("Réinitialiser", variant="secondary")

    predict_button.click(
        fn=predict_irm,
        inputs=[image_input, sensitive_mode_input],
        outputs=[result_output, probability_output],
    )
    clear_button.click(
        fn=lambda: (None, "", None),
        inputs=None,
        outputs=[image_input, result_output, probability_output],
    )


if __name__ == "__main__":
    demo.launch()
