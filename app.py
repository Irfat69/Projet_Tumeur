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

        result_html = (
            "<div style='padding: 16px; border-radius: 12px; border: 1px solid #2a2a2a; '"
            f"background-color: {'#3a1f1f' if has_tumor else '#1f3a26'}; color: white;'>"
            f"<h3 style='margin: 0 0 8px 0;'>{result_text}</h3>"
            f"<p style='margin: 0 0 6px 0;'>{confidence_text}</p>"
            f"<p style='margin: 0;'>{probability_text}</p>"
            f"<p style='margin: 6px 0 0 0;'>{threshold_text}</p>"
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
.footer-note {
    font-size: 0.92rem;
    color: #9ca3af;
}
"""


with gr.Blocks(title="Détection de tumeur cérébrale - Modèle avancé", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# Détection de tumeur cérébrale - Modèle avancé")
    gr.Markdown(
        "Importe une image IRM 2D et le modèle avancé renvoie un verdict binaire : tumeur ou pas de tumeur."
    )
    gr.Markdown(
        "<div class='footer-note'>Prototype pédagogique. Ce projet ne remplace pas un avis médical.</div>"
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Image IRM", sources=["upload"], height=360)
        with gr.Column():
            sensitive_mode_input = gr.Checkbox(
                value=True,
                label="Mode sensible (réduit les faux négatifs)",
                info="Active un seuil plus bas pour détecter plus de cas suspects.",
            )
            result_output = gr.HTML(label="Résultat")
            probability_output = gr.Label(label="Répartition des classes")

    predict_button = gr.Button("Analyser l'image")
    predict_button.click(
        fn=predict_irm,
        inputs=[image_input, sensitive_mode_input],
        outputs=[result_output, probability_output],
    )


if __name__ == "__main__":
    demo.launch()
