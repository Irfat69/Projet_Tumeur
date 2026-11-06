"""
Gestion du modèle: sauvegarde, chargement et prédictions.
"""

import json
from pathlib import Path
from tensorflow import keras


class ModelManager:
    """Gestionnaire pour sauvegarder et charger les modèles Keras."""

    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name="brain_tumor_model"):
        """
        Sauvegarde le modèle entraîné.

        Paramètres:
        - model: modèle Keras à sauvegarder
        - model_name: nom du modèle (sans extension)

        Retour:
        - chemin du fichier sauvegardé
        """
        model_path = self.models_dir / f"{model_name}.keras"
        if model_path.exists():
            raise FileExistsError(
                f"Preserve existing models; choose a new name or run directory: {model_path}"
            )
        model.save(model_path)
        print(f"Modèle sauvegardé à: {model_path}")
        return model_path

    def load_model(self, model_name="brain_tumor_model"):
        """
        Charge un modèle entraîné.

        Paramètres:
        - model_name: nom du modèle à charger (sans extension)

        Retour:
        - modèle Keras chargé
        """
        model_path = self.models_dir / f"{model_name}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")

        model = keras.models.load_model(model_path, compile=False)
        print(f"Modèle chargé depuis: {model_path}")
        return model

    def save_metrics(self, metrics, output_dir="outputs", filename="results.json"):
        """
        Sauvegarde les métriques d'évaluation.

        Paramètres:
        - metrics: dictionnaire des métriques
        - output_dir: répertoire de sortie
        - filename: nom du fichier
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metrics_path = output_path / filename
        if metrics_path.exists():
            raise FileExistsError(
                f"Preserve existing results; choose a new run directory: {metrics_path}"
            )
        encoded = json.dumps(metrics, indent=4, allow_nan=False)
        with open(metrics_path, "x", encoding="utf-8") as f:
            f.write(encoded)

        print(f"Métriques sauvegardées à: {metrics_path}")
        return metrics_path

    def load_metrics(self, output_dir="outputs", filename="results.json"):
        """
        Charge les métriques sauvegardées.

        Paramètres:
        - output_dir: répertoire contenant les métriques
        - filename: nom du fichier

        Retour:
        - dictionnaire des métriques
        """
        metrics_path = Path(output_dir) / filename
        if not metrics_path.exists():
            raise FileNotFoundError(f"Métriques non trouvées: {metrics_path}")

        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        print(f"Métriques chargées depuis: {metrics_path}")
        return metrics
