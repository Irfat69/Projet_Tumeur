"""
Utilitaires pour le projet de détection de tumeurs cérébrales.
Fonctions réutilisables pour chargement, prétraitement et évaluation.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def load_images_from_folder(folder_path, img_size=256, label=0):
    images = []
    labels = []
    folder = Path(folder_path)

    for img_path in sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png"))):
        try:
            img = Image.open(img_path).convert("L")
            img = img.resize((img_size, img_size))
            images.append(np.array(img) / 255.0)
            labels.append(label)
        except Exception as e:
            print(f"Erreur avec {img_path}: {e}")

    return np.array(images), np.array(labels)


def load_dataset(yes_folder, no_folder, img_size=256):
    X_yes, y_yes = load_images_from_folder(yes_folder, img_size=img_size, label=1)
    X_no, y_no = load_images_from_folder(no_folder, img_size=img_size, label=0)

    X = np.concatenate([X_yes, X_no], axis=0)
    y = np.concatenate([y_yes, y_no], axis=0)

    return X, y


def evaluate_model(model, X_test, y_test, model_name="Model", threshold=0.5):
    """Evaluate binary predictions with the shared fixed-label metric contract."""
    from brain_tumor.evaluation import classification_metrics

    probabilities = np.asarray(model.predict(X_test)).reshape(-1)
    metrics = classification_metrics(
        np.asarray(y_test), probabilities, ["no_tumor", "tumor"], threshold
    )
    metrics["f1_score"] = metrics["f1"]  # Historical caller compatibility.
    print(f"{model_name}: accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    return metrics


def plot_training_history(history, output_path=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss pendant l'entraînement")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy pendant l'entraînement")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Graphe sauvegardé à: {output_path}")

    plt.close()


def display_sample_images(yes_folder, no_folder, n_samples=2):
    yes_folder = Path(yes_folder)
    no_folder = Path(no_folder)

    yes_images = list(yes_folder.glob("*.jpg")) + list(yes_folder.glob("*.png"))
    no_images = list(no_folder.glob("*.jpg")) + list(no_folder.glob("*.png"))

    fig, axes = plt.subplots(2, n_samples, figsize=(12, 6))

    for i in range(n_samples):
        img = Image.open(yes_images[i])
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title("Avec tumeur (yes)")
        axes[0, i].axis("off")

        img = Image.open(no_images[i])
        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].set_title("Sans tumeur (no)")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()
