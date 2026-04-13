"""Entraînement avancé pour la détection binaire de tumeurs cérébrales.

Pipeline:
- téléchargement du dataset KaggleHub
- split train/validation/test
- augmentation de données
- transfer learning avec MobileNetV2
- fine-tuning léger
- sauvegarde du modèle et des métriques
"""

from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import kagglehub

from model_manager import ModelManager
from utils import plot_training_history


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 8

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_image_paths(folder: Path):
    patterns = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for pattern in patterns:
        paths.extend(folder.glob(pattern))
    return sorted(paths)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def _folder_to_binary_label(folder_name: str):
    #Mappe un nom de classe vers binaire: no_tumor=0, autres tumeurs=1.
    name = folder_name.lower().replace("-", "_").replace(" ", "_")
    if "no_tumor" in name or name in {"no", "normal", "healthy"}:
        return 0
    if any(key in name for key in ["tumor", "tumour", "glioma", "meningioma", "pituitary"]):
        return 1
    return None


def load_filepaths_and_labels(dataset_dir: Path):
    #Charge un dataset multiclasses et le convertit en binaire.
    image_paths = []
    labels = []
    class_counts = {"no_tumor": 0, "tumor": 0}

    # Cas Sartaj: sous-dossiers Training/Testing contenant les classes
    candidate_roots = []
    for split_name in ["Training", "Testing", "train", "test", "validation", "val"]:
        split_dir = dataset_dir / split_name
        if split_dir.exists() and split_dir.is_dir():
            candidate_roots.append(split_dir)

    # Fallback: classes directement à la racine
    if not candidate_roots:
        candidate_roots.append(dataset_dir)

    for root in candidate_roots:
        for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            label = _folder_to_binary_label(class_dir.name)
            if label is None:
                continue

            files = [p for p in class_dir.rglob("*") if p.is_file() and _is_image_file(p)]
            image_paths.extend([str(path) for path in files])
            labels.extend([label] * len(files))

            if label == 0:
                class_counts["no_tumor"] += len(files)
            else:
                class_counts["tumor"] += len(files)

    if not image_paths:
        raise RuntimeError(
            "Aucune image trouvée. Vérifie la structure du dataset téléchargé par KaggleHub."
        )

    return np.array(image_paths), np.array(labels), class_counts


def decode_and_resize(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)


def find_calibrated_threshold(y_true, y_proba):
    #Calibre un seuil en privilégiant le rappel pour limiter les faux négatifs.
    best_threshold = 0.5
    best_recall = -1.0
    best_precision = -1.0
    best_f1 = -1.0

    for threshold in np.arange(0.20, 0.81, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if recall > best_recall:
            best_recall = recall
            best_precision = precision
            best_f1 = f1
            best_threshold = float(threshold)
        elif np.isclose(recall, best_recall):
            if precision > best_precision:
                best_precision = precision
                best_f1 = f1
                best_threshold = float(threshold)
            elif np.isclose(precision, best_precision) and f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)

    return best_threshold


print("Téléchargement du dataset...")
dataset_path = Path(kagglehub.dataset_download("sartajbhuvaji/brain-tumor-classification-mri"))

image_paths, labels, class_counts = load_filepaths_and_labels(dataset_path)
print(f"Images trouvées: {len(image_paths)}")
print(f"Tumeurs: {class_counts['tumor']} | Sans tumeur: {class_counts['no_tumor']}")

X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
    image_paths,
    labels,
    test_size=0.3,
    random_state=SEED,
    stratify=labels,
)
X_val_paths, X_test_paths, y_val, y_test = train_test_split(
    X_temp_paths,
    y_temp,
    test_size=0.5,
    random_state=SEED,
    stratify=y_temp,
)

print(f"Train: {len(X_train_paths)} | Val: {len(X_val_paths)} | Test: {len(X_test_paths)}")

train_ds = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
train_ds = train_ds.shuffle(buffer_size=len(X_train_paths), seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val_paths, y_val))
val_ds = val_ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test_paths, y_test))
test_ds = test_ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
], name="augmentation")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
outputs = augmentation(inputs)
outputs = tf.keras.applications.mobilenet_v2.preprocess_input(outputs * 255.0)
outputs = base_model(outputs, training=False)
outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
outputs = tf.keras.layers.Dropout(0.3)(outputs)
outputs = tf.keras.layers.Dense(128, activation="relu")(outputs)
outputs = tf.keras.layers.Dropout(0.3)(outputs)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(outputs)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    ),
]

print("\n=== Phase 1: entraînement des couches de classification ===")
history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

print("\n=== Phase 2: fine-tuning léger ===")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_initial.epoch[-1] + 1,
    callbacks=callbacks,
    verbose=1,
)

print("\n=== Calibration du seuil (validation) ===")
val_proba = model.predict(val_ds, verbose=0).flatten()
calibrated_threshold = find_calibrated_threshold(y_val, val_proba)
val_pred_calibrated = (val_proba >= calibrated_threshold).astype(int)
val_precision = precision_score(y_val, val_pred_calibrated, zero_division=0)
val_recall = recall_score(y_val, val_pred_calibrated, zero_division=0)
val_f1 = f1_score(y_val, val_pred_calibrated, zero_division=0)

print(f"Seuil calibré retenu: {calibrated_threshold:.2f}")
print(f"Validation - Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

print("\n=== Évaluation finale ===")
y_true = []
y_pred_default = []
y_pred_calibrated = []
for images, labels_batch in test_ds:
    predictions = model.predict(images, verbose=0).flatten()
    y_true.extend(labels_batch.numpy().astype(int).tolist())
    y_pred_default.extend((predictions >= 0.5).astype(int).tolist())
    y_pred_calibrated.extend((predictions >= calibrated_threshold).astype(int).tolist())

y_true = np.array(y_true)
y_pred_default = np.array(y_pred_default)
y_pred_calibrated = np.array(y_pred_calibrated)

acc_default = accuracy_score(y_true, y_pred_default)
prec_default = precision_score(y_true, y_pred_default, zero_division=0)
rec_default = recall_score(y_true, y_pred_default, zero_division=0)
f1_default = f1_score(y_true, y_pred_default, zero_division=0)
cm_default = confusion_matrix(y_true, y_pred_default)

acc = accuracy_score(y_true, y_pred_calibrated)
prec = precision_score(y_true, y_pred_calibrated, zero_division=0)
rec = recall_score(y_true, y_pred_calibrated, zero_division=0)
f1 = f1_score(y_true, y_pred_calibrated, zero_division=0)
cm = confusion_matrix(y_true, y_pred_calibrated)

print("\n--- Métriques test au seuil par défaut 0.50 ---")
print(f"Accuracy:  {acc_default:.4f}")
print(f"Precision: {prec_default:.4f}")
print(f"Recall:    {rec_default:.4f}")
print(f"F1-Score:  {f1_default:.4f}")
print(f"Matrice de confusion:\n{cm_default}")

print(f"\n--- Métriques test au seuil calibré {calibrated_threshold:.2f} ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"Matrice de confusion:\n{cm}")
print("\nRapport de classification (seuil calibré):")
print(classification_report(y_true, y_pred_calibrated, target_names=["No Tumor", "Tumor"]))

manager = ModelManager(models_dir=MODELS_DIR)
model_path = manager.save_model(model, model_name="brain_tumor_transfer")

metrics = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "confusion_matrix": cm.tolist(),
    "calibrated_threshold": float(calibrated_threshold),
    "validation_at_calibrated_threshold": {
        "precision": float(val_precision),
        "recall": float(val_recall),
        "f1_score": float(val_f1),
    },
    "default_threshold_metrics": {
        "threshold": 0.5,
        "accuracy": float(acc_default),
        "precision": float(prec_default),
        "recall": float(rec_default),
        "f1_score": float(f1_default),
        "confusion_matrix": cm_default.tolist(),
    },
    "selected_threshold_metrics": {
        "threshold": float(calibrated_threshold),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    },
    "model_path": str(model_path),
}
manager.save_metrics(metrics, output_dir=OUTPUT_DIR, filename="transfer_results.json")

history_combined = {
    "loss": history_initial.history["loss"] + history_finetune.history["loss"],
    "val_loss": history_initial.history["val_loss"] + history_finetune.history["val_loss"],
    "accuracy": history_initial.history["accuracy"] + history_finetune.history["accuracy"],
    "val_accuracy": history_initial.history["val_accuracy"] + history_finetune.history["val_accuracy"],
}

class HistoryWrapper:
    def __init__(self, history):
        self.history = history

plot_training_history(
    HistoryWrapper(history_combined),
    output_path=OUTPUT_DIR / "transfer_training_history.png",
)

print("\nEntraînement avancé terminé.")
