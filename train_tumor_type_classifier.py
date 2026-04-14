"""Entrainement multiclasses pour reconnaitre le type de tumeur cerebrale.

Classes ciblees:
- glioma
- meningioma
- pituitary
"""

from pathlib import Path
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import kagglehub

try:
    import h5py
except Exception:
    h5py = None

from model_manager import ModelManager
from utils import plot_training_history


SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 8

DATASET_SOURCES = [
    {
        "name": "sartaj",
        "kaggle_id": "sartajbhuvaji/brain-tumor-classification-mri",
    },
    {
        "name": "figshare_cheng",
        "kaggle_id": "ashkhagan/figshare-brain-tumor-dataset",
        "format": "figshare_mat",
    },
    {
        "name": "mri_7000",
        "kaggle_id": "masoudnickparvar/brain-tumor-mri-dataset",
    },
]

TUMOR_CLASS_NAMES = ["glioma", "meningioma", "pituitary"]
TUMOR_CLASS_TO_INDEX = {name: idx for idx, name in enumerate(TUMOR_CLASS_NAMES)}
FIGSHARE_LABEL_TO_CLASS = {1: "meningioma", 2: "glioma", 3: "pituitary"}

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def _normalize_token(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _infer_tumor_type_from_path(image_path: Path, dataset_dir: Path):
    """Infere le type de tumeur via les noms de dossiers parents."""
    relative_parts = image_path.relative_to(dataset_dir).parts[:-1]

    for part in reversed(relative_parts):
        token = _normalize_token(part)
        if "glioma" in token:
            return "glioma"
        if "meningioma" in token:
            return "meningioma"
        if "pituitary" in token:
            return "pituitary"

    return None


def load_tumor_type_data(dataset_dir: Path):
    image_paths = []
    labels = []
    class_counts = {name: 0 for name in TUMOR_CLASS_NAMES}

    for file_path in dataset_dir.rglob("*"):
        if not file_path.is_file() or not _is_image_file(file_path):
            continue

        tumor_type = _infer_tumor_type_from_path(file_path, dataset_dir)
        if tumor_type is None:
            continue

        image_paths.append(str(file_path))
        labels.append(TUMOR_CLASS_TO_INDEX[tumor_type])
        class_counts[tumor_type] += 1

    if not image_paths:
        raise RuntimeError("Aucune image de type tumoral trouvee. Verifie la structure du dataset.")

    return np.array(image_paths), np.array(labels), class_counts


def _read_figshare_label(mat_path: Path):
    if h5py is None:
        raise RuntimeError("h5py est requis pour lire les fichiers .mat v7.3 du dataset Figshare.")

    with h5py.File(mat_path, "r") as f:
        cj = f.get("cjdata", None)
        if cj is None or "label" not in cj:
            return None

        arr = np.array(cj["label"])
        if arr.size == 0:
            return None

        label_value = int(arr.reshape(-1)[0])
        return FIGSHARE_LABEL_TO_CLASS.get(label_value)


def load_figshare_mat_data(dataset_dir: Path):
    candidate_dirs = [dataset_dir / "dataset" / "data", dataset_dir / "data"]
    data_dir = None
    for candidate in candidate_dirs:
        if candidate.exists():
            data_dir = candidate
            break

    if data_dir is None:
        raise RuntimeError("Dossier data introuvable pour Figshare (attendu: dataset/data ou data).")

    mat_paths = sorted(data_dir.glob("*.mat"))
    if not mat_paths:
        raise RuntimeError("Aucun fichier .mat trouve dans le dataset Figshare.")

    image_paths = []
    labels = []
    class_counts = {name: 0 for name in TUMOR_CLASS_NAMES}

    for mat_path in mat_paths:
        tumor_type = _read_figshare_label(mat_path)
        if tumor_type is None:
            continue

        image_paths.append(str(mat_path))
        labels.append(TUMOR_CLASS_TO_INDEX[tumor_type])
        class_counts[tumor_type] += 1

    if not image_paths:
        raise RuntimeError("Aucune image labelisee utilisable trouvee dans Figshare.")

    return np.array(image_paths), np.array(labels), class_counts


def load_multiple_tumor_datasets(dataset_sources):
    all_image_paths = []
    all_labels = []
    source_summaries = {}

    print("Telechargement et fusion des datasets de types tumoraux...")
    for source in dataset_sources:
        source_name = source["name"]
        source_id = source.get("kaggle_id")
        local_dir = source.get("local_dir")
        source_format = source.get("format", "path_infer")

        if local_dir:
            dataset_path = Path(local_dir)
            if not dataset_path.is_absolute():
                dataset_path = BASE_DIR / dataset_path

            if not dataset_path.exists():
                print(f"- [{source_name}] ignore (dossier absent): {dataset_path}")
                continue
            else:
                print(f"- [{source_name}] chargement local: {dataset_path}")

        elif source_id:
            print(f"- [{source_name}] telechargement: {source_id}")
            dataset_path = Path(kagglehub.dataset_download(source_id))
        else:
            print(f"- [{source_name}] ignore (source invalide: kaggle_id/local_dir manquant)")
            continue

        try:
            if source_format == "figshare_mat":
                image_paths, labels, class_counts = load_figshare_mat_data(dataset_path)
            else:
                image_paths, labels, class_counts = load_tumor_type_data(dataset_path)
        except RuntimeError as error:
            print(f"  -> source ignoree [{source_name}]: {error}")
            continue

        all_image_paths.append(image_paths)
        all_labels.append(labels)
        source_summaries[source_name] = {
            "kaggle_id": source_id,
            "local_dir": local_dir,
            "format": source_format,
            "path": str(dataset_path),
            "num_images": int(len(image_paths)),
            "class_counts": {k: int(v) for k, v in class_counts.items()},
        }

        print(
            f"  -> {len(image_paths)} images "
            f"(glioma={class_counts['glioma']}, meningioma={class_counts['meningioma']}, pituitary={class_counts['pituitary']})"
        )

    if not all_image_paths:
        raise RuntimeError("Aucun dataset multiclasses n'a ete charge.")

    merged_paths = np.concatenate(all_image_paths)
    merged_labels = np.concatenate(all_labels)
    return merged_paths, merged_labels, source_summaries


def decode_and_resize(path, label):
    def _load_image(path_tensor):
        path_str = path_tensor.numpy().decode("utf-8")
        p = Path(path_str)

        if p.suffix.lower() == ".mat":
            if h5py is None:
                raise RuntimeError("h5py est requis pour charger les images .mat Figshare.")

            with h5py.File(p, "r") as f:
                cj = f.get("cjdata", None)
                if cj is None or "image" not in cj:
                    raise RuntimeError(f"Champ cjdata/image introuvable dans {p}")
                image_np = np.array(cj["image"], dtype=np.float32)

            # Normalisation min-max robuste sur chaque image.
            min_v = float(np.min(image_np))
            max_v = float(np.max(image_np))
            if max_v > min_v:
                image_np = (image_np - min_v) / (max_v - min_v)
            else:
                image_np = np.zeros_like(image_np, dtype=np.float32)

            if image_np.ndim == 2:
                image_np = np.stack([image_np, image_np, image_np], axis=-1)
            elif image_np.ndim == 3 and image_np.shape[-1] == 1:
                image_np = np.repeat(image_np, 3, axis=-1)
            elif image_np.ndim != 3 or image_np.shape[-1] != 3:
                raise RuntimeError(f"Format image .mat non supporte: shape={image_np.shape} ({p})")

            return image_np.astype(np.float32)

        with Image.open(p) as image_pil:
            image_pil = image_pil.convert("RGB")
            image_np = np.array(image_pil, dtype=np.float32) / 255.0
            return image_np

    image = tf.py_function(func=_load_image, inp=[path], Tout=tf.float32)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, tf.cast(label, tf.int32)


image_paths, labels, source_summaries = load_multiple_tumor_datasets(DATASET_SOURCES)

global_counts = {
    "glioma": int(np.sum(labels == TUMOR_CLASS_TO_INDEX["glioma"])),
    "meningioma": int(np.sum(labels == TUMOR_CLASS_TO_INDEX["meningioma"])),
    "pituitary": int(np.sum(labels == TUMOR_CLASS_TO_INDEX["pituitary"])),
}

print(f"Images tumorales totales: {len(image_paths)}")
print(
    f"Distribution -> glioma={global_counts['glioma']} | "
    f"meningioma={global_counts['meningioma']} | pituitary={global_counts['pituitary']}"
)

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

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2]),
    y=y_train,
)
class_weight = {i: float(weight) for i, weight in enumerate(class_weights_array)}
print(f"Class weight: {class_weight}")

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
outputs = tf.keras.layers.Dense(len(TUMOR_CLASS_NAMES), activation="softmax")(outputs)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
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

print("\n=== Phase 1: entrainement des couches de classification type ===")
history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1,
)

print("\n=== Phase 2: fine-tuning leger ===")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_initial.epoch[-1] + 1,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1,
)

print("\n=== Evaluation finale type de tumeur ===")
y_true = []
y_pred = []
for images, labels_batch in test_ds:
    predictions = model.predict(images, verbose=0)
    y_true.extend(labels_batch.numpy().astype(int).tolist())
    y_pred.extend(np.argmax(predictions, axis=1).astype(int).tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

acc = accuracy_score(y_true, y_pred)
prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=TUMOR_CLASS_NAMES, zero_division=0, output_dict=True)

print(f"Accuracy:      {acc:.4f}")
print(f"Precision mac: {prec_macro:.4f}")
print(f"Recall mac:    {rec_macro:.4f}")
print(f"F1 mac:        {f1_macro:.4f}")
print(f"\nMatrice de confusion:\n{cm}")
print("\nRapport de classification:")
print(classification_report(y_true, y_pred, target_names=TUMOR_CLASS_NAMES, zero_division=0))

manager = ModelManager(models_dir=MODELS_DIR)
model_path = manager.save_model(model, model_name="brain_tumor_type")

metrics = {
    "accuracy": float(acc),
    "precision_macro": float(prec_macro),
    "recall_macro": float(rec_macro),
    "f1_macro": float(f1_macro),
    "confusion_matrix": cm.tolist(),
    "class_names": TUMOR_CLASS_NAMES,
    "class_distribution_total": global_counts,
    "dataset_sources": source_summaries,
    "classification_report": report,
    "model_path": str(model_path),
}
manager.save_metrics(metrics, output_dir=OUTPUT_DIR, filename="tumor_type_results.json")

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
    output_path=OUTPUT_DIR / "tumor_type_training_history.png",
)

print("\nEntrainement type de tumeur termine.")
