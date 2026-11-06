"""Versioned MobileNetV2 experiments with explicit, immutable artifacts.

Importing this module never downloads data or trains a model.
"""

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from brain_tumor.datasets import TYPE_CLASSES
from brain_tumor.evaluation import classification_metrics, select_threshold
from brain_tumor.preprocessing import read_array
from brain_tumor.safe_split import grouped_train_val_test

ROOT = Path(__file__).resolve().parents[2]


def build_model(task, seed=42, weights="imagenet"):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.08, seed=seed + 1),
            tf.keras.layers.RandomZoom(0.1, seed=seed + 2),
            tf.keras.layers.RandomTranslation(0.08, 0.08, seed=seed + 3),
        ],
        name="augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights=weights
    )
    backbone.trainable = False
    inputs = tf.keras.Input((224, 224, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(augmentation(inputs) * 255.0)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3, seed=seed + 4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3, seed=seed + 5)(x)
    outputs = tf.keras.layers.Dense(
        1 if task == "binary" else 3, activation="sigmoid" if task == "binary" else "softmax"
    )(x)
    return tf.keras.Model(inputs, outputs), backbone


def make_dataset(paths, labels, batch_size=16, training=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((np.asarray(paths, dtype=str), np.asarray(labels)))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        def read(value):
            return read_array(value.decode("utf-8"))

        image = tf.numpy_function(read, [path], tf.float32)
        image.set_shape([None, None, 3])
        return tf.image.resize(image, (224, 224)), label

    options = tf.data.Options()
    options.deterministic = True
    options.threading.private_threadpool_size = 2
    return (
        ds.map(decode, num_parallel_calls=2, deterministic=True)
        .batch(batch_size)
        .prefetch(1)
        .with_options(options)
    )


def prepare_records(task, sources):
    cache = ROOT / "outputs/audit_data/records.json"
    if not cache.exists() or not (ROOT / "outputs/audit_data/summary.json").exists():
        raise FileNotFoundError("First run: python -m brain_tumor.data_audit --download")
    records = json.loads(cache.read_text())
    # Canonicalize group IDs by content, independently of union-find roots,
    # path locations and the candidate-retrieval traversal order.
    minimum_sha = {}
    for record in records:
        group = record["audit_group"]
        minimum_sha[group] = min(minimum_sha.get(group, record["sha256"]), record["sha256"])
    canonical = {
        group: i
        for i, (group, _) in enumerate(sorted(minimum_sha.items(), key=lambda pair: pair[1]))
    }
    for record in records:
        record["audit_group"] = canonical[record["audit_group"]]
    records = [
        r
        for r in records
        if r["source"] in sources and (task == "binary" or r["class_name"] in TYPE_CLASSES)
    ]
    for r in records:
        r["label"] = (
            int(r["class_name"] != "no_tumor")
            if task == "binary"
            else TYPE_CLASSES.index(r["class_name"])
        )
    by_group = defaultdict(set)
    for r in records:
        by_group[r["audit_group"]].add(r["label"])
    conflicts = {g for g, labels in by_group.items() if len(labels) > 1}
    excluded = [r for r in records if r["audit_group"] in conflicts]
    # Retain all aliases before deduplication; never discard contradictory labels.
    by_sha = defaultdict(list)
    for r in records:
        if r["audit_group"] not in conflicts:
            by_sha[r["sha256"]].append(r)
    representatives = []
    for sha in sorted(by_sha):
        aliases = sorted(by_sha[sha], key=lambda r: (r["source"], r["relative_path"]))
        representative = dict(aliases[0])
        representative["source_aliases"] = sorted({r["source"] for r in aliases})
        representative["original_folder_aliases"] = sorted(
            {r["source"] + "/" + r["original_split"] for r in aliases}
        )
        representatives.append(representative)
    return representatives, excluded


def main(task=None):
    parser = argparse.ArgumentParser(description="Train a leakage-aware MobileNetV2 experiment")
    parser.add_argument("--task", choices=("binary", "multiclass"), default=task or "binary")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs-head", type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=8)
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Dataset source keys; defaults depend on the selected task",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    task = args.task
    if args.sources is None:
        args.sources = (
            ["sartaj", "mri_7000", "negative_boost"]
            if task == "binary"
            else ["sartaj", "mri_7000", "figshare"]
        )
    if args.batch_size < 1 or args.epochs_head < 1 or args.epochs_finetune < 0:
        parser.error(
            "Batch size/head epochs must be positive; fine-tuning epochs must be nonnegative"
        )
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = args.run_dir or ROOT / "outputs/runs" / f"{stamp}_{task}"
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Choose an empty run directory; preserve {run}")
    run.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.require_gpu and not gpus:
        raise RuntimeError("GPU required but unavailable; configure the venv before training")
    print("Execution devices:", gpus if gpus else "CPU", flush=True)
    records, excluded = prepare_records(task, set(args.sources))
    paths = [r["path"] for r in records]
    labels = np.array([r["label"] for r in records], dtype=np.int32)
    split = grouped_train_val_test(
        paths,
        labels,
        task_name=task,
        output_dir=run,
        seed=args.seed,
        precomputed_groups=[r["audit_group"] for r in records],
    )
    (run / "inventory.json").write_text(
        json.dumps({"items": records, "excluded": excluded}, indent=2)
    )
    config = {
        **vars(args),
        "run_dir": str(run),
        "task": task,
        "protocol": "global_provenance_groups_v3",
        "interpretation": "Exploratory internal evaluation; development has already used related public data.",
        "preprocessing": "PIL RGB / Figshare per-image min-max; TF bilinear 224; MobileNet scaling inside model",
        "class_names": ["no_tumor", "tumor"] if task == "binary" else TYPE_CLASSES,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "gpu_devices": [g.name for g in gpus],
        "python": sys.version,
        "tensorflow_build": dict(tf.sysconfig.get_build_info()),
        "code_sha256": {
            f"src/brain_tumor/{name}": hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "training.py",
                "safe_split.py",
                "datasets.py",
                "preprocessing.py",
                "evaluation.py",
            )
        },
    }
    (run / "config.json").write_text(json.dumps(config, indent=2))
    if args.prepare_only or os.environ.get("MRI_SPLIT_ONLY") == "1":
        print("Prepared immutable experiment:", run, flush=True)
        return
    x_train, y_train, x_val, y_val, x_test, y_test, _ = split
    train_ds = make_dataset(x_train, y_train, args.batch_size, True, args.seed)
    val_ds = make_dataset(x_val, y_val, args.batch_size)
    test_ds = make_dataset(x_test, y_test, args.batch_size)
    classes = np.unique(y_train)
    class_weight = dict(
        zip(classes.tolist(), compute_class_weight("balanced", classes=classes, y=y_train).tolist())
    )
    model, backbone = build_model(task, args.seed)
    loss = "binary_crossentropy" if task == "binary" else "sparse_categorical_crossentropy"
    histories, candidates = {}, []
    for phase, epochs, lr in [
        ("head", args.epochs_head, 1e-3),
        ("finetune", args.epochs_finetune, 1e-5),
    ]:
        if not epochs:
            continue
        if phase == "finetune":
            backbone.trainable = True
            for layer in backbone.layers[:-20]:
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
        checkpoint = run / f"best_{phase}.weights.h5"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_loss", save_best_only=True, save_weights_only=True
            ),
            tf.keras.callbacks.CSVLogger(run / f"{phase}.csv"),
        ]
        h = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2,
        )
        histories[phase] = {k: [float(v) for v in values] for k, values in h.history.items()}
        candidates.append((min(histories[phase]["val_loss"]), checkpoint, phase))
        (run / "history.json").write_text(json.dumps(histories, indent=2, allow_nan=False))
    best_loss, best_weights, phase = min(candidates, key=lambda x: x[0])
    model.load_weights(best_weights)
    val_p = model.predict(val_ds, verbose=0)
    threshold = select_threshold(y_val, val_p.ravel()) if task == "binary" else 0.5
    # Final test is used once, only after training/model/threshold selection.
    test_p = model.predict(test_ds, verbose=0)
    if task == "binary":
        test_p = test_p.ravel()
    model.save(run / "model.keras")
    model_sha = hashlib.file_digest((run / "model.keras").open("rb"), "sha256").hexdigest()
    metrics = classification_metrics(y_test, test_p, config["class_names"], threshold)
    metrics.update(
        protocol=config["protocol"],
        interpretation=config["interpretation"],
        decision_threshold=threshold,
        model_sha256=model_sha,
        selection_phase=phase,
        selection_validation_loss=best_loss,
        split_sha256=hashlib.sha256((run / f"split_{task}.json").read_bytes()).hexdigest(),
    )
    np.savez(run / "test_predictions.npz", labels=y_test, probabilities=test_p, paths=x_test)
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print("Completed experiment:", run, metrics["accuracy"], flush=True)
