"""Metrics with fixed class definitions and cluster-aware uncertainty."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(y, probabilities, class_names, threshold=0.5):
    y = np.asarray(y)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or y.dtype.kind not in "iu" or p.ndim not in (1, 2) or len(class_names) < 2:
        raise ValueError("Expected integer vector labels and a probability vector/matrix")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and lie in [0,1]")
    binary = len(class_names) == 2 and (p.ndim == 1 or p.shape[-1] == 1)
    p = p.reshape(-1) if binary else p
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected matching nonempty labels and finite probabilities in [0,1]")
    labels = np.arange(len(class_names))
    if not np.isin(y, labels).all():
        raise ValueError("Labels outside the declared class order")
    if not binary and (
        p.shape != (len(y), len(labels)) or not np.allclose(p.sum(axis=1), 1, atol=1e-4)
    ):
        raise ValueError("Expected one normalized probability per declared class")
    pred = (p >= threshold).astype(int) if binary else p.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, labels=labels, average="binary" if binary else "macro", zero_division=0
    )
    result = {
        "n": len(y),
        "class_names": class_names,
        "class_counts": np.bincount(y, minlength=len(labels)).tolist(),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "averaging": "positive_class" if binary else "macro_over_declared_classes",
        "balanced_accuracy": float(balanced_accuracy_score(y, pred))
        if len(np.unique(y)) > 1
        else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y, pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
        ),
    }
    if binary:
        result.update(
            threshold=float(threshold),
            brier_score=float(brier_score_loss(y, p)),
            roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            average_precision=float(average_precision_score(y, p))
            if len(np.unique(y)) == 2
            else None,
        )
        bins = np.minimum((p * 10).astype(int), 9)
        result["ece_10_bins"] = float(
            sum(
                np.mean(bins == k) * abs(p[bins == k].mean() - y[bins == k].mean())
                for k in range(10)
                if np.any(bins == k)
            )
        )
        tn, fp, fn, tp = np.asarray(result["confusion_matrix"]).ravel()
        result["specificity"] = float(tn / (tn + fp)) if tn + fp else None
    return result


def group_bootstrap(y, pred, groups, n_classes, repetitions=2000, seed=42):
    """Resample allocation groups, retaining all slices and their multiplicities.

    Percentile intervals are conditional on this cohort/model, not clinical CIs.
    """
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    if not (len(y) == len(pred) == len(groups)) or not len(y):
        raise ValueError("Matching nonempty arrays required")
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.zeros((len(unique), n_classes * n_classes), dtype=np.int64)
    np.add.at(counts, (inverse, y * n_classes + pred), 1)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        cm = (
            counts[rng.integers(0, len(unique), len(unique))]
            .sum(axis=0)
            .reshape(n_classes, n_classes)
        )
        tp = np.diag(cm)
        precision = np.divide(
            tp, cm.sum(axis=0), out=np.zeros(n_classes), where=cm.sum(axis=0) != 0
        )
        recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(n_classes), where=cm.sum(axis=1) != 0)
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(n_classes),
            where=precision + recall != 0,
        )
        values.append(
            [
                tp.sum() / cm.sum(),
                precision[1] if n_classes == 2 else precision.mean(),
                recall[1] if n_classes == 2 else recall.mean(),
                f1[1] if n_classes == 2 else f1.mean(),
            ]
        )
    intervals = np.quantile(values, [0.025, 0.975], axis=0)
    return {
        "method": "percentile allocation-group bootstrap",
        "repetitions": repetitions,
        "seed": seed,
        "groups": len(unique),
        "confidence_level": 0.95,
        "intervals": {
            name: intervals[:, i].tolist()
            for i, name in enumerate(("accuracy", "precision", "recall", "f1"))
        },
    }


def select_threshold(y_validation, probabilities):
    """Historical rule: maximize validation recall, then precision, then F1.

    This chooses an operating threshold; it does not calibrate probabilities.
    """
    scores = []
    for threshold in np.arange(0.20, 0.81, 0.01):
        result = classification_metrics(
            y_validation, probabilities, ["no_tumor", "tumor"], threshold
        )
        scores.append(((result["recall"], result["precision"], result["f1"]), float(threshold)))
    return max(scores, key=lambda x: x[0])[1]
