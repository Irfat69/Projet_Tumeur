import numpy as np
import pytest
from brain_tumor.evaluation import classification_metrics, group_bootstrap, select_threshold


def test_binary_metrics_and_fixed_confusion_shape():
    result = classification_metrics(np.array([0, 0, 1, 1]), [0.1, 0.8, 0.2, 0.9], ["no", "yes"])
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["accuracy"] == result["precision"] == result["recall"] == result["f1"] == 0.5
    absent = classification_metrics(np.array([1, 1]), [0.1, 0.9], ["no", "yes"])
    assert absent["confusion_matrix"] == [[0, 0], [1, 1]]
    assert absent["specificity"] is None and absent["roc_auc"] is None


@pytest.mark.parametrize(
    "y,p",
    [
        ([], []),
        ([0, 1], [np.nan, 0.8]),
        ([0, 1], [-0.1, 0.8]),
        ([0.2, 1.0], [0.1, 0.8]),
        ([0, 1], [0.8]),
    ],
)
def test_invalid_metrics_input(y, p):
    with pytest.raises(ValueError):
        classification_metrics(np.asarray(y), p, ["no", "yes"])


def test_multiclass_uses_declared_order_even_when_class_absent():
    result = classification_metrics(
        np.array([0, 2]), [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], ["g", "m", "p"]
    )
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 0, 0], [0, 1, 0]]
    with pytest.raises(ValueError):
        classification_metrics(np.array([0]), [[0.8, 0.8, 0.1]], ["g", "m", "p"])


def test_threshold_is_selected_using_validation_only():
    assert np.isclose(select_threshold(np.array([0, 1, 1]), [0.205, 0.4, 0.215]), 0.21)


def test_cluster_bootstrap_preserves_slice_dependence():
    y = np.array([0] * 20 + [1] * 20)
    pred = np.array([0] * 40)
    groups = np.array([0] * 20 + [1] * 20)
    a = group_bootstrap(y, pred, groups, 2, repetitions=500)
    b = group_bootstrap(y, pred, groups, 2, repetitions=500)
    assert a == b and a["groups"] == 2
    assert a["intervals"]["accuracy"] == [0.0, 1.0]
