import json
import numpy as np
import pytest
from PIL import Image
from brain_tumor.safe_split import grouped_train_val_test


def fixture_scans(tmp_path, count=80):
    rng = np.random.default_rng(123)
    paths = []
    for i in range(count):
        path = tmp_path / f"{i:03}.png"
        Image.fromarray(rng.integers(0, 256, (40, 40), dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths, np.arange(count, dtype=np.int32) % 2


def test_conflicts_are_quarantined_including_all_copies(tmp_path):
    paths, y = fixture_scans(tmp_path)
    paths.append(paths[0])
    y = np.append(y, 1)
    grouped_train_val_test(paths, y, task_name="conflict", output_dir=tmp_path / "out")
    m = json.loads((tmp_path / "out/split_conflict.json").read_text())
    assert len(m["excluded_items"]) == 2
    assert all(x["path"] != paths[0] for x in m["items"])


def test_input_permutation_does_not_change_assignment(tmp_path):
    paths, y = fixture_scans(tmp_path)
    a = grouped_train_val_test(paths, y, task_name="a", output_dir=tmp_path / "a")
    b = grouped_train_val_test(paths[::-1], y[::-1], task_name="b", output_dir=tmp_path / "b")
    for left, right in zip(a[:-1], b[:-1]):
        np.testing.assert_array_equal(left, right)


def test_precomputed_groups_are_disjoint_and_manifest_cannot_be_replaced(tmp_path):
    paths, y = fixture_scans(tmp_path)
    groups = np.arange(len(paths))
    grouped_train_val_test(
        paths, y, task_name="fixed", output_dir=tmp_path / "out", precomputed_groups=groups
    )
    with pytest.raises(FileExistsError):
        grouped_train_val_test(
            paths,
            y,
            task_name="fixed",
            output_dir=tmp_path / "out",
            seed=991,
            precomputed_groups=groups,
        )


def test_too_few_groups_and_fractional_labels_fail(tmp_path):
    paths, y = fixture_scans(tmp_path)
    with pytest.raises(ValueError, match="five independent"):
        grouped_train_val_test(paths, y, task_name="few", output_dir=tmp_path, precomputed_groups=y)
    with pytest.raises(ValueError, match="integer"):
        grouped_train_val_test(paths, y + 0.1, task_name="fraction", output_dir=tmp_path)
