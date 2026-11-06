"""Small deterministic regression tests for MRI split leakage prevention."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import h5py
import numpy as np
from PIL import Image

from brain_tumor.safe_split import grouped_train_val_test


class SplitTests(unittest.TestCase):
    def test_exact_visual_duplicates_remain_in_same_partition(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(81)
            paths = []
            labels = []
            for i in range(80):
                pixels = rng.integers(0, 256, (96, 96), dtype=np.uint8)
                p = root / f"scan_{i}.png"
                Image.fromarray(pixels).save(p)
                paths.append(str(p))
                labels.append(i % 2)

                # The same pixels saved to a different *file* must never
                # cross from train to validation or test.
                if i in (3, 21, 44, 67):
                    copy = root / f"scan_{i}_copy.bmp"
                    Image.fromarray(pixels).save(copy)
                    paths.append(str(copy))
                    labels.append(i % 2)

            first = grouped_train_val_test(
                paths, labels, task_name="test", output_dir=root / "outputs"
            )
            summary = first[-1]
            self.assertGreaterEqual(summary["exact_pixel_links"], 4)

            manifest = json.loads((root / "outputs" / "split_test.json").read_text())
            assignment = {Path(item["path"]).name: item["split"] for item in manifest["items"]}
            for i in (3, 21, 44, 67):
                self.assertEqual(assignment[f"scan_{i}.png"], assignment[f"scan_{i}_copy.bmp"])

            second = grouped_train_val_test(
                paths, labels, task_name="test", output_dir=root / "outputs"
            )
            for a, b in zip(first[:-1], second[:-1]):
                np.testing.assert_array_equal(a, b)

    def test_figshare_patient_id_prevents_cross_split_leak(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(321)
            paths, labels = [], []
            for patient in range(60):
                for scan in range(2):
                    path = root / f"patient_{patient}_slice_{scan}.mat"
                    # Different images from the same patient do not need to
                    # look alike to share a partition.
                    pixels = rng.integers(0, 256, (48, 48)).astype(np.float32)
                    with h5py.File(path, "w") as file:
                        group = file.create_group("cjdata")
                        group.create_dataset("image", data=pixels)
                        group.create_dataset(
                            "PID", data=np.array([ord(c) for c in f"PAT{patient}"], dtype=np.uint16)
                        )
                    paths.append(str(path))
                    labels.append(patient % 3)

            grouped_train_val_test(paths, labels, task_name="patients", output_dir=root / "outputs")
            manifest = json.loads((root / "outputs" / "split_patients.json").read_text())
            by_path = {Path(item["path"]).name: item["split"] for item in manifest["items"]}
            for patient in range(60):
                self.assertEqual(
                    by_path[f"patient_{patient}_slice_0.mat"],
                    by_path[f"patient_{patient}_slice_1.mat"],
                )


if __name__ == "__main__":
    unittest.main()
