import numpy as np
from PIL import Image
from brain_tumor.training import make_dataset


def test_training_dataset_decodes_without_changing_validation_order(tmp_path):
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (32, 40), color=(i * 80, 20, 30)).save(path)
        paths.append(str(path))
    images, labels = next(iter(make_dataset(paths, np.array([0, 1, 0]), batch_size=3)))
    assert tuple(images.shape) == (3, 224, 224, 3)
    np.testing.assert_array_equal(labels.numpy(), [0, 1, 0])
    assert np.isfinite(images.numpy()).all() and images.numpy().max() <= 1
