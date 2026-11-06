import h5py
import numpy as np
import pytest
from PIL import Image
from brain_tumor.preprocessing import read_array, validate_image
from brain_tumor.datasets import figshare_patient, infer_class


def test_decoding_rgb_grayscale_and_lossless_formats(tmp_path):
    image = Image.fromarray(np.arange(32 * 32, dtype=np.uint8).reshape(32, 32))
    a, b = tmp_path / "a.png", tmp_path / "b.bmp"
    image.save(a)
    image.save(b)
    np.testing.assert_array_equal(read_array(a), read_array(b))
    assert read_array(a).shape == (32, 32, 3)
    assert read_array(a).dtype == np.float32


@pytest.mark.parametrize("image", [None, "bad", Image.new("RGB", (2, 2))])
def test_invalid_images(image):
    with pytest.raises(ValueError):
        validate_image(image)


def test_figshare_constant_image_and_padded_patient(tmp_path):
    path = tmp_path / "scan.mat"
    with h5py.File(path, "w") as f:
        f["cjdata/image"] = np.ones((32, 32), dtype=np.float32)
        f["cjdata/PID"] = np.array([65, 66, 0], dtype=np.uint16)
    assert np.isfinite(read_array(path)).all()
    assert not read_array(path).any()
    key = figshare_patient(path)
    assert key.startswith("figshare:") and "AB" not in key


def test_label_inference_uses_explicit_folder_not_filename():
    assert infer_class("/data/Training/no_tumor/glioma_001.jpg") == "no_tumor"
    assert infer_class("/data/Testing/pituitary/001.jpg") == "pituitary"
    assert infer_class("/data/unlabelled/001.jpg") is None
