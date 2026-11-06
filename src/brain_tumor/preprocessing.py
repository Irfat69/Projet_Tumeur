"""Explicit image decoding shared by training, evaluation and the demo.

Resizing is TensorFlow bilinear, without fitting statistics on the cohort.
Figshare normalization uses each image's own range, as in the historical run.
"""

from pathlib import Path
import h5py
import numpy as np
from PIL import Image


def read_array(path):
    path = Path(path)
    if path.suffix.lower() == ".mat":
        with h5py.File(path, "r") as f:
            image = np.asarray(f["cjdata/image"], dtype=np.float32)
        if image.ndim != 2 or not np.isfinite(image).all():
            raise ValueError("Expected a finite 2D Figshare image")
        low, high = float(image.min()), float(image.max())
        image = (image - low) / (high - low) if high > low else np.zeros_like(image)
        return np.repeat(image[..., None], 3, axis=-1)
    with Image.open(path) as image:
        return validate_image(image)


def validate_image(image):
    if not isinstance(image, Image.Image):
        raise ValueError("Upload a valid MRI image (PNG or JPEG).")
    if min(image.size) < 16 or image.width * image.height > 25_000_000:
        raise ValueError("Image dimensions must be at least 16 px and at most 25 megapixels.")
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if not np.isfinite(array).all():
        raise ValueError("Image contains non-finite pixels.")
    return array


def resize_array(array, size=224):
    import tensorflow as tf

    return tf.image.resize(
        np.asarray(array, dtype=np.float32), (size, size), method="bilinear"
    ).numpy()


def prepare_image(image, size=224):
    return resize_array(validate_image(image), size)[None, ...]


def legacy_batch(paths, task, size=224):
    """Exactly preserve each historical script's decoder for frozen evaluation."""
    import tensorflow as tf

    arrays = []
    for path in paths:
        if task == "binary":
            image = tf.image.decode_image(
                tf.io.read_file(str(path)), channels=3, expand_animations=False
            )
            array = tf.image.resize(tf.cast(image, tf.float32), (size, size)) / 255.0
        else:
            array = tf.image.resize(read_array(path), (size, size))
        arrays.append(np.asarray(array, dtype=np.float32))
    return np.stack(arrays)
