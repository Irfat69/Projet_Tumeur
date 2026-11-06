import numpy as np
import pytest
import tensorflow as tf
from PIL import Image
from brain_tumor.inference import (
    build_gradcam_model,
    compute_gradcam,
    predict_probabilities,
    preprocess_image_for_model,
)


def tiny_model():
    inputs = tf.keras.Input((32, 32, 3))
    x = tf.keras.layers.Conv2D(4, 3, activation="relu")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(1, activation="sigmoid")(x))


def test_preprocessing_and_gradcam_preserve_prediction():
    tf.keras.utils.set_random_seed(42)
    model = tiny_model()
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (48, 64, 3), dtype=np.uint8))
    a = preprocess_image_for_model(image, model)
    assert a.shape == (1, 32, 32, 3)
    heatmap, p = compute_gradcam(image, model, class_index=1)
    expected = predict_probabilities(image, model, 2)
    np.testing.assert_allclose(expected, p, rtol=1e-6, atol=1e-7)
    assert heatmap.shape == (30, 30) and np.isfinite(heatmap).all()
    assert 0 <= heatmap.min() <= heatmap.max() <= 1
    negative, _ = compute_gradcam(image, model, class_index=0)
    assert np.isfinite(negative).all()


def test_gradcam_requires_convolutional_features():
    model = tf.keras.Sequential(
        [
            tf.keras.Input((32, 32, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    with pytest.raises(ValueError, match="convolutional"):
        build_gradcam_model(model)


def test_nested_backbone_and_keras_arithmetic_are_preserved():
    inputs = tf.keras.Input((32, 32, 3))
    backbone = tf.keras.Sequential(
        [tf.keras.Input((32, 32, 3)), tf.keras.layers.Conv2D(4, 3, activation="relu")]
    )
    x = backbone((inputs * 255.0) / 127.5 - 1.0)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(1, activation="sigmoid")(x))
    image = Image.new("RGB", (32, 32), color=(240, 100, 40))
    _, explained = compute_gradcam(image, model)
    np.testing.assert_allclose(explained, predict_probabilities(image, model, 2), atol=1e-6)
