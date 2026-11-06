"""Inference and Grad-CAM for the project's sequential MobileNetV2 graph."""

from functools import lru_cache
import numpy as np
import tensorflow as tf
from PIL import Image

from brain_tumor.preprocessing import prepare_image


def model_input_spec(model):
    shape = model.input_shape
    if isinstance(shape, list) or len(shape) != 4 or shape[-1] != 3 or shape[1] != shape[2]:
        raise ValueError("Expected one square RGB model input")
    return int(shape[1]), int(shape[2]), 3


def preprocess_image_for_model(image, model):
    size, _, _ = model_input_spec(model)
    return prepare_image(image, size)


def predict_probabilities(image, model, classes):
    scores = np.asarray(model(preprocess_image_for_model(image, model), training=False))[0]
    expected = 1 if classes == 2 else classes
    if (
        scores.shape != (expected,)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("Model output does not match the declared classes")
    if expected > 1 and not np.isclose(scores.sum(), 1, atol=1e-4):
        raise ValueError("Invalid multiclass probability distribution")
    return scores


@lru_cache(maxsize=4)
def build_gradcam_model(model):
    def contains_convolution(operation):
        return isinstance(operation, tf.keras.layers.Conv2D) or (
            isinstance(operation, tf.keras.Model)
            and any(contains_convolution(layer) for layer in operation.layers)
        )

    def find_feature(tensor):
        # Keras 3 stores arithmetic (Multiply/Divide/Subtract) as Operations,
        # absent from model.layers. Traverse the saved symbolic graph instead
        # of replaying layers, which silently skips MobileNet preprocessing.
        history = getattr(tensor, "_keras_history", None)
        if history is None:
            return None
        operation = history.operation
        if len(tensor.shape) == 4 and contains_convolution(operation):
            return tensor
        node = operation._inbound_nodes[history.node_index]
        for parent in node.input_tensors:
            result = find_feature(parent)
            if result is not None:
                return result
        return None

    feature = find_feature(model.outputs[0])
    if feature is None:
        raise ValueError("No convolutional feature map is available for Grad-CAM")
    return tf.keras.Model(model.inputs, [feature, model.outputs[0]])


def compute_gradcam(image, model, class_index=None):
    inputs = tf.convert_to_tensor(preprocess_image_for_model(image, model))
    grad_model = build_gradcam_model(model)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        features, predictions = grad_model(inputs, training=False)
        if predictions.shape[-1] == 1:
            index = 1 if class_index is None else int(class_index)
            if index not in (0, 1):
                raise ValueError("Binary Grad-CAM class must be 0 or 1")
            target = predictions[:, 0] if index == 1 else 1 - predictions[:, 0]
        else:
            index = int(tf.argmax(predictions[0])) if class_index is None else int(class_index)
            if not 0 <= index < predictions.shape[-1]:
                raise ValueError("Invalid Grad-CAM class")
            target = predictions[:, index]
    gradients = tape.gradient(target, features)
    if gradients is None:
        raise RuntimeError("Grad-CAM graph is disconnected")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.nn.relu(tf.reduce_sum(features * weights, axis=-1))[0]
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap)).numpy()
    if not np.isfinite(heatmap).all():
        raise RuntimeError("Non-finite Grad-CAM attribution")
    return heatmap, predictions.numpy()[0]


def generate_gradcam(image: Image.Image, model, class_index=None):
    heatmap, _ = compute_gradcam(image, model, class_index)
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
        image.size, Image.Resampling.BILINEAR
    )
    h = np.asarray(heatmap_image, dtype=np.float32) / 255
    color = np.stack([np.clip(2 * h, 0, 1), np.clip(2 * h - 1, 0, 1), np.zeros_like(h)], axis=-1)
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    alpha = (0.52 * h)[..., None]
    return Image.fromarray(np.uint8(np.clip(source * (1 - alpha) + color * alpha, 0, 1) * 255))
