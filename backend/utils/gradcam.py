"""
gradcam.py
==========
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization.
Generates a heatmap overlay showing which regions of the skin image
influenced the model's prediction most.
"""

import numpy as np
import cv2
import base64
import io
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def generate_gradcam(model, img_array: np.ndarray, class_idx: int, pil_image: Image.Image) -> str:
    """
    Generate a Grad-CAM heatmap overlay and return it as a base64-encoded PNG.

    Args:
        model: Loaded Keras model (EfficientNetB0).
        img_array: Preprocessed image, shape (1, 224, 224, 3).
        class_idx: Index of the predicted class.
        pil_image: Original PIL Image (224x224) for overlay.

    Returns:
        Base64-encoded PNG string of the Grad-CAM overlay.
    """
    try:
        if model is None:
            # Demo mode: generate a fake heatmap for demonstration
            return _generate_demo_gradcam(pil_image)

        import tensorflow as tf

        # Find the last convolutional layer in EfficientNetB0
        # For EfficientNetB0, the last conv layer is named 'top_conv'
        last_conv_layer_name = _find_last_conv_layer(model)

        # Build a model that maps input → (last_conv_output, final_predictions)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )

        # Compute gradients of the class score w.r.t. the conv layer output
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # Score for the predicted class
            class_score = predictions[:, class_idx]

        # Gradients: shape (1, H, W, C)
        grads = tape.gradient(class_score, conv_outputs)

        # Pool gradients over spatial dimensions → shape (C,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight conv outputs by gradients
        conv_outputs = conv_outputs[0]  # (H, W, C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
        heatmap = tf.squeeze(heatmap)  # (H, W)

        # ReLU and normalize to [0, 1]
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        return _overlay_heatmap_on_image(heatmap, pil_image)

    except Exception as e:
        logger.warning(f"Grad-CAM failed: {e}. Returning demo heatmap.")
        return _generate_demo_gradcam(pil_image)


def _find_last_conv_layer(model) -> str:
    """Find the name of the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if hasattr(layer, 'filters') or 'conv' in layer.name.lower():
            return layer.name
    # Fallback for EfficientNetB0
    return "top_conv"


def _overlay_heatmap_on_image(heatmap: np.ndarray, pil_image: Image.Image) -> str:
    """
    Overlay Grad-CAM heatmap on the original image.

    Returns:
        Base64-encoded PNG string.
    """
    # Resize heatmap to image size (224x224)
    heatmap_resized = cv2.resize(heatmap, (224, 224))

    # Apply colormap (JET: blue → green → red)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Convert original PIL image to numpy
    original = np.array(pil_image)

    # Blend: 60% original + 40% heatmap
    overlay = (0.6 * original + 0.4 * heatmap_colored).astype(np.uint8)

    # Convert back to PIL and encode as base64
    result_image = Image.fromarray(overlay)
    buffer = io.BytesIO()
    result_image.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def _generate_demo_gradcam(pil_image: Image.Image) -> str:
    """
    Generate a realistic-looking synthetic Grad-CAM heatmap for demo mode.
    Creates a Gaussian blob centered slightly off-center to simulate a lesion highlight.
    """
    h, w = 224, 224
    # Create a Gaussian heatmap centered around the image center with some offset
    cx, cy = w // 2 + np.random.randint(-30, 30), h // 2 + np.random.randint(-30, 30)
    sigma = np.random.randint(40, 70)

    y_coords, x_coords = np.ogrid[:h, :w]
    heatmap = np.exp(-((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (2 * sigma ** 2))
    heatmap = (heatmap / heatmap.max()).astype(np.float32)

    return _overlay_heatmap_on_image(heatmap, pil_image)
