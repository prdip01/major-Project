"""
gradcam.py
==========
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization.
Generates a heatmap overlay showing which regions of the skin image
influenced the model's prediction most.
Strictly relies on real backpropagation gradients. Mock fallbacks removed.
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
        Base64-encoded PNG string of the Grad-CAM overlay, or None if failed.
    """
    if model is None:
        raise RuntimeError("CNN Model not loaded. Cannot execute Grad-CAM backpropagation.")

    import tensorflow as tf

    # Dynamically find the last convolutional layer (supports custom CNNs and EfficientNet)
    last_conv_layer_name = _find_last_conv_layer(model)
    logger.info(f"🧬 Extracting convolutional features from layer: '{last_conv_layer_name}'")

    # Check if the model is a Keras Sequential model (which requires dynamic layer-by-layer forward execution in Keras 3)
    is_sequential = model.__class__.__name__.lower() == "sequential" or isinstance(model, tf.keras.Sequential)

    if is_sequential:
        logger.info("🤖 Sequential model detected. Executing robust layer-by-layer forward pass.")
        conv_layer_idx = None
        for idx, layer in enumerate(model.layers):
            if layer.name == last_conv_layer_name:
                conv_layer_idx = idx
                break
                
        if conv_layer_idx is None:
            raise RuntimeError(f"Could not find conv layer {last_conv_layer_name} in model.")
            
        # Run forward pass up to the conv layer
        out = tf.cast(img_array, tf.float32)
        for idx in range(conv_layer_idx + 1):
            out = model.layers[idx](out)
        conv_outputs = out
        
        # Tape records remaining layers starting from the conv layer output
        with tf.GradientTape() as tape:
            tape.watch(conv_outputs)
            y = conv_outputs
            for idx in range(conv_layer_idx + 1, len(model.layers)):
                y = model.layers[idx](y)
            predictions = y
            class_score = predictions[:, class_idx]
            
        # Gradients of class score w.r.t last conv output
        grads = tape.gradient(class_score, conv_outputs)
    else:
        logger.info("🕸️ Functional/Subclassed model detected. Using standard Keras model grafting.")
        # Build functional model outputs
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output],
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
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


def _find_last_conv_layer(model) -> str:
    """
    Dynamically search the Keras model in reverse order to find the last 2D Convolutional layer.
    """
    for layer in reversed(model.layers):
        class_name = layer.__class__.__name__.lower()
        layer_name = layer.name.lower()
        # Look for standard 2D Convolutional layers
        if "conv2d" in class_name or "conv2d" in layer_name:
            return layer.name
        if "conv" in class_name or "conv" in layer_name:
            return layer.name
    # Fallback to original default
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
