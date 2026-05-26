"""
preprocess.py
=============
Image preprocessing pipeline for skin cancer classification.
Resizes, normalizes, and prepares images for EfficientNetB0 inference.
Includes image parameter validations (resolution size guards).
"""

import numpy as np
from PIL import Image
import io

# Target input size for EfficientNetB0
IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess a raw image (bytes) for model inference.

    Steps:
      1. Open with PIL and convert to RGB (handles RGBA, grayscale, etc.)
      2. Resize to 224x224
      3. Normalize pixel values to [0, 1]
      4. Add batch dimension → shape (1, 224, 224, 3)

    Args:
        image_bytes: Raw image bytes from file upload.

    Returns:
        Preprocessed numpy array of shape (1, 224, 224, 3).
    """
    # Open image from bytes buffer
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Image dimension check
    width, height = image.size
    if width < 128 or height < 128:
        raise ValueError(
            f"Image resolution too low ({width}x{height}px). "
            f"Minimum allowed size is 128x128 pixels."
        )

    # Resize to target size using high-quality Lanczos resampling
    image = image.resize(IMG_SIZE, Image.LANCZOS)

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def preprocess_image_for_gradcam(image_bytes: bytes) -> tuple:
    """
    Preprocess image and also return the original PIL Image for overlay.

    Returns:
        (img_array, pil_image) — preprocessed array + original PIL Image resized to 224x224.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Image dimension check
    width, height = image.size
    if width < 128 or height < 128:
        raise ValueError(
            f"Image resolution too low ({width}x{height}px). "
            f"Minimum allowed size is 128x128 pixels."
        )

    image_resized = image.resize(IMG_SIZE, Image.LANCZOS)

    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, image_resized
