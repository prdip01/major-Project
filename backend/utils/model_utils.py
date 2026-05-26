"""
model_utils.py
==============
Model loading and inference. Enforces strict actual CNN model execution,
completely terminating simulated mock random predictions (demo mode fallbacks).
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Path to the saved model (absolute to avoid issues)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "skin_cancer_model.h5")

# Environment variables for macOS stability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging clutter
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Prevent OpenMP runtime conflict

# Cached model reference (loaded once)
_model = None

# Class labels in the same order as the model's output neurons
CLASS_LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def load_model():
    """
    Load the EfficientNetB0 model from disk if available.
    Returns the model object or None if not found.
    """
    if os.environ.get("DEMO_ONLY") == "true":
        logger.warning("DEMO_ONLY environment variable set. Enforcing offline status.")
        return None

    # --- RENDER CLOUD HOSTING MEMORY LIMIT GUARD ---
    # Render's free tier has a strict 512 MB RAM cap.
    # Importing TensorFlow and loading the 134 MB CNN model consumes 600MB+ RAM.
    # To prevent Out-Of-Memory (OOM) crashes on Render, we keep the guard but don't fake predictions.
    if os.environ.get("RENDER") == "true":
        logger.warning(
            "🌐 ACTIVE CLOUD DEPLOYMENT DETECTED (Render.com)\n"
            "   Cloud RAM exceeds 512MB limit. Server runs offline mode."
        )
        return None

    global _model
    if _model is not None:
        return _model

    # --- INCOMPATIBILITY GUARD for macOS ARM + Python 3.13 ---
    import sys
    import platform
    is_macos_arm = sys.platform == "darwin" and platform.machine() == "arm64"
    is_py313 = sys.version_info.major == 3 and sys.version_info.minor == 13

    if is_macos_arm and is_py313:
        logger.error(
            "⚠️  INCOMPATIBLE ENVIRONMENT DETECTED (macOS ARM + Python 3.13)\n"
            "   TensorFlow crashes on this specific combination. Model cannot load.\n"
            "   💡 FIX: Please install Python 3.11 or 3.12 for full model support."
        )
        return None

    model_path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(model_path):
        logger.warning(f"❌ Model file not found at {model_path}.")
        return None

    try:
        import tensorflow as tf
        logger.info(f"🔄 Loading model from {model_path} (on-the-fly) …")
        
        # Use compile=False to avoid initializer crashes on some macOS environments
        _model = tf.keras.models.load_model(model_path, compile=False)
        
        logger.info("✅ Model loaded successfully.")
        return _model
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None


def predict(img_array: np.ndarray) -> dict:
    """
    Run inference on a preprocessed image array.
    Strictly performs actual neural predictions. Raises exception if model not active.

    Args:
        img_array: Numpy array of shape (1, 224, 224, 3), normalized to [0, 1].

    Returns:
        dict with keys:
          - predicted_label (str): e.g. "mel"
          - predicted_index (int): index into CLASS_LABELS
          - confidence (float): 0–1 value of top prediction
          - probabilities (dict): {label: probability} for all 7 classes
    """
    model = load_model()

    if model is None:
        raise RuntimeError(
            "EfficientNet-B0 model weights not loaded. "
            f"Please verify model file exists at: {os.path.abspath(MODEL_PATH)}"
        )

    # --- REAL CNN INFERENCE ---
    logger.info("⚡ Executing real CNN inference pass...")
    raw = model.predict(img_array, verbose=0)[0]  # shape: (7,)
    probs = dict(zip(CLASS_LABELS, raw.tolist()))

    # --- CLINICAL DECISION BOUNDARY CALIBRATION (Cost-Sensitive Learning) ---
    # In medical screening, a false negative (missing a cancer) is extremely critical.
    # We apply a clinical calibration factor of 1.25x to the raw probabilities of
    # malignant classes (mel, bcc, akiec) during the argmax step. 
    # Note: We continue to report the raw, unaltered model softmax probabilities 
    # to the user dashboard to ensure full neural explanation transparency.
    malignant_classes = {"mel", "bcc", "akiec"}
    calibrated_probs = {}
    for k, v in probs.items():
        if k in malignant_classes:
            calibrated_probs[k] = v * 1.25
        else:
            calibrated_probs[k] = v

    # Identify top prediction based on calibrated clinical decision boundaries
    predicted_label = max(calibrated_probs, key=calibrated_probs.get)
    predicted_index = CLASS_LABELS.index(predicted_label)
    confidence = probs[predicted_label]

    return {
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "probabilities": probs,
        "demo_mode": False,
    }
