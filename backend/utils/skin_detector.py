"""
skin_detector.py
================
Utility module containing a robust hybrid skin detection algorithm.
Validates if an uploaded image contains a valid human skin surface/lesion.
Specifically designed to reject QR codes, documents, and other non-skin images.
Uses YCbCr and HSV color spaces for ethnic skin tone inclusivity.
"""

import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

def detect_skin(image_bytes: bytes) -> tuple:
    """
    Analyzes image bytes to detect if it is a proper clinical skin scan.
    
    Checks performed:
      1. QR Code Detection using OpenCV's QRCodeDetector.
      2. Grayscale/Monochrome check (clinical scans must be in color).
      3. YCbCr + HSV Color Clustering (independent of luminance to ensure 
         Fitzpatrick skin tone inclusivity).
         - YCbCr: Cb in [77, 127], Cr in [133, 173]
         - HSV: H in [0, 25] or [150, 180], S in [20, 170], V in [40, 255]
      4. Skin Pixel Ratio validation (requires >= 25% skin coverage).

    Args:
        image_bytes: Raw upload image bytes.

    Returns:
        (is_skin: bool, reason: str, skin_ratio: float)
          - is_skin: True if the image is a valid skin scan, False otherwise.
          - reason: A descriptive code indicating the detection result.
          - skin_ratio: Percentage of pixels identified as skin (0.0 to 1.0).
    """
    try:
        # Load image with PIL and convert to RGB
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = pil_img.size
        
        # Convert PIL Image to OpenCV BGR array
        open_cv_image = np.array(pil_img)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        # Convert to Grayscale for structural checks
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # ── 1. QR Code Detection ──────────────────────────────────────────────
        qr_detector = cv2.QRCodeDetector()
        retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(img_gray)
        if retval:
            logger.info("❌ Skin Detector: QR Code detected in uploaded image.")
            return False, "QR_CODE", 0.0

        # ── 2. Grayscale / Monochrome Validation ──────────────────────────────
        # Standard clinical dermoscopic photos contain rich RGB color profiles.
        # Grayscale images lack the essential color characteristics required for
        # classification (such as the 'C' in the ABCDE guidelines for color variation).
        # We calculate the mean absolute difference between channels.
        b_channel, g_channel, r_channel = cv2.split(img_bgr)
        diff_rg = np.mean(np.abs(r_channel.astype(float) - g_channel.astype(float)))
        diff_gb = np.mean(np.abs(g_channel.astype(float) - b_channel.astype(float)))
        
        if diff_rg < 4.0 and diff_gb < 4.0:
            logger.info(f"❌ Skin Detector: Image appears grayscale (mean diff RG={diff_rg:.2f}, GB={diff_gb:.2f}).")
            return False, "GRAYSCALE", 0.0

        # ── 3. Luminance-Independent Skin Color Segmentation ──────────────────
        # We use a hybrid of YCbCr and HSV color spaces for ethnic inclusivity.
        #
        # YCbCr Color Space:
        # Isolates luminance (Y) from chrominance (Cb, Cr). Human skin clusters
        # tightly in chrominance regardless of skin tone (Fitzpatrick scale).
        # Bounds: 77 <= Cb <= 127, 133 <= Cr <= 173
        ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycbcr[:, :, 1]
        cb = ycbcr[:, :, 2]
        ycbcr_mask = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)

        # HSV Color Space:
        # Captures hue, saturation, and value. Reduces impact of lighting variance.
        # Bounds: Hue in red/orange range [0, 25] or pink/reddish [150, 180],
        # Saturation in organic range [20, 170] (filters grey/neon),
        # Value in active visible range [40, 255] (filters pitch black).
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        hsv_mask = (((h <= 25) | (h >= 150)) & 
                    (s >= 20) & (s <= 170) & 
                    (v >= 40))

        # Intersect YCbCr and HSV masks for high precision
        skin_mask = ycbcr_mask & hsv_mask
        
        # Calculate coverage ratio
        total_pixels = width * height
        skin_pixels = np.sum(skin_mask)
        skin_ratio = float(skin_pixels) / total_pixels
        
        logger.info(f"ℹ️ Skin Detector: Skin pixel ratio is {skin_ratio * 100:.2f}%.")
        
        # ── 4. Skin Coverage Ratio Threshold ──────────────────────────────────
        # A valid clinical scan is focused closely on the skin surface.
        # Therefore, at least 25% of the frame must match valid skin chrominance.
        if skin_ratio < 0.25:
            logger.info("❌ Skin Detector: Skin pixel ratio below 25% threshold.")
            return False, "LOW_SKIN_RATIO", skin_ratio
            
        logger.info("✅ Skin Detector: Image validated successfully as a skin surface.")
        return True, "OK", skin_ratio

    except Exception as e:
        logger.error(f"❌ Skin Detector error during execution: {e}")
        return False, f"ERROR: {str(e)}", 0.0
