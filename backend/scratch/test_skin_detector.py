"""
test_skin_detector.py
====================
Automated verification script for backend/utils/skin_detector.py.
Synthesizes test images (skin surface, grayscale, checkerboard, text/doc)
and validates that the hybrid skin detector accepts and rejects them correctly.
"""

import os
import sys
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Add backend directory to path to import skin_detector
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.skin_detector import detect_skin

def run_tests():
    print("🧪 Starting Skin Detector Automated Tests...")
    print("=" * 60)
    
    # ── Test Case 1: Synthetic Skin Image ────────────────────────────────────
    # Uniform light-brownish skin color (R=220, G=160, B=140)
    skin_img = Image.new("RGB", (224, 224), (220, 160, 140))
    buf = io.BytesIO()
    skin_img.save(buf, format="JPEG")
    skin_bytes = buf.getvalue()
    
    is_skin, reason, ratio = detect_skin(skin_bytes)
    print(f"CASE 1 (Synthetic Skin Image):")
    print(f"  - Expected: Accepted (is_skin=True, reason=OK)")
    print(f"  - Actual:   is_skin={is_skin}, reason={reason}, ratio={ratio * 100:.2f}%")
    assert is_skin is True, "FAIL: Skin image should be accepted!"
    assert reason == "OK", "FAIL: Reason should be OK!"
    print("  ✅ PASS\n")

    # ── Test Case 2: Grayscale Image ─────────────────────────────────────────
    # Uniform gray color (R=128, G=128, B=128)
    gray_img = Image.new("RGB", (224, 224), (128, 128, 128))
    buf = io.BytesIO()
    gray_img.save(buf, format="JPEG")
    gray_bytes = buf.getvalue()
    
    is_skin, reason, ratio = detect_skin(gray_bytes)
    print(f"CASE 2 (Grayscale Image):")
    print(f"  - Expected: Rejected (is_skin=False, reason=GRAYSCALE)")
    print(f"  - Actual:   is_skin={is_skin}, reason={reason}, ratio={ratio * 100:.2f}%")
    assert is_skin is False, "FAIL: Grayscale image should be rejected!"
    assert reason == "GRAYSCALE", "FAIL: Reason should be GRAYSCALE!"
    print("  ✅ PASS\n")

    # ── Test Case 3: Grayscale Gradient Image ──────────────────────────────────
    # Non-uniform gray scale to ensure mean diff logic is tested on complex gray
    gray_grad = np.tile(np.arange(224, dtype=np.uint8), (224, 1))
    gray_grad_rgb = cv2.merge([gray_grad, gray_grad, gray_grad])
    is_success, buffer = cv2.imencode(".jpg", gray_grad_rgb)
    assert is_success, "Failed to encode gray gradient"
    gray_grad_bytes = buffer.tobytes()
    
    is_skin, reason, ratio = detect_skin(gray_grad_bytes)
    print(f"CASE 3 (Grayscale Gradient Image):")
    print(f"  - Expected: Rejected (is_skin=False, reason=GRAYSCALE)")
    print(f"  - Actual:   is_skin={is_skin}, reason={reason}, ratio={ratio * 100:.2f}%")
    assert is_skin is False, "FAIL: Grayscale gradient image should be rejected!"
    assert reason == "GRAYSCALE", "FAIL: Reason should be GRAYSCALE!"
    print("  ✅ PASS\n")

    # ── Test Case 4: Checkerboard Non-Skin Image (QR-Like) ────────────────────
    # B&W checkerboard pattern of 16x16 squares
    checker = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(14):
        for j in range(14):
            if (i + j) % 2 == 0:
                checker[i*16:(i+1)*16, j*16:(j+1)*16] = 255 # White
            else:
                checker[i*16:(i+1)*16, j*16:(j+1)*16] = 0   # Black
                
    is_success, buffer = cv2.imencode(".jpg", checker)
    assert is_success, "Failed to encode checkerboard"
    checker_bytes = buffer.tobytes()
    
    is_skin, reason, ratio = detect_skin(checker_bytes)
    print(f"CASE 4 (B&W Checkerboard / QR-Like Pattern):")
    print(f"  - Expected: Rejected (is_skin=False, reason=GRAYSCALE or LOW_SKIN_RATIO)")
    print(f"  - Actual:   is_skin={is_skin}, reason={reason}, ratio={ratio * 100:.2f}%")
    assert is_skin is False, "FAIL: Non-skin pattern should be rejected!"
    assert reason in ["GRAYSCALE", "LOW_SKIN_RATIO"], f"FAIL: Reason should be GRAYSCALE or LOW_SKIN_RATIO, got {reason}!"
    print("  ✅ PASS\n")

    # ── Test Case 5: Document Screenshot ─────────────────────────────────────
    # White background with black lines simulating text blocks
    doc_img = Image.new("RGB", (224, 224), (250, 250, 250))
    draw = ImageDraw.Draw(doc_img)
    # Draw horizontal text lines
    for y in range(20, 200, 15):
        draw.line([(20, y), (200, y)], fill=(20, 20, 20), width=4)
        
    buf = io.BytesIO()
    doc_img.save(buf, format="JPEG")
    doc_bytes = buf.getvalue()
    
    is_skin, reason, ratio = detect_skin(doc_bytes)
    print(f"CASE 5 (Document Screenshot / Text Block):")
    print(f"  - Expected: Rejected (is_skin=False, reason=GRAYSCALE or LOW_SKIN_RATIO)")
    print(f"  - Actual:   is_skin={is_skin}, reason={reason}, ratio={ratio * 100:.2f}%")
    assert is_skin is False, "FAIL: Document image should be rejected!"
    assert reason in ["GRAYSCALE", "LOW_SKIN_RATIO"], f"FAIL: Reason should be GRAYSCALE or LOW_SKIN_RATIO, got {reason}!"
    print("  ✅ PASS\n")

    print("=" * 60)
    print("🎉 All 5 skin detector validation test cases PASSED successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
