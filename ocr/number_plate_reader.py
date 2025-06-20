import os
import re
import cv2
import numpy as np
import easyocr

easy_ocr = easyocr.Reader(['en'], gpu=False)

CORRECTION_MAP = {
    'O': '0',
    'I': '1',
    'S': '5',
    'G': '6',
    'B': '8'
}

def preprocess_plate(image):
    """
    Preprocess license plate image for OCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    resized = cv2.resize(filtered, (300, 80))  # Resize to plate aspect
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)  # EasyOCR expects 3-channel

def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper().strip())

def correct_text(text, enable_correction=False):
    if not enable_correction:
        return text
    return ''.join(CORRECTION_MAP.get(c, c) for c in text)

def plate_valid(text):
    """
    Acceptable length & must contain digits (to reject false reads).
    """
    return 6 <= len(text) <= 12 and any(char.isdigit() for char in text)

# --- Main OCR function ---
def read_plate_text(plate_img, debug=False, debug_dir=None, frame_info=None, enable_correction=False):
    """
    Run OCR on the number plate image using EasyOCR with optional correction.
    """
    preprocessed = preprocess_plate(plate_img)

    # Optional debug image saving
    if debug and debug_dir and frame_info:
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"{frame_info}_preprocessed.jpg")
        cv2.imwrite(debug_path, preprocessed)

    try:
        result = easy_ocr.readtext(preprocessed)
        if result:
            # Join all recognized text chunks
            raw_text = ''.join([line[1] for line in result])
            cleaned = clean_plate_text(raw_text)
            corrected = correct_text(cleaned, enable_correction)

            if plate_valid(corrected):
                return corrected
    except Exception as e:
        print(f"❌ EasyOCR Error: {e}")

    # Save failure frame for debugging
    # if debug and debug_dir and frame_info:
    #     error_path = os.path.join(debug_dir, f"{frame_info}_ocr_failed.jpg")
    #     cv2.imwrite(error_path, plate_img)

    return "N/A"
