# vehicle_monitoring/ocr/number_plate_reader.py

import easyocr
import cv2
import numpy as np

# Initialize EasyOCR Reader (only once)
reader = easyocr.Reader(['en'], gpu=False)  # Use GPU if available

def preprocess_plate(image):
    """
    Enhance number plate readability for OCR by converting to grayscale
    and applying smoothing + adaptive threshold.
    """
    # converting to grayscale reducing complexity and helps ocr (OCR works better with contrast-based black & white images.)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

     #smooths the image to remove noise without blurring edges
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)


    """
    Converts a grayscale image into black-and-white (binary) image, with text clearly separated from background.
    Why it is used:
    Normal thresholding uses one fixed threshold value for the whole image.
    But license plates are often under uneven lighting — shadow on one side, sun on the other.
    Adaptive thresholding computes different thresholds for different parts of the image → handles lighting variation.
    """
    thresh = cv2.adaptiveThreshold(
        filtered, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    """
    return thresh
    Returns the processed black-and-white image where text is:
    Clearly visible
    Well-separated from background
    Ready for EasyOCR to recognize characters
    """
    return thresh

def read_plate_text(image, debug=False, use_preprocessing=True):
    """
    Extract number plate text from image using EasyOCR.

    Parameters:
        image (np.ndarray): Cropped car image
        debug (bool): Show intermediate steps
        use_preprocessing (bool): Whether to use thresholding or raw image

    Returns:
        str: Plate number text or 'unknown'
    """

    # 1. Preprocess if enabled
    if use_preprocessing:
        processed = preprocess_plate(image)
    else:
        processed = image  # raw input image (color)

    if debug:
        cv2.imshow("Processed Plate", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 2. Run EasyOCR
    results = reader.readtext(processed)

    print("\n🔍 OCR Raw Output:")
    for (bbox, text, confidence) in results:
        print(f"Text: {text}, Confidence: {confidence}")

        cleaned = ''.join(char for char in text if char.isalnum())
        print(f"🧼 Cleaned: {cleaned}")

        if confidence > 0.4 and 5 <= len(cleaned) <= 12:
            return cleaned.upper()

    return "unknown"
