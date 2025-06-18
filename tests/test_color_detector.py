# tests/test_color_detector.py
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import cv2
from ocr.number_plate_reader import read_plate_text

# Load the test image
img_path = "sample_images/plate1.png"  # Or .jpg
image = cv2.imread(img_path)

if image is None:
    print("❌ Failed to load image. Check the file path.")
else:
    text = read_plate_text(image, debug=True, use_preprocessing=False)
    print("✅ Detected Number Plate Text:", text)
