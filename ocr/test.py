import cv2
from number_plate_reader import run_ocr  # ✅ This works ONLY IF test.py is in same folder

def test_sample_plate(image_path):
    print(f"🔍 Testing OCR on: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Failed to load image. Check path.")
        return

    plate_text = run_ocr(img)
    print(f"✅ Detected Plate Text: {plate_text}")

if __name__ == "__main__":
    # Provide a valid cropped plate image (JPG or PNG)
    test_sample_plate("sample_plates/sample1.jpg")
