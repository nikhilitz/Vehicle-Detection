import os
import re
import cv2
import numpy as np
import easyocr

easy_ocr = easyocr.Reader(['en'], gpu=False)

TO_NUMBER = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8',
    'A': '4' 
}

TO_ALPHA = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '6': 'G',
    '8': 'B',
    '4': 'A' 
}

def preprocess_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    resized = cv2.resize(filtered, (300, 80))
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper().strip())

def correct_segment(segment, expected_type):
    corrected = ''
    for char in segment:
        if expected_type == 'alpha' and char.isdigit():
            corrected += TO_ALPHA.get(char, char)
        elif expected_type == 'digit' and char.isalpha():
            corrected += TO_NUMBER.get(char, char)
        else:
            corrected += char
    return corrected

def post_process_indian_plate(text):
    text = clean_plate_text(text)
    text = text[-10:] if len(text) > 10 else text

    if len(text) != 10:
        return text

    state_code   = correct_segment(text[0:2], 'alpha')
    district     = correct_segment(text[2:4], 'digit')
    series       = correct_segment(text[4:6], 'alpha')
    unique_num   = correct_segment(text[6:],  'digit')

    return state_code + district + series + unique_num

def plate_valid(text):
    return (
        len(text) == 10 and
        text[0:2].isalpha() and
        text[2:4].isdigit() and
        text[4:6].isalpha() and
        text[6:].isdigit()
    )

def read_plate_text(plate_img, debug=False, debug_dir=None, frame_info=None, enable_correction=True):
    preprocessed = preprocess_plate(plate_img)

    if debug and debug_dir and frame_info:
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"{frame_info}_preprocessed.jpg")
        cv2.imwrite(debug_path, preprocessed)

    try:
        result = easy_ocr.readtext(preprocessed)
        if result:
            raw_text = ''.join([line[1] for line in result])
            cleaned = clean_plate_text(raw_text)

            if enable_correction:
                cleaned = post_process_indian_plate(cleaned)

            if plate_valid(cleaned):
                return cleaned

    except Exception as e:
        print(f"❌ EasyOCR Error: {e}")

    return "N/A"
