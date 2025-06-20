# color_detection/color_detector.py

import cv2
import numpy as np

# Define color ranges in HSV.
# Hue (H): 0-179 (OpenCV's range)
# Saturation (S): 0-255
# Value (V): 0-255

# Each color is defined by a list of [lower_bound, upper_bound] tuples for HSV.
# Some colors (like Red) wrap around the hue spectrum (0 and 179), so they need two ranges.
HSV_COLOR_RANGES = {
    # Lowered minimum Saturation (S) and Value (V) to 50 for more tolerance towards darker/less vibrant colors.
    # Red: split into two ranges because hue wraps around 0/179
    "red": [([0, 50, 50], [10, 255, 255]), ([170, 50, 50], [179, 255, 255])], 
    "orange": [([11, 50, 50], [25, 255, 255])],
    "yellow": [([26, 50, 50], [35, 255, 255])],
    "green": [([36, 50, 50], [85, 255, 255])], 
    "blue": [([100, 50, 50], [130, 255, 255])],
    "purple": [([131, 50, 50], [169, 255, 255])],
    # Brown is often a darker, less saturated orange/red/yellow.
    # Its S and V ranges are kept relatively lower.
    "brown": [([10, 40, 20], [25, 180, 150])], # Adjusted upper S/V for brown slightly
}

def get_dominant_color(image):
    """
    Extracts the dominant perceptual color of an image using HSV color ranges.
    Prioritizes white, black, gray, then checks other colors based on pixel counts.
    
    Args:
        image: cropped vehicle image (BGR format from OpenCV).

    Returns:
        str: closest color label (e.g., "red", "blue", "white", "black", "gray", "unknown").
    """
    if image is None or image.size == 0:
        return "unknown"

    # Reduce image size for faster processing and to average out noise
    img_resized = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
    
    # Convert BGR to HSV for color range detection
    hsv_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    total_pixels = img_resized.shape[0] * img_resized.shape[1]

    # --- Step 1: Detect White, Black, Gray based on Value (Brightness) and Saturation ---
    # These ranges are made more specific to reduce overlap with dark/desaturated colors.

    # Black: Very low Value (brightness) and very low Saturation
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([179, 40, 40]) # Max Saturation for black: 40, Max Value for black: 40
    mask_black = cv2.inRange(hsv_image, lower_black, upper_black)
    black_pixels = np.sum(mask_black > 0)

    # White: Very high Value (brightness) and very low Saturation
    lower_white = np.array([0, 0, 210]) # Min Value for white: 210
    upper_white = np.array([179, 40, 255]) # Max Saturation for white: 40
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
    white_pixels = np.sum(mask_white > 0)

    # Gray: Low Saturation (regardless of Hue) and mid-range Value
    # The Value range for gray is now strictly between black and white.
    lower_gray = np.array([0, 0, 41])  # Min Value for gray: just above max black value
    upper_gray = np.array([179, 25, 209]) # Max Saturation for gray: 25, Max Value for gray: just below min white value
    mask_gray = cv2.inRange(hsv_image, lower_gray, upper_gray)
    gray_pixels = np.sum(mask_gray > 0)

    # Prioritize white, black, gray if they represent a significant portion of the image
    if white_pixels / total_pixels > 0.4: 
        return "white"
    if black_pixels / total_pixels > 0.4: 
        return "black"
    if gray_pixels / total_pixels > 0.4: 
        return "gray"

    # --- Step 2: Detect other colors based on HSV ranges ---
    color_pixel_counts = {}
    for color_name, ranges in HSV_COLOR_RANGES.items():
        count = 0
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            count += np.sum(mask > 0)
        color_pixel_counts[color_name] = count

    # Find the color with the most pixels from the defined HSV ranges
    if color_pixel_counts:
        dominant_color_name = max(color_pixel_counts, key=color_pixel_counts.get)
        
        # If the dominant color's pixels exceed a certain threshold, return it.
        if color_pixel_counts[dominant_color_name] / total_pixels > 0.2: 
            return dominant_color_name

    # Fallback if no dominant color is found with sufficient confidence
    return "unknown"