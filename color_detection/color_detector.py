# vehicle_monitoring/color_detection/color_detector.py

import cv2
import numpy as np

# ✅ Define HSV color ranges for common car colors
# Each color is mapped to one or more (lower, upper) HSV threshold tuples
# These values are empirically tuned for common lighting conditions

color_ranges = {
    "black":  [([0, 0, 0], [180, 255, 46])],
    "white":  [([0, 0, 200], [180, 30, 255])],
    "gray":   [([0, 0, 40], [180, 20, 200])],

    #  Red is split into two ranges (HSV hue is circular: 0-10 and 160-180)
    "red":    [([0, 70, 50], [10, 255, 255]),
               ([160, 70, 50], [180, 255, 255])],

    "orange": [([11, 100, 100], [25, 255, 255])],
    "yellow": [([26, 100, 100], [35, 255, 255])],
    "green":  [([36, 100, 100], [89, 255, 255])],
    "blue":   [([90, 100, 100], [130, 255, 255])],
    "brown":  [([10, 100, 20], [20, 200, 200])]  # Optional for dark or muddy colors
}


def get_dominant_color(image):
    """
    Detect the dominant color of the input car image using HSV thresholds.

    Parameters:
        image (numpy.ndarray): Cropped image of the vehicle in BGR format.

    Returns:
        str: Detected color name (e.g., "black", "white", "red", etc.)
    """

    #  Step 1: Convert image from BGR (OpenCV default) to HSV
    # HSV makes it easier to define color ranges because it separates hue from intensity
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 2: Initialize values to track the most dominant color
    # max_pixels: highest number of matching pixels for any color
    # dominant_color: name of the most dominant color
    max_pixels = 0
    dominant_color = "unknown"

    #  Step 3: Loop through each color name and its HSV range list
    for color_name, ranges in color_ranges.items():
        total_count = 0  # total pixels matched for this color

        for lower, upper in ranges:
            # Convert Python lists to NumPy arrays for OpenCV
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)

            #  Step 4: Create a binary mask
            # For every pixel in hsv_img:
            # If pixel ∈ range → 255 (white), else → 0 (black)
            mask = cv2.inRange(hsv_img, lower_np, upper_np)

            #  Step 5: Count number of white pixels = matched pixels
            count = cv2.countNonZero(mask)
            total_count += count

        #  Step 6: If this color has more matching pixels than previous best
        # → update dominant color
        if total_count > max_pixels:
            max_pixels = total_count
            dominant_color = color_name

    return dominant_color
