import cv2
import numpy as np

GRAYSCALE_RANGES = {
    "black": ([0, 0, 0], [180, 255, 50]),
    "white": ([0, 0, 200], [180, 30, 255]),
    "gray":  ([0, 0, 51], [180, 30, 199]),
}

# HSV color ranges (approximate)
HSV_COLOR_RANGES = {
    "red":     [([0, 100, 100], [10, 255, 255]), ([170, 100, 100], [179, 255, 255])],
    "orange":  [([11, 100, 100], [25, 255, 255])],
    "yellow":  [([26, 100, 100], [35, 255, 255])],
    "green":   [([36, 50, 50], [89, 255, 255])],
    "cyan":    [([90, 50, 50], [99, 255, 255])],
    "blue":    [([100, 100, 100], [130, 255, 255])],
    "purple":  [([131, 50, 50], [145, 255, 255])],
    "magenta": [([146, 50, 50], [169, 255, 255])],
    "brown":   [([10, 50, 50], [25, 180, 120])],
}


def get_dominant_color(image):
    if image is None or image.size == 0:
        return "unknown"

    # Resize image for performance
    img_resized = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    hsv_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    total_pixels = hsv_image.shape[0] * hsv_image.shape[1]

    # Combine grayscale + color ranges
    all_color_ranges = {}
    all_color_ranges.update(GRAYSCALE_RANGES)
    all_color_ranges.update(HSV_COLOR_RANGES)

    color_pixel_counts = {}

    for color_name, ranges in all_color_ranges.items():
        if isinstance(ranges[0][0], int):  # grayscale is a single [lower, upper]
            ranges = [ranges]  # wrap in a list

        mask_total = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            mask_total = cv2.bitwise_or(mask_total, mask)

        pixel_count = np.sum(mask_total > 0)
        color_pixel_counts[color_name] = pixel_count

    # Determine dominant color
    dominant_color = max(color_pixel_counts, key=color_pixel_counts.get)
    percentage = color_pixel_counts[dominant_color] / total_pixels

    if percentage > 0.15:  # only return if confidently dominant
        return dominant_color
    else:
        return "unknown"
