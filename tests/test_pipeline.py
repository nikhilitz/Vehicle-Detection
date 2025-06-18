# tests/test_pipeline_cam1.py
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
from detection.detect_vehicles import detect_vehicles
from color_detection.color_detector import get_dominant_color
from ocr.number_plate_reader import read_plate_text

# Load video
cap = cv2.VideoCapture('sample_videos/cam1.mp4')

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 != 0:
        continue  # Skip every 9 frames for speed

    # Step 1: Detect vehicles
    detections = detect_vehicles(frame)

    for idx, (label, (x1, y1, x2, y2)) in enumerate(detections):
        cropped_vehicle = frame[y1:y2, x1:x2]

        # Step 2: Color detection
        color = get_dominant_color(cropped_vehicle)

        # Step 3: Number plate reading
        plate_text = read_plate_text(cropped_vehicle, debug=False, use_preprocessing=True)

        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label}, {color}, {plate_text}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # Show output
    cv2.imshow("Pipeline Output (Cam1)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
