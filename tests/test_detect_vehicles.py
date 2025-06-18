# Test file: tests/test_detect_vehicles.py
import cv2
import sys
import os

# ✅ Add the root path of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from detection.detect_vehicles import detect_vehicles

cap = cv2.VideoCapture('sample_videos/sample1.mp4')  # simulate CCTV

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_vehicles(frame)

    for label, (x1, y1, x2, y2) in detections:
        # Draw box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
