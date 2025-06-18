# vehicle_monitoring/main.py

import cv2
import threading
import os

from detection.detect_vehicles import detect_vehicles
from color_detection.color_detector import get_dominant_color
from ocr.number_plate_reader import read_plate_text
from chat.query_state import get_latest_query
from matcher.matcher import is_match  # New matcher logic

# Simulated camera sources (you can add more)
camera_feeds = {
    'cam1': 'sample_videos/cam1.mp4',
    # 'cam2': 'sample_videos/cam2.mp4',
    # 'cam3': 'sample_videos/cam3.mp4'
}

def process_camera(camera_id, video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 0: Read current chat command (e.g. action=track, color=black, plate=HR26AB1234)
        current_query = get_latest_query()

        # Step 1: Vehicle detection
        detections = detect_vehicles(frame)

        for label, (x1, y1, x2, y2) in detections:
            cropped_vehicle = frame[y1:y2, x1:x2]

            # Step 2: Color Detection
            color = get_dominant_color(cropped_vehicle)

            # Step 3: OCR
            plate_text = read_plate_text(cropped_vehicle)

            # Step 4: Match vehicle with chat query
            match = is_match(current_query, color, plate_text)

            if match:
                # Draw if it matches user request
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                info = f"{color} {label} - {plate_text}"
                cv2.putText(frame, info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 0), 2)

        # Display the result
        cv2.imshow(f"Live - {camera_id}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 🔁 One thread per camera
threads = []

for cam_id, path in camera_feeds.items():
    if not os.path.exists(path):
        print(f"❌ Video not found for {cam_id} → {path}")
        continue

    t = threading.Thread(target=process_camera, args=(cam_id, path))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
