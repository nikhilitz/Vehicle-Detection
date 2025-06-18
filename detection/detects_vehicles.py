# vehicle_monitoring/detection/detect_vehicles.py

# 1. Import necessary libraries
from ultralytics import YOLO  # for using YOLOv8 pre-trained model
import cv2  # for image processing

# 2. Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # 'n' = nano (smallest and fastest version)
# Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
# We use 'n' because: fast for development; we can switch to 'm' or 'l' later.

# 3. Define which object classes are vehicles
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']

def detect_vehicles(frame):
    """
    Detect vehicles in the input video frame using YOLOv8.

    Parameters:
        frame (numpy array): input video frame (from OpenCV)

    Returns:
        List of tuples: [(label, (x1, y1, x2, y2)), ...]
    """
    results = model(frame)[0]  # returns list of detections
    detections = []

    for r in results.boxes:
        cls_id = int(r.cls[0])  # class index (e.g., 2 = car)
        label = model.names[cls_id]  # class name from YOLO model
        if label in vehicle_classes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])  # bounding box
            detections.append((label, (x1, y1, x2, y2)))

    return detections
