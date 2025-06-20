# vehicle_monitoring/detection/detect_vehicles.py

import torch
from ultralytics import YOLO
import cv2



# Import custom modules from ultralytics' core model architecture
from ultralytics.nn.tasks import DetectionModel, BaseModel
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF, DFL 
from ultralytics.nn.modules.head import Detect

# Import standard PyTorch modules that are commonly part of the YOLOv8 model's pickle
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU, LeakyReLU
from torch.nn.modules.container import Sequential, ModuleList
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.dropout import Dropout # Dropout might be in some models

# Registering all these classes as safe for unpickling 
torch.serialization.add_safe_globals([
    DetectionModel,
    BaseModel,
    Conv,
    Concat,
    C2f,
    Bottleneck,
    SPPF,
    DFL, 
    Detect,
    Conv2d,
    BatchNorm2d,
    SiLU,
    LeakyReLU,
    Sequential,
    ModuleList,
    MaxPool2d,
    Upsample,
    Dropout
])

# 2. Load the YOLOv8 model
# This line should now work after the safe global registrations
model = YOLO('yolov8x.pt')  # 'n' = nano (smallest and fastest version)

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
    # model(frame) performs inference. [0] gets the first (and usually only) result object.
    results = model(frame)[0]
    detections = []

    for r in results.boxes:
        cls_id = int(r.cls[0])  # Get the class ID (e.g., 2 for car)
        label = model.names[cls_id]  # Get the class name from the model's names dictionary
        if label in vehicle_classes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])  # Get bounding box coordinates as integers
            detections.append((label, (x1, y1, x2, y2)))

    return detections