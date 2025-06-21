import sys
import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from datetime import datetime
from transformers import YolosImageProcessor, YolosForObjectDetection
import multiprocessing

try:
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn', force=True)
        print("\U0001F4A1 Multiprocessing start method set to 'spawn' for GPU compatibility.")
except RuntimeError:
    print("\U0001F4A1 Multiprocessing start method already set.")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from color_detection.color_detector import get_dominant_color
from ocr.number_plate_reader import read_plate_text

if torch.backends.mps.is_available():
    device = 'mps'
    print("\U0001F680 Using Apple MPS (Metal Performance Shaders) for GPU acceleration.")
elif torch.cuda.is_available():
    device = 'cuda'
    print("\U0001F680 Using NVIDIA CUDA for GPU acceleration.")
else:
    device = 'cpu'
    print("\u26A0\uFE0F Using CPU. This will be slower than GPU (MPS/CUDA).")

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

MAX_FRAMES = 100

def process_single_camera(cam_name, video_path, debug_dir_path, max_frames_limit, device_for_model):
    print(f" [PID {os.getpid()}] Loading YOLOS model for {cam_name}...")
    try:
        processor = YolosImageProcessor.from_pretrained("nickmuchi/yolos-small-finetuned-license-plate-detection")
        model = YolosForObjectDetection.from_pretrained("nickmuchi/yolos-small-finetuned-license-plate-detection")
        model.to(device_for_model)
        model.eval()
        print(f"[PID {os.getpid()}] YOLOS model loaded using {device_for_model.upper()}.")
    except Exception as e:
        print(f"\u274C [PID {os.getpid()}] Error loading YOLOS model: {e}")
        return

    from storage.database import init_db, insert_detection
    init_db()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\u274C Could not open video file: {video_path}")
        return

    frame_count = 0

    while cap.isOpened() and frame_count < max_frames_limit:
        ret, frame = cap.read()
        if not ret:
            break

        orig_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        inputs = processor(images=pil_image, return_tensors="pt").to(device_for_model)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]]).to(device_for_model)
        results = processor.post_process_object_detection(outputs, threshold=0.4, target_sizes=target_sizes)[0]

        print(f"\U0001F50D Frame {frame_count}: Detected {len(results['boxes'])} boxes")

        for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            x1, y1, x2, y2 = map(int, box.tolist())
            h, w, _ = orig_frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if x1 >= x2 or y1 >= y2:
                continue

            padding = 5
            x1_p, y1_p = max(0, x1 - padding), max(0, y1 - padding)
            x2_p, y2_p = min(w, x2 + padding), min(h, y2 + padding)
            plate_crop = orig_frame[y1_p:y2_p, x1_p:x2_p]
            if plate_crop.size == 0:
                continue

            try:
                color = get_dominant_color(plate_crop)
                raw_plate = read_plate_text(
                    plate_crop,
                    debug=True,
                    debug_dir=debug_dir_path,
                    frame_info=f"{cam_name}_frame{frame_count}_plate{i}"
                )
                plate_text = ''.join(filter(str.isalnum, raw_plate)).upper()
                plate_text = plate_text if plate_text and plate_text != "UNKNOWN" else "N/A"
            except Exception as e:
                print(f"\u274C Error processing plate: {e}")
                plate_text = "N/A"

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{cam_name} - Frame {frame_count}] Color: {color}, Plate: {plate_text}")

            if plate_text != "N/A":
                # ✅ Save full vehicle frame (not just the plate)
                vehicle_img_path = os.path.join(
                    debug_dir_path,
                    f"{cam_name}_frame{frame_count}_full_vehicle_{i}.jpg"
                )
                cv2.imwrite(vehicle_img_path, orig_frame)

                insert_detection(
                    plate=plate_text,
                    color=color.lower(),
                    timestamp=timestamp,
                    camera=cam_name,
                    image_path=vehicle_img_path
                )

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_count += 1
        cv2.imshow(f"Live - {cam_name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\u2705 [PID {os.getpid()}] Finished processing {cam_name}.")

def run_on_all_cameras(cameras):
    from storage.database import init_db
    init_db()
    print("\U0001F680 Starting multiprocessing for all cameras...")

    processes = []
    for cam_name, video_path in cameras.items():
        p = multiprocessing.Process(
            target=process_single_camera,
            args=(cam_name, video_path, DEBUG_DIR, MAX_FRAMES, device)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("\n\U0001F69C All processing complete.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    cameras = {
        "cam3": "sample_videos/cam3.mov",
        "cam4": "sample_videos/cam4.mov"
    }
    for cam, path in cameras.items():
        if not os.path.exists(path):
            print(f"\u274C File for {cam} not found: {path}")
            sys.exit(1)

    run_on_all_cameras(cameras)
