# pipeline/runner.py
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
        print("💡 Multiprocessing start method set to 'spawn' for GPU compatibility.")
except RuntimeError:
    print("💡 Multiprocessing start method already set.")


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from color_detection.color_detector import get_dominant_color
from ocr.number_plate_reader import read_plate_text

if torch.backends.mps.is_available():
    device = 'mps'
    print("🚀 Using Apple MPS (Metal Performance Shaders) for GPU acceleration.")
elif torch.cuda.is_available():
    device = 'cuda'
    print("🚀 Using NVIDIA CUDA for GPU acceleration.")
else:
    device = 'cpu'
    print("⚠️ Using CPU. Note: This module is much faster with a GPU (MPS/CUDA).")

# Define and create a directory for debugging outputs (moved to main block for clarity)
# This part stays in the main block because DEBUG_DIR needs to be created once
# by the parent process, and its path passed to children.
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)


MAX_FRAMES = 100 

def process_single_camera(cam_name, video_path, debug_dir_path, max_frames_limit, device_for_model):
    """
    Function to process a single camera stream.
    This function will be executed by each child process.
    """
    import os
    
    print(f" [Process {os.getpid()}] Loading YOLOS license plate model for {cam_name}...")
    try:
        processor = YolosImageProcessor.from_pretrained("nickmuchi/yolos-small-finetuned-license-plate-detection")
        model = YolosForObjectDetection.from_pretrained("nickmuchi/yolos-small-finetuned-license-plate-detection")
        model.to(device_for_model) 
        model.eval()
        print(f"[Process {os.getpid()}] YOLOS model loaded successfully (using {device_for_model.upper()}).")
    except Exception as e:
        print(f"\U0000274C [Process {os.getpid()}] Error loading YOLOS model: {e}")
        return 

    from storage.database import init_db, insert_detection
    init_db() 

    print(f"\n [Process {os.getpid()}] Processing {cam_name}: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f" [Process {os.getpid()}] Error: Could not open video file '{video_path}'. "
              "Please check the path, file integrity, and installed codecs.")
        return 

    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames_limit:
        ret, frame = cap.read()
        if not ret:
            print(f" [Process {os.getpid()}] Failed to read frame {frame_count} from {cam_name}. End of stream or error.")
            break 

        orig_frame = frame.copy() 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        inputs = processor(images=pil_image, return_tensors="pt").to(device_for_model)
        
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]]).to(device_for_model)
        results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

        for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            x1, y1, x2, y2 = map(int, box.tolist())
            
            h, w, _ = orig_frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x1 >= x2 or y1 >= y2:
                print(f" [Process {os.getpid()}] Warning: Invalid bounding box for detection {i} in frame {frame_count} of {cam_name}. Skipping.")
                continue

            padding = 5 
            x1_padded = max(0, x1 - padding)
            y1_padded = max(0, y1 - padding)
            x2_padded = min(w, x2 + padding)
            y2_padded = min(h, y2 + padding)

            plate_crop = orig_frame[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if plate_crop.size == 0:
                print(f" [Process {os.getpid()}] Warning: Empty plate crop for detection {i} in frame {frame_count}. Skipping.")
                continue

            color = "unknown"
            plate_text = "N/A" 

            try:
                color = get_dominant_color(plate_crop)
                
                raw_plate_text = read_plate_text(
                    plate_crop, 
                    debug=True, 
                    debug_dir=debug_dir_path, 
                    frame_info=f"{cam_name}_frame{frame_count}_plate{i}"
                )
                
                cleaned_plate_text = ''.join(filter(str.isalnum, raw_plate_text)).upper()

                if not cleaned_plate_text or cleaned_plate_text == "UNKNOWN":
                    plate_text = "N/A"
                else:
                    plate_text = cleaned_plate_text

            except Exception as e:
                print(f"\U0000274C [Process {os.getpid()}] Error processing plate {i} in frame {frame_count} from {cam_name}: name 'os' is not defined" if "name 'os' is not defined" in str(e) else f"\U0000274C [Process {os.getpid()}] Error processing plate {i} in frame {frame_count} from {cam_name}: {e}")
                cv2.imwrite(os.path.join(debug_dir_path, f"{cam_name}_frame{frame_count}_plate{i}_error.jpg"), plate_crop)
                plate_text = "N/A" 

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{cam_name} - Frame {frame_count} - Plate {i}] Color: {color}, Plate: {plate_text}")

            if plate_text != "N/A": 
                insert_detection(
                    plate=plate_text,
                    color=color.lower(),
                    timestamp=timestamp,
                    camera=cam_name
                )
            else:
                print(f"\U000026A0 [Process {os.getpid()}] Skipping database insertion for plate {i} (Frame {frame_count}) from {cam_name} as plate text is '{plate_text}'.")

            cv2.imwrite(os.path.join(debug_dir_path, f"{cam_name}_frame{frame_count}_plate{i}_cropped.jpg"), plate_crop)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_count += 1
        
        cv2.imshow(f"Live - {cam_name} (PID: {os.getpid()})", frame) 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"\U0001F6A6 [Process {os.getpid()}] 'q' pressed. Stopping playback.")
            break

    cap.release()
    cv2.destroyAllWindows() 
    print(f"\U00002705 [Process {os.getpid()}] Finished processing {cam_name}.")

def run_on_all_cameras(cameras):
    from storage.database import init_db
    init_db() 

    print("\U0001F680 Starting camera loop using multiprocessing...")

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

    print("\n\U0001F69C All camera processing complete. Exiting.")

if __name__ == "__main__":
    multiprocessing.freeze_support() 

    # Define DEBUG_DIR here once in the parent process
    DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "debug")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f"Debug output directory: {DEBUG_DIR}") 

    cameras = {
        "cam1": "sample_videos/cam1.mp4", 
        "cam2": "sample_videos/cam2.mp4"  
    }
    
    all_videos_found = True
    for cam_name, video_path in cameras.items():
        if not os.path.exists(video_path):
            print(f"\U0000274C Error: Video file for '{cam_name}' not found at '{video_path}'.")
            all_videos_found = False
    
    if not all_videos_found:
        print("Please ensure all specified video files exist in the 'sample_videos/' directory.")
        sys.exit(1)

    run_on_all_cameras(cameras)
    print("\n\U0001F69C Main process finished. All camera processing complete.")