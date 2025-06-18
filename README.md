# 🚘 Vehicle Monitoring with AI Chat

A real-time AI surveillance system that detects vehicles in live video streams, reads their number plates, identifies colors, and responds to natural chat commands like:

- "Show all black cars in the parking lot"
- "Track the white car with number plate HR26AB1234"

---

## 📦 Features Implemented

| Feature                         | Status    |
|---------------------------------|-----------|
| Vehicle detection using YOLOv8  | ✅ Done    |
| Vehicle color classification    | ✅ Done    |
| Number plate recognition (OCR)  | ✅ Done    |
| Human-style chat command parser | ✅ Done    |
| Live video processing (multi-cam)| ✅ Done   |
| Query matching system           | 🔄 Next    |
| Command input (UI/CLI/Chatbot)  | 🔄 Next    |

---

## 🔍 Use Case

Imagine monitoring a parking lot. A human supervisor sends a message like:

> “Track the red car with number plate HR26AB1234”

The system:
1. Parses the chat
2. Tracks matching vehicles across all camera feeds
3. Shows them live on screen

---

## 🧠 Tech Stack

- **YOLOv8**: Vehicle detection (cars, trucks, buses, bikes)
- **OpenCV**: Image processing
- **EasyOCR**: Number plate recognition
- **HSV Color Segmentation**: Color classification
- **Regex-based NLP**: Human command parsing
- **Multithreading**: Parallel video stream processing
- *(Coming Soon)*: LLM-enhanced chat (e.g., GPT) for more advanced instructions

---

## 🛠️ How It Works

### 🎥 1. Vehicle Detection
- Uses YOLOv8 to detect vehicles in each frame.

### 🎨 2. Color Detection
- Converts vehicle image to HSV and checks against defined color masks.

### 🔤 3. Number Plate OCR
- Converts vehicle crop to grayscale
- Applies bilateral filter + adaptive threshold
- Feeds to EasyOCR for plate recognition

### 💬 4. Command Parsing
- Extracts action (e.g. "track"), color (e.g. "black"), and plate number from free-text chat.

---

## 📁 Project Structure

vehicle_monitoring/
├── detection/
│ └── detect_vehicles.py # YOLOv8 vehicle detection
├── color_detection/
│ └── color_detector.py # Dominant color using HSV
├── ocr/
│ └── number_plate_reader.py # EasyOCR pipeline with preprocessing
├── chat/
│ └── chat_command_parser.py # Regex-based parser for user input
├── tests/
│ ├── test_color_detector.py
│ ├── test_number_plate_reader.py
│ ├── test_detect_vehicles.py
│ └── test_command_chat_parser.py
├── main.py # Threaded video processor for multiple cameras
├── sample_videos/ # Input test videos (local only)
├── yolov8n.pt # YOLO model (not pushed to GitHub)
└── requirements.txt

---

## 🚫 Note on YOLO Model

The `yolov8n.pt` model file is **not included** in this repo due to GitHub's file size limits.

Please download it from [Ultralytics](https://github.com/ultralytics/ultralytics) or [Google Drive Link (if hosted)] and place it in the root folder.

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vehicle-monitoring-ai.git
cd vehicle-monitoring-ai
Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:
pip install -r requirements.txt
Run main system:
python main.py
✅ Next Steps
 query_state.py → shared memory for active query
 matcher.py → match vehicle color/plate to chat
 chat_input.py → user can input message during runtime
 (Optional) Integrate with LLM (GPT/OpenAI API) for advanced instructions
📜 License
MIT License – free to use for research, academic, and commercial projects with credit.
🤝 Contributors
Nikhil Gupta – AI Developer, Chat Integration
(You can add more team members here)

Let me know when you finish pushing, and I’ll help you with:
- `query_state.py`
- `matcher.py`
- Live chat CLI (or Streamlit later if needed)  
And finally, packaging it cleanly for presentation or demo!