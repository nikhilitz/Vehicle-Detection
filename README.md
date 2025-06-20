# 🚘 Vehicle Monitoring with AI Chat

A real-time AI surveillance system that detects vehicles in live video streams, reads their number plates, identifies colors, stores the information in a database, and responds to natural chat commands like:

> “Show all black cars in the parking lot”  
> “Track the red car with number plate RJ14AB1234”

---

## 📦 Features

| Feature                           | Status    |
|----------------------------------|-----------|
| Vehicle detection using YOLOv8   | ✅ Done    |
| Vehicle color classification     | ✅ Done    |
| Number plate recognition (OCR)   | ✅ Done    |
| Chat command parser (Regex NLP)  | ✅ Done    |
| Multi-camera video processing    | ✅ Done    |
| SQLite-based vehicle database    | ✅ Done    |
| Query matching and retrieval     | ✅ Done    |
| Live chat interface (Streamlit)  | ✅ Done    |
| LLM integration for commands     | 🔜 Planned |

---

## 🧠 Tech Stack

- **YOLOv8 / YOLOS**: For vehicle and license plate detection
- **EasyOCR**: For number plate text recognition
- **OpenCV + HSV**: For dominant vehicle color classification
- **Regex-based NLP**: For chat command understanding
- **SQLite3**: For storing plate, color, camera, and timestamp
- **Python Multiprocessing**: For handling multiple video feeds
- **Streamlit**: For interactive chat-based UI

---

## 🛠️ How It Works

### 🎥 Vehicle & Plate Detection
- YOLOv8 (`yolov8x.pt`) detects cars, buses, trucks, bikes.
- YOLOS (`nickmuchi/yolos-small-finetuned-license-plate-detection`) finds number plates.

### 🎨 Color Detection
- Uses HSV-based segmentation to detect white, black, gray, red, etc.

### 🔤 OCR Pipeline
- Extracted plates are enhanced with filters, resized, and passed to EasyOCR.
- Includes correction map (e.g., O→0, B→8) to reduce misreads.

### 💬 Chat Command Parser
- Input like `"track black car RJ14AB1234"` gets parsed into:
```json
{
  "action": "track",
  "color": "black",
  "plate": "RJ14AB1234"
}

🗂️ Project Structure
vehicle-monitoring/
├── chat/
│   ├── chat_command_parser.py   # NLP command parsing
│   └── query_state.py           # Stores current command
├── color_detection/
│   └── color_detector.py        # HSV color classifier
├── detection/
│   └── detect_vehicles.py       # YOLOv8 vehicle detector
├── ocr/
│   └── number_plate_reader.py   # EasyOCR plate reader
├── pipeline/
│   └── runner.py                # Multiprocessing runner for live cameras
├── storage/
│   ├── database.py              # SQLite insert, query, delete
│   └── check_db.py              # CLI tool to view entries
├── ui/
│   └── chat_input_streamlit.py  # Streamlit UI for chat commands
├── sample_videos/               # Test videos for cameras
├── sample_images/               # Debug images
├── yolov8x.pt                   # YOLOv8 weights (manually placed)
├── requirements.txt             # All dependencies
├── README.md
└── venv/                        # Python virtual environment
▶️ Running the System
1. 🔧 Setup
# Clone repo
git clone https://github.com/yourusername/vehicle-monitoring-ai.git
cd vehicle-monitoring-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
⚠️ Download yolov8x.pt manually from Ultralytics or your preferred source and place it in the root folder.
2. 🎥 Start Vehicle Monitoring
python pipeline/runner.py
This will process all videos in sample_videos/
Runs YOLO + OCR + Color classifier in parallel for each camera
Saves detections to SQLite (storage/vehicle_data.db)
Cropped plates & debug images saved to /debug/
3. 💬 Start Chat UI (Streamlit)
streamlit run ui/chat_input_streamlit.py
Enter natural language commands like:
"track red car RJ14AB1234"
"show all white vehicles"
See matching vehicle info from live DB
🧪 Testing Database
python storage/check_db.py
View all detected vehicles stored so far.
📈 Next Features (Roadmap)
 GPT/LLM chat support for more complex commands
 Live camera feed UI
 Notification system for matches
 Export/Report generation
📜 License
MIT License – free to use for research, academic, and commercial use with proper credit.
🤝 Contributors
Nikhil Gupta – AI Developer, Chat Integration
(Add more contributors if needed)

Let me know if you want:
- A lighter version for submission
- A one-pager for a poster/demo
- Or a `.pdf` version of this README for documentation

Ready to go 🚀