# FireGuard Backend Web App

This project is a backend web application for real-time fire and smoke detection using computer vision (YOLO), with alerting, logging, and a dashboard. It is built with Flask and OpenCV, and provides APIs for video streaming, event logging, and dashboard analytics.

## Features
- Real-time fire and smoke detection from webcam, RTSP, or uploaded videos
- Multi-camera support
- Alert logging to SQLite database
- Email alerts (Node.js script)
- REST APIs for dashboard and event management
- Serve processed video streams and snapshots

## Getting Started

### 1. Clone the Repository
```sh
git clone <repo-url>
cd <repo-folder>
```

### 2. Create and Activate a Virtual Environment
```sh
# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux/Mac:
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Python Packages
```sh
pip install -r web_app/requirements.txt
```

### 4. (Optional) Install Node.js for Email Alerts
- If you want email alerts, install Node.js and run `npm install` in the `alert_email` folder.

### 5. Run Database Setup (First Time Only)
```sh
python web_app/db_setup.py
```

### 6. Run the Backend Web App
```sh
python web_app/newapp.py
```

The app will start on `http://127.0.0.1:5000/` by default.

## API Endpoints
- `/api/cameras/start` : Start camera stream
- `/api/cameras/stop` : Stop camera stream
- `/video_feed/<camera_id>` : MJPEG video stream
- `/process_video` : Upload and process video
- `/api/events` : Get alert logs
- `/api/dashboard` : Dashboard analytics

## Notes
- Place your YOLO model file in `web_app/detection/yolo11-d-fire-dataset.pt`.
- Alarm sound file path is set in `detection_engine.py`.
- Snapshots and processed videos are saved in `web_app/static/snapshots` and `web_app/processed_videos`.

## Requirements
- Python 3.8+
- OpenCV
- Flask
- Flask-CORS
- ultralytics (YOLO)
- playsound
- winsound (Windows only)

## License
MIT
