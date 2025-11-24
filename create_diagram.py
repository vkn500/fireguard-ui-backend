from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import Users, Client
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL
from diagrams.programming.framework import React, Flask
from diagrams.programming.language import Python, NodeJS
from diagrams.onprem.network import Nginx
from diagrams.custom import Custom
import os

# Set output format and filename
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # Windows Graphviz path (adjust if needed)

with Diagram("FireGuard AI System Architecture", 
             filename="fireguard_architecture", 
             show=False, 
             direction="TB",
             graph_attr={"fontsize": "16", "bgcolor": "white"}):
    
    # ==========================================
    # INPUT LAYER
    # ==========================================
    with Cluster("INPUT LAYER"):
        usb_cam = Custom("USB Webcam\n(Camera ID: 0)", "./icons/webcam.png") if os.path.exists("./icons/webcam.png") else Client("USB Webcam\n(ID: 0)")
        ip_cam = Custom("IP Cameras\n(RTSP/HTTP)", "./icons/ipcam.png") if os.path.exists("./icons/ipcam.png") else Client("IP Cameras\n(ID: 1+)")
        video_upload = Custom("Video Upload\n(.mp4/.avi)", "./icons/video.png") if os.path.exists("./icons/video.png") else Client("Video Upload")
    
    # ==========================================
    # PROCESSING LAYER (BACKEND)
    # ==========================================
    with Cluster("PROCESSING LAYER (Flask Backend)"):
        
        # Flask API Server
        flask_server = Flask("Flask API Server\n(app.py)\nPort: 5000")
        
        with Cluster("Camera Management"):
            single_cam = Python("Single Camera\nStream (ID=0)\nThread-based")
            multi_cam = Python("Multi-Camera\nStreams (ID≥1)\nParallel Threads")
        
        with Cluster("Detection Engine (detection_engine.py)"):
            yolo = Python("YOLOv11 Model\n(yolo11-d-fire-dataset.pt)\nRTX 3050 GPU")
            temporal = Python("Temporal Smoothing\n7-Frame Queue\nMajority Voting")
            severity = Python("Severity Classifier\n4 Levels (0-3)\nFire Area Calculation")
        
        with Cluster("Alert System"):
            alarm = Python("Alarm System\nwinsound (Windows)\nManual Override")
            email = NodeJS("Email Service\nNode.js SMTP\n10-min Cooldown")
        
        with Cluster("Evidence Logger"):
            snapshot = Python("Snapshot Capture\n5 Images/Incident\nJPEG Format")
            database = PostgreSQL("SQLite Database\nalerts.db\nIncident Metadata")
    
    # ==========================================
    # PRESENTATION LAYER (FRONTEND)
    # ==========================================
    with Cluster("PRESENTATION LAYER (React Frontend)"):
        react_app = React("React + TypeScript\nVite Build Tool\nTailwind CSS")
        
        with Cluster("Pages"):
            home = Client("Home\nVideo Upload\nLive Stream")
            monitoring = Client("Multi-Camera\nGrid Dashboard\n2×2, 3×3, 4×4")
            logs = Client("Incident Logs\nSnapshot Gallery\nDelete/Filter")
            dashboard = Client("Analytics\nCharts & Stats\nReal-time Updates")
    
    # ==========================================
    # EXTERNAL SERVICES
    # ==========================================
    with Cluster("External Services"):
        smtp_server = Server("SMTP Server\nGmail/Outlook\nTLS Encryption")
        browser = Users("End Users\nSecurity Personnel\nFire Safety Team")
    
    # ==========================================
    # CONNECTIONS (Data Flow)
    # ==========================================
    
    # Input → Flask
    usb_cam >> Edge(label="USB Stream\n30 FPS") >> flask_server
    ip_cam >> Edge(label="RTSP/HTTP\n30 FPS") >> flask_server
    video_upload >> Edge(label="POST /process_video\nMultipart Upload") >> flask_server
    
    # Flask → Camera Management
    flask_server >> Edge(label="POST /api/cameras/start") >> single_cam
    flask_server >> Edge(label="POST /api/cameras/start\n{camera_id: 1+}") >> multi_cam
    
    # Camera → Detection Engine
    single_cam >> Edge(label="Frame Data\n640×640") >> yolo
    multi_cam >> Edge(label="Parallel Frames\n640×640") >> yolo
    
    # Detection Pipeline
    yolo >> Edge(label="Raw Labels\n(fire/smoke)") >> temporal
    temporal >> Edge(label="Smoothed Label\nMajority Vote") >> severity
    
    # Severity → Alerts
    severity >> Edge(label="Severity ≥ 2\nTrigger Alarm") >> alarm
    severity >> Edge(label="Severity ≥ 2\nSend Alert") >> email
    
    # Email → SMTP
    email >> Edge(label="SMTP Protocol\nHTML Email") >> smtp_server
    
    # Evidence Logging
    severity >> Edge(label="Save Frame\nMax 5/Incident") >> snapshot
    snapshot >> Edge(label="INSERT INTO alerts\nTimestamp, Severity") >> database
    
    # Flask → Frontend (API Responses)
    flask_server >> Edge(label="GET /video_feed/<id>\nMJPEG Stream") >> react_app
    flask_server >> Edge(label="GET /api/events\nJSON Response") >> react_app
    flask_server >> Edge(label="GET /api/dashboard\nChart Data") >> react_app
    database >> Edge(label="SELECT * FROM alerts\nORDER BY id DESC") >> flask_server
    
    # Frontend → Pages
    react_app >> home
    react_app >> monitoring
    react_app >> logs
    react_app >> dashboard
    
    # Users → Frontend
    browser >> Edge(label="HTTPS\nPort: 3000") >> react_app
    
    # Alert Notification Path
    smtp_server >> Edge(label="Email Delivery\nAlert Notification") >> browser

print("✅ Architecture diagram generated: fireguard_architecture.png")
