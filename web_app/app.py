from flask import Flask, render_template, Response, request
import cv2
from detection.detection_engine import process_frame
import sqlite3
import json


app = Flask(__name__, static_folder="static")


# Global control
streaming = False
camera = None


# ------------------------
# Start monitoring (route)
# ------------------------
@app.route("/monitor")
def monitor():
    global streaming, camera

    cam_id = request.args.get("camera_id", "0")

    streaming = True

    if cam_id == "rtsp":
        rtsp_url = request.args.get("rtsp_url")
        camera = cv2.VideoCapture(rtsp_url)
        cam_name = rtsp_url
    else:
        cam_index = int(cam_id)
        camera = cv2.VideoCapture(cam_index)
        cam_name = f"Webcam {cam_index}"

    if not camera.isOpened():
        return f"❌ Failed to open camera: {cam_name}"

    return render_template("monitor.html", cam_name=cam_name)



# ------------------------
# Stop monitoring API
# ------------------------
@app.route("/stop_stream")
def stop_stream():
    global streaming, camera
    streaming = False

    if camera is not None:
        camera.release()
        camera = None

    return "stopped"


# ------------------------
# Frame generator
# ------------------------
def generate_frames():
    global streaming, camera

    while streaming:
        if camera is None:
            break

        success, frame = camera.read()
        if not success:
            break

        # Apply YOLO detection
        frame, _, _ = process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame_bytes + b'\r\n')

    # Stop camera safely
    if camera is not None:
        camera.release()


# ------------------------
# Video feed route
# ------------------------
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------------
# Home & Logs
# ------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/logs")
def logs():
    conn = sqlite3.connect("alerts.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM alerts ORDER BY id DESC")
    records = cursor.fetchall()

    conn.close()
    return render_template("logs.html", records=records)

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("alerts.db")
    cursor = conn.cursor()

    cursor.execute("SELECT label, COUNT(*) FROM alerts GROUP BY label")
    label_counts = cursor.fetchall()

    cursor.execute("SELECT severity, COUNT(*) FROM alerts GROUP BY severity")
    severity_counts = cursor.fetchall()

    cursor.execute("""
        SELECT DATE(timestamp), COUNT(*)
        FROM alerts
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """)
    daily_counts = cursor.fetchall()

    conn.close()

    # Convert to clean JSON
    data = {
        "labels": [row[0] for row in label_counts],
        "label_values": [row[1] for row in label_counts],

        "severity_values": [row[1] for row in severity_counts],

        "daily_labels": [row[0] for row in daily_counts],
        "daily_values": [row[1] for row in daily_counts]
    }

    return render_template("dashboard.html",
                           chart_data=json.dumps(data))

# ------------------------
# Run Flask
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
