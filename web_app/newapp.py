from flask import Flask, Response, request, jsonify, send_from_directory
import cv2
import sqlite3
import os
import threading
from flask_cors import CORS

# import functions from detection module
from detection.detection_engine import process_frame, stop_alarm_manual

# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploaded_videos"
OUTPUT_DIR = "processed_videos"
SNAPSHOT_DIR = "static/snapshots"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ============================================================
# GLOBAL CAMERA STREAMING (Single Camera)
# ============================================================

streaming = False
camera = None
camera_lock = threading.Lock()

# ============================================================
# MULTI CAMERA SYSTEM
# ============================================================

camera_streams = {}   # {camera_id: {"cap":..., "frame":..., "running":...}}
streams_lock = threading.Lock()


def start_camera_stream(camera_id):
    camera_id = str(camera_id)

    with streams_lock:
        # If already running, skip
        if camera_id in camera_streams and camera_streams[camera_id]["running"]:
            return True

        # attempt to open camera
        try:
            cap = cv2.VideoCapture(int(camera_id))
        except Exception:
            cap = cv2.VideoCapture(camera_id)  # try string (rtsp etc)

        if not cap.isOpened():
            return False

        camera_streams[camera_id] = {
            "cap": cap,
            "frame": None,
            "running": True
        }

        th = threading.Thread(target=update_camera_frame, args=(camera_id,), daemon=True)
        camera_streams[camera_id]["thread"] = th
        th.start()

        return True


def update_camera_frame(camera_id):
    camera_id = str(camera_id)

    while camera_streams.get(camera_id, {}).get("running", False):
        cap = camera_streams[camera_id]["cap"]
        ok, frame = cap.read()

        if not ok:
            # small sleep to avoid tight loop on failures
            import time
            time.sleep(0.1)
            continue

        # process frame with YOLO
        try:
            frame, _, _ = process_frame(frame)
        except Exception:
            # if processing fails, keep original frame
            pass

        camera_streams[camera_id]["frame"] = frame

    # cleanup
    try:
        camera_streams[camera_id]["cap"].release()
    except Exception:
        pass
    camera_streams[camera_id]["running"] = False


def stop_camera_stream(camera_id):
    camera_id = str(camera_id)
    with streams_lock:
        if camera_id in camera_streams:
            camera_streams[camera_id]["running"] = False
            try:
                camera_streams[camera_id]["cap"].release()
            except Exception:
                pass
            # optionally remove the dict entry
            camera_streams.pop(camera_id, None)
            return True
    return False


def generate_camera_stream(camera_id):
    camera_id = str(camera_id)

    # keep serving frames while running
    while True:
        cam = camera_streams.get(camera_id)
        if not cam or not cam.get("running"):
            break

        frame = cam.get("frame")
        if frame is None:
            # small sleep to avoid busy loop
            import time
            time.sleep(0.02)
            continue

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

# ============================================================
# API: START CAMERA (improved: supports single camera id 0 and multi-camera id >=1)
# ============================================================

@app.route("/api/cameras/start", methods=["POST"])
def api_start_camera():
    """
    Expect JSON: { "camera_id": <int_or_string> }
    If camera_id == 0 -> start single global webcam (existing behavior)
    Else -> start per-camera stream using start_camera_stream(camera_id)
    """
    global streaming, camera

    data = request.get_json(force=True) or {}
    cam_id_raw = data.get("camera_id", 0)

    # try to parse integer camera id if possible
    try:
        cam_int = int(cam_id_raw)
    except Exception:
        cam_int = None

    # If cam_id is 0 -> use single-camera pipeline
    if cam_int == 0:
        with camera_lock:
            if camera:
                try:
                    camera.release()
                except Exception:
                    pass

            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not camera.isOpened():
                camera = None
                streaming = False
                return jsonify({"ok": False, "error": "Camera failed to open"}), 500

            streaming = True
            return jsonify({"ok": True, "camera_id": 0})

    # Otherwise use multi-camera system (works for numeric other indexes or RTSP strings)
    else:
        success = start_camera_stream(cam_id_raw)
        if not success:
            return jsonify({"ok": False, "error": "Failed to open camera stream"}), 500
        return jsonify({"ok": True, "camera_id": cam_id_raw})


# ============================================================
# API: STOP CAMERA (supports optional camera_id to stop specific camera)
# ============================================================
@app.route("/api/cameras/stop", methods=["POST"])
def api_stop_camera():
    """
    JSON body may include camera_id.
    - If camera_id == 0 or not provided -> stop global single-camera streaming.
    - Else -> stop that per-camera stream.
    """
    global streaming, camera

    data = request.get_json(silent=True) or {}
    cam_id_raw = data.get("camera_id", None)

    # if camera_id not provided -> stop single global camera
    if cam_id_raw is None:
        # stop global camera
        streaming = False
        if camera:
            try:
                camera.release()
            except Exception:
                pass
            camera = None
        return jsonify({"ok": True, "stopped": "global"})

    # try parse integer
    try:
        cam_int = int(cam_id_raw)
    except Exception:
        cam_int = None

    if cam_int == 0:
        # stop global
        streaming = False
        if camera:
            try:
                camera.release()
            except Exception:
                pass
            camera = None
        return jsonify({"ok": True, "stopped": "global"})
    else:
        stopped = stop_camera_stream(cam_id_raw)
        return jsonify({"ok": stopped, "stopped": cam_id_raw if stopped else None})


# ============================================================
# SINGLE CAMERA STREAM generator (existing behavior)
# ============================================================
def generate_frames():
    global streaming, camera

    while streaming:
        if camera is None:
            break

        ok, frame = camera.read()
        if not ok:
            # small sleep to avoid busy loop on errors
            import time
            time.sleep(0.05)
            continue

        try:
            frame, _, _ = process_frame(frame)
        except Exception:
            pass

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

    if camera:
        try:
            camera.release()
        except Exception:
            pass


# ============================================================
# Route: /video_feed/<camera_id> - serves single or multi-camera generator
# ============================================================
@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    """
    If camera_id == "0" -> return single-camera streaming
    If camera_id matches a running multi-camera stream -> return that stream
    Otherwise return 404
    """
    # if numeric and equals 0 -> single camera
    try:
        if int(camera_id) == 0:
            return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    except Exception:
        pass

    # if multi-camera stream exists, return generator
    if camera_id in camera_streams and camera_streams[camera_id]["running"]:
        return Response(generate_camera_stream(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")

    # If not running, try to start it on demand
    started = start_camera_stream(camera_id)
    if started:
        return Response(generate_camera_stream(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")

    return "Camera stream not available", 404


# ============================================================
# VIDEO STREAM FROM UPLOADED FILE (Live Processed)
# ============================================================
@app.route("/video_stream/<path:filename>")
def video_stream(filename):
    video_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(video_path):
        return "Video not found", 404

    def generate():
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                frame, _, _ = process_frame(frame)
            except Exception:
                pass

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                buffer.tobytes() +
                b"\r\n"
            )

        cap.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================================================
# PROCESS VIDEO UPLOAD → RETURN STREAM URL
# ============================================================
@app.route("/process_video", methods=["POST"])
def process_video():
    file = request.files.get("video")
    if not file:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400

    filename = file.filename
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    return jsonify({
        "ok": True,
        "stream_url": f"/video_stream/{filename}"
    })


# ============================================================
# GET ALERT LOGS
# ============================================================
@app.route("/api/events")
def api_events():
    conn = sqlite3.connect("alerts.db")
    conn.row_factory = sqlite3.Row
    rows = conn.cursor().execute("SELECT * FROM alerts ORDER BY id DESC").fetchall()
    conn.close()

    return jsonify([dict(r) for r in rows])


# ============================================================
# DELETE SELECTED LOGS
# ============================================================
@app.route("/api/events/delete", methods=["POST"])
def api_delete_logs():
    data = request.get_json() or {}
    ids = data.get("ids", [])

    if not ids:
        return jsonify({"ok": False, "error": "No IDs provided"}), 400

    conn = sqlite3.connect("alerts.db")
    conn.executemany("DELETE FROM alerts WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    conn.close()

    return jsonify({"ok": True})


# ============================================================
# DELETE ALL LOGS
# ============================================================
@app.route("/api/events/delete_all", methods=["POST"])
def api_delete_all_logs():
    conn = sqlite3.connect("alerts.db")
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()

    return jsonify({"ok": True})


# ============================================================
# DASHBOARD API (frontend expects this structure)
# ============================================================
@app.route("/api/dashboard")
def api_dashboard():
    try:
        conn = sqlite3.connect("alerts.db")
        cursor = conn.cursor()

        # Label counts
        cursor.execute("SELECT label, COUNT(*) FROM alerts GROUP BY label")
        rows = cursor.fetchall()

        labels = [row[0] for row in rows]
        label_values = [row[1] for row in rows]

        # Severity counts
        cursor.execute("SELECT severity, COUNT(*) FROM alerts GROUP BY severity")
        rows = cursor.fetchall()

        severity_values = [0, 0, 0, 0]   # safe, smoke, small fire, large fire
        for sev, count in rows:
            severity_values[int(sev)] = count

        # Alerts per day
        cursor.execute("""
            SELECT DATE(timestamp), COUNT(*)
            FROM alerts
            GROUP BY DATE(timestamp)
            ORDER BY DATE(timestamp)
        """)
        rows = cursor.fetchall()

        daily_labels = [row[0] for row in rows]
        daily_values = [row[1] for row in rows]

        conn.close()

        return jsonify({
            "labels": labels,
            "label_values": label_values,
            "severity_values": severity_values,
            "daily_labels": daily_labels,
            "daily_values": daily_values
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# STOP ALARM (single route, calls detection_engine.stop_alarm_manual)
# ============================================================
@app.route("/api/alarm/stop", methods=["POST"])
def api_stop_alarm():
    try:
        stop_alarm_manual()
        return jsonify({"ok": True, "message": "Alarm stopped manually"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ============================================================
# SERVE FILES
# ============================================================
@app.route("/uploaded_videos/<path:filename>")
def serve_uploaded(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/processed_videos/<path:filename>")
def serve_processed(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/static/snapshots/<path:filename>")
def serve_snapshots(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


# ============================================================
# ROOT
# ============================================================
@app.route("/")
def home():
    return "🔥 FireGuard Backend Running"


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
