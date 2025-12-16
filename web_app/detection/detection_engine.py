import cv2
from ultralytics import YOLO
from collections import deque
import threading
from playsound import playsound
import subprocess
import time
import sqlite3
import os
import winsound
from detection.anti_spoofing import TemporalConsistencyFilter


# =========================================
# CONFIGURATION
# =========================================
MODEL_PATH = "detection/yolo11-d-fire-dataset.pt"
ALARM_SOUND = "../alarm-301729 (1).wav.crdownload"
SNAPSHOT_DIR = "static/snapshots"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

EMAIL_COOLDOWN = 600   # 10 minutes

# =========================================
# STATE TRACKING (for clean console output)
# =========================================
last_alarm_state = None  # 'running' or 'stopped'
last_spoofing_state = None  # True (spoofing detected) or False (real fire)
last_incident_state = None  # True (incident active) or False (safe)

# ANTI-SPOOFING
ANTI_SPOOFING_ENABLED = True
anti_spoofing_detector = TemporalConsistencyFilter()

alarm_process = None
alarm_playing = False
email_sent = False
last_email_time = 0

# INCIDENT SYSTEM
in_incident = False
incident_label = None
incident_snap_count = 0
incident_last_seen = 0
MAX_SNAPS = 5
INCIDENT_TIMEOUT = 5

# MANUAL ALARM OVERRIDE
manual_alarm_override = False

# Load YOLO model
model = YOLO(MODEL_PATH)
CLASS_MAP = {0: "smoke", 1: "fire"}
label_queue = deque(maxlen=7)


# =========================================
# DATABASE
# =========================================
def save_alert_to_db(timestamp, label, severity, snapshot_path):
    conn = sqlite3.connect("alerts.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO alerts (timestamp, label, severity, snapshot_path)
        VALUES (?, ?, ?, ?)
    """, (timestamp, label, severity, snapshot_path))
    conn.commit()
    conn.close()


# =========================================
# ALARM SYSTEM
# =========================================
def play_alarm():
    global alarm_playing
    while alarm_playing:
        playsound(ALARM_SOUND)


def start_alarm():
    global alarm_playing, manual_alarm_override
    if alarm_playing or manual_alarm_override:
        return

    alarm_playing = True
    winsound.PlaySound(ALARM_SOUND, winsound.SND_ASYNC)


def stop_alarm():
    global alarm_playing
    winsound.PlaySound(None, winsound.SND_PURGE)
    alarm_playing = False


def stop_alarm_manual():
    global manual_alarm_override
    manual_alarm_override = True
    stop_alarm()
    print("🛑 Manual alarm override activated")


# =========================================
# EMAIL SYSTEM
# =========================================
def send_email(status, severity):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    threading.Thread(
        target=subprocess.Popen,
        args=([
            "node",
            "../alert_email/sendMail.js",
            status,
            str(severity),
            timestamp
        ],),
        daemon=True
    ).start()

    print("📧 Email alert sent!")


# =========================================
# SEVERITY LOGIC
# =========================================
def compute_severity(label, fire_ratio, smoke_present):
    if label == "no_fire" and not smoke_present:
        return 0
    if label == "smoke" and not fire_ratio:
        return 1
    if label == "fire":
        return 2 if fire_ratio < 0.05 else 3
    if smoke_present:
        return 1
    return 0


# =========================================
# MAIN PROCESSING FUNCTION
# =========================================
def process_frame(frame):
    global email_sent, last_email_time
    global in_incident, incident_label, incident_snap_count, incident_last_seen
    global manual_alarm_override
    global alarm_playing
    global last_alarm_state, last_spoofing_state, last_incident_state
    global anti_spoofing_detector

    h, w, _ = frame.shape
    total_area = w * h

    results = model(frame, conf=0.4, verbose=False)
    boxes = results[0].boxes

    fire_area = 0
    smoke_present = False
    detected_label = "no_fire"

    # =====================================
    # YOLO DETECTION
    # =====================================
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cls_id = int(b.cls[0].item())
            cls_name = CLASS_MAP.get(cls_id, "unknown")
            conf = float(b.conf[0].item()) * 100

            color = (0, 0, 255) if cls_name == "fire" else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.1f}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 2)

            area = (x2 - x1) * (y2 - y1)

            if cls_name == "fire":
                fire_area += area
                detected_label = "fire"
            elif cls_name == "smoke" and detected_label != "fire":
                smoke_present = True
                detected_label = "smoke"

    # =====================================
    # TEMPORAL SMOOTHING
    # =====================================
    if detected_label == "no_fire":
        label_queue.clear()
        label_queue.append("no_fire")
    else:
        label_queue.append(detected_label)

    final_label = max(set(label_queue), key=label_queue.count)
    fire_ratio = fire_area / total_area if total_area > 0 else 0

    severity = compute_severity(final_label, fire_ratio, smoke_present)

    now = time.time()
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # =====================================
    # 🆕 ANTI-SPOOFING CHECK
    # =====================================
    is_real_fire = True  # default

    if ANTI_SPOOFING_ENABLED and final_label != "no_fire":
        conf_value = 0.85  # default confidence
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            conf_value = float(results[0].boxes[0].conf[0])

        anti_spoofing_detector.add_detection(final_label, conf_value)
        is_real_fire, spoofing_reason, _ = anti_spoofing_detector.is_real_fire()

        # ONLY PRINT IF STATE CHANGED
        if not is_real_fire:
            if last_spoofing_state != True:  # State changed to SPOOFING
                print(f"❌ SPOOFING DETECTED: {spoofing_reason}")
                last_spoofing_state = True
            severity = 0
            final_label = "no_fire"
        else:
            if last_spoofing_state != False:  # State changed to REAL FIRE
                print(f"✅ Real fire verified - {spoofing_reason}")
                last_spoofing_state = False
    else:
        last_spoofing_state = None

    # ======================================================
    # 🔥 INCIDENT CONTROL (with state tracking)
    # ======================================================
    if severity >= 2:
        # Only print once when alarm STARTS (not every frame)
        if not alarm_playing:
            if last_alarm_state != 'running':
                print("🔊 ALARM STARTED")
                last_alarm_state = 'running'

        if not manual_alarm_override:
            start_alarm()

        # Start a new incident
        if not in_incident:
            in_incident = True
            incident_label = final_label
            incident_snap_count = 0
            manual_alarm_override = False
            print(f"🔥 NEW INCIDENT: {final_label}")
            last_incident_state = True

        incident_last_seen = now

        # Save snapshots
        if incident_snap_count < MAX_SNAPS:
            snapshot_name = f"{int(now)}_{final_label}_sev{severity}.jpg"
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_name)
            cv2.imwrite(snapshot_path, frame)
            save_alert_to_db(timestamp_str, final_label, severity, snapshot_path)
            incident_snap_count += 1
            print(f"📸 Snapshot {incident_snap_count}/{MAX_SNAPS}")

        # Email cooldown
        if (not email_sent) or (now - last_email_time > EMAIL_COOLDOWN):
            send_email(final_label, severity)
            email_sent = True
            last_email_time = now

    else:
        # Only print once when alarm STOPS (not every frame)
        if alarm_playing or last_alarm_state == 'running':
            if last_alarm_state != 'stopped':
                print("🔕 ALARM STOPPED")
                last_alarm_state = 'stopped'

        if not manual_alarm_override:
            stop_alarm()
            email_sent = False

        # End incident after 5 sec inactivity
        if in_incident and (now - incident_last_seen > INCIDENT_TIMEOUT):
            if last_incident_state != False:
                print("✅ INCIDENT ENDED")
                last_incident_state = False
            in_incident = False
            incident_label = None
            incident_snap_count = 0
            manual_alarm_override = False

    # =====================================
    # DRAW SEVERITY TEXT
    # =====================================
    severity_colors = [
        (0, 255, 0),
        (0, 255, 255),
        (0, 165, 255),
        (0, 0, 255)
    ]

    severity_text = [
        "SAFE",
        "SMOKE WARNING",
        "SMALL FIRE",
        "LARGE FIRE"
    ]

    cv2.putText(frame, f"FINAL: {final_label}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1, severity_colors[severity], 2)

    cv2.putText(frame, f"SEVERITY: {severity_text[severity]}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, severity_colors[severity], 2)

    return frame, final_label, severity