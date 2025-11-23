import cv2
from ultralytics import YOLO
from collections import deque
import os
from datetime import datetime
from playsound import playsound
import threading
import subprocess
import time

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "../models/yolo11-d-fire-dataset.pt"

VIDEO_PATH = "../data/firesense/fire/posVideo10.869.avi"   # change if needed

SMOOTH_WINDOW = 7
CONF_THRESHOLD = 0.4

ALERT_FOLDER = "../alerts"
os.makedirs(ALERT_FOLDER, exist_ok=True)

ALARM_AUDIO = "../alarm-301729.mp3"

# Anti-spam settings
EMAIL_COOLDOWN_SECONDS = 600   # 10 min
email_sent = False
last_email_time = 0
# ------------------------------

# Load model
model = YOLO(MODEL_PATH)
print("🔥 Model loaded — Starting Video Monitoring ...")

CLASS_MAP = {0: "smoke", 1: "fire"}
label_queue = deque(maxlen=SMOOTH_WINDOW)

# ------------------------------
# Video Reader
# ------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Could not open video: {VIDEO_PATH}")
    exit()

# ------------------------------
# Alarm Controller
# ------------------------------
alarm_playing = False

def play_alarm():
    global alarm_playing
    while alarm_playing:
        playsound(ALARM_AUDIO)

def start_alarm():
    global alarm_playing
    if not alarm_playing:
        alarm_playing = True
        threading.Thread(target=play_alarm, daemon=True).start()

def stop_alarm():
    global alarm_playing
    alarm_playing = False


# ------------------------------
# Severity Logic
# ------------------------------
def get_severity(label, fire_area_ratio, smoke_present):
    if label == "no_fire" and not smoke_present:
        return 0
    if label == "smoke" and not fire_area_ratio:
        return 1
    if label == "fire":
        return 2 if fire_area_ratio < 0.05 else 3
    if smoke_present:
        return 1
    return 0


# ------------------------------
# Threaded Email (non-blocking)
# ------------------------------
def send_email_thread(status, severity, timestamp):
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

    print("📧 Email Alert Sent (Threaded)")


# ------------------------------
# MAIN VIDEO LOOP
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("\n📄 Video Finished")
        break

    frame_h, frame_w, _ = frame.shape
    frame_area = frame_w * frame_h

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    detected_label = "no_fire"
    fire_area_total = 0
    smoke_present = False

    # Fire/Smoke detection
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cls_id = int(b.cls[0].item())
            cls_name = CLASS_MAP.get(cls_id, "unknown")
            conf = round(float(b.conf[0].item()) * 100, 1)

            w, h = (x2 - x1), (y2 - y1)
            box_area = w * h

            if cls_name == "fire":
                detected_label = "fire"
                fire_area_total += box_area
            elif cls_name == "smoke" and detected_label != "fire":
                detected_label = "smoke"
                smoke_present = True

            color = (0,0,255) if cls_name == "fire" else (0,255,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,f"{cls_name} {conf}%",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # temporal smoothing
    label_queue.append(detected_label)
    final_label = max(set(label_queue), key=label_queue.count)

    fire_area_ratio = fire_area_total / frame_area if frame_area > 0 else 0
    severity = get_severity(final_label, fire_area_ratio, smoke_present)

    # ------------------------------
    # ANTI-SPAM EMAIL LOGIC
    # ------------------------------
    if severity >= 2:
        start_alarm()

        current_time = time.time()

        # Only send once per fire, or after cooldown
        if (not email_sent) or (current_time - last_email_time > EMAIL_COOLDOWN_SECONDS):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            send_email_thread(final_label, severity, timestamp)

            email_sent = True
            last_email_time = current_time

    else:
        stop_alarm()
        email_sent = False
    # ------------------------------

    # UI overlay
    sev_color = [(0,255,0),(0,255,255),(0,165,255),(0,0,255)][severity]
    sev_text  = ["SAFE","WARNING: SMOKE","DANGER: SMALL FIRE","CRITICAL: LARGE FIRE"][severity]

    cv2.putText(frame,f"FINAL: {final_label}",(10,35),
                cv2.FONT_HERSHEY_SIMPLEX,1,sev_color,2)
    cv2.putText(frame,f"SEVERITY: {sev_text}",(10,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,sev_color,2)

    cv2.imshow("🔥 Video Fire Detection | Alarm + Email (Anti-Spam)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

stop_alarm()
cap.release()
cv2.destroyAllWindows()
print("\n✔ Video Monitoring Stopped")
