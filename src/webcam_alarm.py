import cv2
from ultralytics import YOLO
from collections import deque
import os
from datetime import datetime
from playsound import playsound
import threading

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "../models/yolo11-d-fire-dataset.pt"
SMOOTH_WINDOW = 7
CONF_THRESHOLD = 0.4

ALERT_FOLDER = "../alerts"
os.makedirs(ALERT_FOLDER, exist_ok=True)

ALARM_AUDIO = "../alarm.mp3"  # place alarm.mp3 in project root
# ------------------------------

# Load model
model = YOLO(MODEL_PATH)
print("🔥 Model loaded — Starting Webcam ...")

CLASS_MAP = {0: "smoke", 1: "fire"}
label_queue = deque(maxlen=SMOOTH_WINDOW)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found")
    exit()

# -------- Alarm Thread Handler --------
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

# -------- Severity Logic --------
def get_severity(label, fire_area_ratio, smoke_present):
    if label == "no_fire" and not smoke_present:
        return 0
    if label == "smoke" and not fire_area_ratio:
        return 1
    if label == "fire":
        if fire_area_ratio < 0.05:
            return 2
        else:
            return 3
    if smoke_present:
        return 1
    return 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w, _ = frame.shape
    frame_area = frame_w * frame_h

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    detected_label = "no_fire"
    fire_area_total = 0
    smoke_present = False

    if boxes and len(boxes) > 0:
        for b in boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())

            cls_name = CLASS_MAP.get(cls_id, "unknown")
            w, h = x2 - x1, y2 - y1
            box_area = w * h

            if cls_name == "fire":
                fire_area_total += box_area
                detected_label = "fire"
            elif cls_name == "smoke" and detected_label != "fire":
                smoke_present = True
                detected_label = "smoke"

            color = (0,0,255) if cls_name == "fire" else (0,255,255)
            label_text = f"{cls_name} {conf*100:.1f}%"
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label_text,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    label_queue.append(detected_label)
    final_label = max(set(label_queue), key=label_queue.count)

    fire_area_ratio = fire_area_total/frame_area if frame_area>0 else 0
    severity = get_severity(final_label, fire_area_ratio, smoke_present)

    # 🚨 Alarm Control
    if severity >= 2:     # play alarm only for fire severity 2 or 3
        start_alarm()
    else:
        stop_alarm()

    # UI Color
    sev_color = [(0,255,0),(0,255,255),(0,165,255),(0,0,255)][severity]
    sev_text = ["SAFE","WARNING: SMOKE","DANGER: SMALL FIRE","CRITICAL: LARGE FIRE"][severity]

    cv2.putText(frame,f"FINAL: {final_label}",(10,35),
                cv2.FONT_HERSHEY_SIMPLEX,1,sev_color,2)
    cv2.putText(frame,f"SEVERITY: {sev_text}",(10,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,sev_color,2)

    cv2.imshow("Webcam Fire/Smoke Detection + Alarm",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_alarm()
cap.release()
cv2.destroyAllWindows()
print("\n✔ Alarm system stopped.")
