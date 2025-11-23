import cv2
from ultralytics import YOLO
from collections import deque

# ------------------------------------
# CONFIG
# ------------------------------------
MODEL_PATH = "runs/detect/fire_smoke_yolo11/weights/best.pt"
VIDEO_PATH = "../data/firesense/smoke/testpos08.824.avi"  # change to your video path
SMOOTH_WINDOW = 7
CONF_THRESHOLD = 0.4

# Load model
model = YOLO(MODEL_PATH)
print("\n🔥 Model loaded successfully!")

CLASS_MAP = {0: "smoke", 1: "fire"}
label_queue = deque(maxlen=SMOOTH_WINDOW)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: could not open video:", VIDEO_PATH)
    exit()

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_area = frame_w * frame_h

def get_severity(label, fire_area_ratio, smoke_present):
    """
    Simple severity rule:
    - 0: no_fire
    - 1: smoke only / very small area
    - 2: big smoke or small fire
    - 3: large fire
    """
    if label == "no_fire" and not smoke_present:
        return 0
    if label == "smoke" and not fire_area_ratio:
        return 1
    if label == "fire":
        if fire_area_ratio < 0.05:   # <5% of frame
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

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    detected_label = "no_fire"
    fire_area_total = 0
    smoke_present = False

    if boxes and len(boxes) > 0:
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())

            cls_name = CLASS_MAP.get(cls_id, "unknown")
            w = x2 - x1
            h = y2 - y1
            box_area = w * h

            if cls_name == "fire":
                fire_area_total += box_area
                detected_label = "fire"
            elif cls_name == "smoke" and detected_label != "fire":
                smoke_present = True
                if detected_label != "fire":
                    detected_label = "smoke"

            color = (0,0,255) if cls_name == "fire" else (0,255,255)
            label_text = f"{cls_name} {conf*100:.1f}%"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label_text, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Temporal smoothing of main label
    label_queue.append(detected_label)
    final_label = max(set(label_queue), key=label_queue.count)

    fire_area_ratio = fire_area_total / frame_area if frame_area > 0 else 0.0
    severity = get_severity(final_label, fire_area_ratio, smoke_present)

    # Formatting severity
    severity_text = {
        0: "SAFE",
        1: "WARNING: SMOKE",
        2: "DANGER: FIRE (SMALL)",
        3: "CRITICAL: FIRE (LARGE)"
    }[severity]

    if severity == 0:
        sev_color = (0,255,0)
    elif severity == 1:
        sev_color = (0,255,255)
    elif severity == 2:
        sev_color = (0,165,255)
    else:
        sev_color = (0,0,255)

    cv2.putText(frame, f"FINAL: {final_label}", (10,35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, sev_color, 2)
    cv2.putText(frame, f"SEVERITY: {severity_text}", (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, sev_color, 2)

    cv2.imshow("Fire/Smoke Detection - VIDEO (Severity)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✔ Video severity test completed.")
