import cv2
from ultralytics import YOLO
from collections import deque

# ------------------------------------
# CONFIG
# ------------------------------------
MODEL_PATH = "../models/best_large (1).pt"
VIDEO_PATH = "../data/firesense/fire/posVideo9.878.avi"  # change path if needed
SMOOTH_WINDOW = 7  # temporal smoothing frames
CONF_THRESHOLD = 0.4

# Load model
model = YOLO(MODEL_PATH)
print("\n🔥 Model loaded successfully!")

# label mapping
CLASS_MAP = {0: "smoke", 1: "fire"}  #CLASS_MAP = {0: "fire", 1: "smoke"} for large model

# Smoothing queue
label_queue = deque(maxlen=SMOOTH_WINDOW)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Error: Video not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run model prediction
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    detected_label = "no_fire"  # default

    if boxes and len(boxes) > 0:
        # choose highest confidence detection
        best = boxes[0]
        cls_id = int(best.cls[0].item())
        detected_label = CLASS_MAP.get(cls_id, "unknown")

        # draw all detections
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            label = f"{CLASS_MAP[cls_id]} {conf*100:.1f}%"
            color = (0,0,255) if cls_id == 1 else (0,255,255)  # fire:red, smoke:yellow

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Temporal smoothing
    label_queue.append(detected_label)
    final_label = max(set(label_queue), key=label_queue.count)

    # Final label color logic
    if final_label == "fire":
        final_color = (0,0,255)
    elif final_label == "smoke":
        final_color = (0,255,255)
    else:
        final_color = (0,255,0)

    cv2.putText(frame, f"FINAL: {final_label}", (10,35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, final_color, 2)

    cv2.imshow("Fire/Smoke Detection - VIDEO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✔ Video test completed.")
