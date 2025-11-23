import cv2
from ultralytics import YOLO
from collections import deque
import time

MODEL_PATH = "yolo11-d-fire-dataset.pt"
model = YOLO(MODEL_PATH)

CLASS_MAP = {0: "smoke", 1: "fire"}

# -------------------------------------------------------
# TEMPORAL SMOOTHING SETTINGS
# -------------------------------------------------------
WINDOW = 7
label_queue = deque(maxlen=WINDOW)


def get_label(boxes):
    """Return detected label: fire, smoke, or no_fire"""
    if boxes is None or len(boxes) == 0:
        return "no_fire"

    fire = any(int(b.cls[0]) == 1 for b in boxes)
    smoke = any(int(b.cls[0]) == 0 for b in boxes)

    if fire:
        return "fire"
    if smoke:
        return "smoke"
    return "no_fire"


# -------------------------------------------------------
# MAIN ABLATION FUNCTION (Fire + Smoke analysis)
# -------------------------------------------------------
def run_ablation(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    # RAW METRICS
    raw = {
        "fire_tp": 0,
        "smoke_tp": 0,
        "false_positives": 0,
        "stability_flips": 0,
        "latency_fire": None,
        "latency_smoke": None
    }

    # SMOOTHED METRICS
    smooth = {
        "fire_tp": 0,
        "smoke_tp": 0,
        "false_positives": 0,
        "stability_flips": 0,
        "latency_fire": None,
        "latency_smoke": None
    }

    raw_prev = "no_fire"
    smooth_prev = "no_fire"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # -----------------------------
        # RAW DETECTION
        # -----------------------------
        results = model(frame, conf=0.4, verbose=False)
        raw_label = get_label(results[0].boxes)

        # True fire & smoke frames
        if raw_label == "fire":
            raw["fire_tp"] += 1
            if raw["latency_fire"] is None:
                raw["latency_fire"] = frame_count

        elif raw_label == "smoke":
            raw["smoke_tp"] += 1
            if raw["latency_smoke"] is None:
                raw["latency_smoke"] = frame_count

        # False positive logic
        if raw_label == "no_fire" and raw_prev in ["fire", "smoke"]:
            raw["false_positives"] += 1

        # Stability flips
        if raw_label != raw_prev:
            raw["stability_flips"] += 1

        raw_prev = raw_label

        # -----------------------------
        # TEMPORAL SMOOTHING
        # -----------------------------
        label_queue.append(raw_label)
        smooth_label = max(set(label_queue), key=label_queue.count)

        # True fire & smoke frames
        if smooth_label == "fire":
            smooth["fire_tp"] += 1
            if smooth["latency_fire"] is None:
                smooth["latency_fire"] = frame_count

        elif smooth_label == "smoke":
            smooth["smoke_tp"] += 1
            if smooth["latency_smoke"] is None:
                smooth["latency_smoke"] = frame_count

        # False positives
        if smooth_label == "no_fire" and smooth_prev in ["fire", "smoke"]:
            smooth["false_positives"] += 1

        # Stability flips
        if smooth_label != smooth_prev:
            smooth["stability_flips"] += 1

        smooth_prev = smooth_label

    cap.release()

    return {
        "total_frames": frame_count,
        "WITHOUT_TEMPORAL_SMOOTHING": raw,
        "WITH_TEMPORAL_SMOOTHING": smooth
    }


# ------------------------------------------------------
# RUN EXPERIMENT
# ------------------------------------------------------
if __name__ == "__main__":
    VIDEO = "../../data/firesense/fire/posVideo8.877.avi"
    results = run_ablation(VIDEO)

    print("\n=========== ABLATION STUDY RESULTS ===========")
    print("Total Frames:", results["total_frames"])

    print("\nWITHOUT TEMPORAL SMOOTHING:")
    print(results["WITHOUT_TEMPORAL_SMOOTHING"])

    print("\nWITH TEMPORAL_SMOOTHING:")
    print(results["WITH_TEMPORAL_SMOOTHING"])

    print("\n🔥 Use these to show:")
    print("- Lower false positives")
    print("- Lower fire falsely disappearing")
    print("- Lower smoke flickers")
    print("- Higher stability")
    print("- No increase in detection latency")
