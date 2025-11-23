# logger.py
import csv
import os

LOG_FILE = "../logs/detection_log.csv"

def init_logger():
    os.makedirs("../logs", exist_ok=True)
    # Create file with header if not exists
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Label", "Severity", "FireAreaRatio", "SavedImage"])

def log_event(timestamp, label, severity, faratio, imagename):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, label, severity, f"{faratio:.4f}", imagename])
