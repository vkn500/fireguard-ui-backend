from ultralytics import YOLO
model = YOLO("../models/yolo11-d-fire-dataset.pt")
print(model.names)
