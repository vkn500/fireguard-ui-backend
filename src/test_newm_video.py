from ultralytics import YOLO

model = YOLO("runs/detect/fire_smoke_yolo11/weights/best.pt")

model.predict(
    source="../data/firesense/fire/posVideo6.875.avi",  #../data/firesense/smoke/testpos08.824.avi
    conf=0.5,
    save=True,
    stream=False
)
