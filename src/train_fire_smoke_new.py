import torch
import multiprocessing
from ultralytics import YOLO

def main():
    print("=========== SYSTEM INFO ===========")
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))

    print("\n=========== LOADING YOLOv11 ===========")
    model = YOLO("yolo11s.pt")   # base model for small GPU

    print("\n🔥 Starting Training...\n")

    results = model.train(
        data="../data/data.yaml",   # modify path if needed
        epochs=50,
        imgsz=640,
        batch=4,           # ✔ safe for 6GB VRAM
        workers=0,         # ✔ important for Windows stability
        device=0,          # use GPU
        name="fire_smoke_yolo11",
        pretrained=True,

        # ---- stability + speed optimizations ----
        mosaic=1.0,        # good augmentation
        mixup=0.1,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        optimizer="AdamW",
        lr0=0.001,

        # early stopping
        patience=20,

        # logging
        verbose=True,
        val=True,
        save=True
    )

    print("\n=========== TRAINING COMPLETE ===========")
    print("Results saved in: runs/detect/fire_smoke_yolo11")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
