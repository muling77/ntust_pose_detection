from ultralytics import YOLO
import torch

def main():
    # Optional: makes DataLoader behavior more stable on Windows
    # torch.multiprocessing.set_start_method("spawn", force=True)

    model = YOLO("yolo11n-pose.pt")
    results = model.train(
        data="hand-keypoints.yaml",  #MCU:要先執行convert_cmu_hand_to_yolo_pose.py轉成data.yaml
        epochs=100,
        imgsz=640,
        batch=24,
        lr0=0.001,
        device=0,
        workers=16,
        amp=True,   
        patience=0,      # early stopping
        close_mosaic = 0
    )

if __name__ == "__main__":
    main()
