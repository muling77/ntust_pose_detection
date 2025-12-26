# infer_video_pose.py
# Use a trained YOLOv11 pose model (best.pt) to run inference on a video

import multiprocessing as mp
from pathlib import Path

def main() -> None:
    from ultralytics import YOLO

    # -----------------------
    # Paths (EDIT THESE)
    # -----------------------
    model_path = "runs/pose/train/weights/best.pt"                  # your trained model
    video_path = "vtest/test.mp4"          # your test video
    output_dir = "runs/pose/infer_video"    # output folder
    # -----------------------

    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        source=video_path,
        imgsz=640,
        conf=0.25,          # confidence threshold
        iou=0.7,
        device=0,           # GPU 0
        stream=False,       # video -> output video file
        save=True,          # save output video
        save_txt=False,     # no txt needed for demo
        save_conf=False,
        project=output_dir,
        name="exp",
        show=False,         # True = pop up window (not recommended on Windows)
        vid_stride=1        # >1 to skip frames if video is long
    )

    print("Inference finished.")
    print(f"Output saved to: {Path(output_dir).resolve()}")

if __name__ == "__main__":
    # Windows multiprocessing safety
    mp.freeze_support()
    main()
