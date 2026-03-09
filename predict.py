from ultralytics import YOLO
import os

def main():
    # ---------------- SETTINGS ----------------
    model_path = "yolov8n.pt"     # change to best.pt if you have trained model
    source = "tested.mp4"         # image / folder / video / webcam (0)
    conf = 0.25
    iou = 0.45
    save_dir = "runs/predict"
    # ------------------------------------------

    # Load model
    model = YOLO(model_path)

    # Run prediction
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=True,
        project=save_dir,
        name="pothole",
        exist_ok=True
    )

    # Print detections
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[cls]
            print(f"Detected: {label} | Confidence: {confidence:.2f}")

    print("\n✅ Prediction completed!")
    print(f"📁 Results saved in: {save_dir}/pothole")


if __name__ == "__main__":
    main()
