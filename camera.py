
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# Load our trained model
model_path = Path(__file__).parent / "checkpoints" / "experiment_35epochs_best.pt"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = YOLO(str(model_path))

# Use Apple GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Open MacBook camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Cannot open camera. Check camera permissions.")

print("Camera started!")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        # Mirror the camera image
        frame = cv2.flip(frame, 1)

        # Detect hand gestures
        results = model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            device=device,
            verbose=False
        )

        # Draw boxes, labels and confidence
        annotated_frame = results[0].plot()

        cv2.imshow("Rock Paper Scissors - YOLO26n", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
