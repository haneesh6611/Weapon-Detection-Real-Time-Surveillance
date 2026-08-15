from ultralytics import YOLO

CONF_THRESHOLD = 0.55

# Load trained model
model = YOLO("best.pt")

# Run detection on a frame
results = model(frame, conf=CONF_THRESHOLD)
