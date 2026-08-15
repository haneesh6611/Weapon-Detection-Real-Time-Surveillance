from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Train the model using the dataset configuration.
# Make sure data.yaml points to your dataset.
model.train(
    data="data.yaml",
    epochs=150,
    imgsz=640,
    patience=20
)
