# Weapon Detection for Real-Time Surveillance

Major project: **Weapon Detection for Real-Time Surveillance**

The project uses YOLOv8, Python, OpenCV and PyTorch to detect weapons from images/video and process a webcam stream in real time.

## Project features

- YOLOv8-based weapon detection
- Real-time webcam/video processing
- Bounding boxes and confidence scores
- Screenshot capture when a detection is confirmed
- Detection-event video recording
- Monitoring status and alert count

## Repository structure

```text
Weapon-Detection-Real-Time-Surveillance/
├── README.md
├── requirements.txt
├── project_report.docx
├── src/
│   ├── train.py
│   ├── detect.py
│   └── realtime_detection.py
├── models/
├── data/
├── screenshots/
└── recordings/
```

## Software mentioned in the project report

- Python 3.8+
- YOLOv8 / Ultralytics
- PyTorch
- OpenCV
- NumPy
- Pandas
- VS Code / Jupyter Notebook

## Setup

```bash
pip install -r requirements.txt
```

Place the trained `best.pt` model at the path expected by the real-time detection script, or update `MODEL_PATH` in `src/realtime_detection.py`.

For training, place your YOLO-format dataset configuration (`data.yaml`) where `src/train.py` can access it.

## Run real-time detection

```bash
python src/realtime_detection.py
```

Press `q` to exit the camera window.

## Important note

The submitted project report contains the project documentation and sample code. The repository separates the sample code into Python files so that the project is easier to understand and present on GitHub.

The dataset and trained model weights are not included in this package because the submitted document describes them but does not contain the actual dataset files or `best.pt` binary model.
