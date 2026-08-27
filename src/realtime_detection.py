import cv2
import os
import datetime
import time
from ultralytics import YOLO

# ==============================
# MODEL
# ==============================
MODEL_PATH = "weapon_debest.pt"
model = YOLO(MODEL_PATH)

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
HEADER_HEIGHT = 50

# ==============================
# STORAGE
# ==============================
os.makedirs("screenshots", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

# ==============================
# DETECTION SETTINGS
# ==============================
CONF_THRESHOLD = 0.55
MIN_BOX_AREA = 15000
CONFIRM_FRAMES = 2

# ==============================
# VARIABLES
# ==============================
weapon_frames = 0
alert_count = 0
screenshot_taken = False
recording = False
video_writer = None

fourcc = cv2.VideoWriter_fourcc(*"XVID")
start_time = time.time()
window_name = "CCTV WEAPON DETECTION"

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    weapon_detected = False
    results = model(frame, conf=CONF_THRESHOLD)

    if results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            area = w * h

            if area < MIN_BOX_AREA:
                continue

            if y1 < HEADER_HEIGHT:
                y1 = HEADER_HEIGHT + 5

            weapon_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"WEAPON {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    # ==============================
    # CONFIRMATION
    # ==============================
    if weapon_detected:
        weapon_frames += 1
    else:
        weapon_frames = 0

    confirmed_weapon = weapon_frames >= CONFIRM_FRAMES

    # ==============================
    # SCREENSHOT
    # ==============================
    if confirmed_weapon and not screenshot_taken:
        filename = (
            f"screenshots/weapon_"
            f"{datetime.datetime.now().strftime('%H%M%S')}.jpg"
        )
        cv2.imwrite(filename, frame)
        screenshot_taken = True
        alert_count += 1

    if not confirmed_weapon:
        screenshot_taken = False

    # ==============================
    # RECORDING
    # ==============================
    if confirmed_weapon and not recording:
        video_path = (
            f"recordings/detection_"
            f"{datetime.datetime.now().strftime('%H%M%S')}.avi"
        )
        video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            20,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )
        recording = True

    if recording and video_writer is not None:
        video_writer.write(frame)

    # ==============================
    # HEADER BAR
    # ==============================
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (FRAME_WIDTH, HEADER_HEIGHT),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    status = "WEAPON DETECTED" if confirmed_weapon else "MONITORING"
    color = (0, 0, 255) if confirmed_weapon else (0, 255, 0)

    elapsed = int(time.time() - start_time)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    y = 30

    status_x = int(FRAME_WIDTH * 0.05)
    alerts_x = int(FRAME_WIDTH * 0.45)
    time_x = int(FRAME_WIDTH * 0.75)

    cv2.putText(
        frame, f"STATUS: {status}",
        (status_x, y), font, scale, color, thickness
    )
    cv2.putText(
        frame, f"ALERTS: {alert_count}",
        (alerts_x, y), font, scale, (255, 255, 255), thickness
    )
    cv2.putText(
        frame, f"TIME: {elapsed}s",
        (time_x, y), font, scale, (255, 255, 255), thickness
    )

    # ==============================
    # DISPLAY
    # ==============================
    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

# ==============================
# CLEANUP
# ==============================
cap.release()

if video_writer is not None:
    video_writer.release()

cv2.destroyAllWindows()
