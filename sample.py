from ultralytics import YOLO
import cv2
import numpy as np
import cvzone
import time
from picamera2 import Picamera2, Preview

# Initialize Picamera
picam = Picamera2()
config = picam.create_preview_configuration()
picam.configure(config)
picam.start_preview(Preview.QTGL)
picam.start()
time.sleep(2)

# Load YOLO model
model = YOLO("./best.pt")
class_names = model.names

while True:
    # Capture frame from Picamera
    picam.capture_file("test-python.jpg")
    img = cv2.imread("test-python.jpg")

    # Resize frame
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape

    # Perform object detection
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        
    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)
                cv2.polylines(img, [contour],True, color=(0, 0, 255), thickness=2)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
    cv2.imshow('img', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Release resources
picam.close()
cv2.destroyAllWindows()
