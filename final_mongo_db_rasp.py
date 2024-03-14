import cv2
import numpy as np
from pymongo import MongoClient
from gridfs import GridFS

from picamera2 import Picamera2
from PIL import Image

from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('./best.pt')  # Replace with the actual path to your YOLOv8 model checkpoint
class_names = model.names

# Set up MongoDB connection
client = MongoClient('mongodb+srv://user:5uewbNA7iLteCcMp@blog-app.kh8tdua.mongodb.net/')
db = client['blog-app']
fs = GridFS(db)

# Initialize video capture from default camera
picam2 = Picamera2()
picam2.start()

def upload_video_to_mongodb(video_path, fs, metadata):
    # Open the video file
    with open(video_path, 'rb') as video_file:
        # Read the video file into memory
        video_bytes = video_file.read()

    # Store the video buffer in MongoDB
    fs.put(video_bytes, **metadata)
    print("Video uploaded to MongoDB")

while True:
    # Read frame from the camera
    img = picam2.capture_array()

    # Resize frame
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape

    # Convert NumPy array to PIL Image
    pil_img = Image.fromarray(img)

    # Convert RGBA image to RGB
    rgb_img = pil_img.convert('RGB')

    # Convert PIL Image back to NumPy array
    img_rgb = np.array(rgb_img)

    # Perform object detection
    results = model.predict(img_rgb)

    # Initialize variables for video recording
    recording = False
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

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
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Start recording if not already recording
                    if not recording:
                        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))
                        recording = True

                # If recording, write frame to the output video
                if recording:
                    out.write(img)

    # Upload recorded video to MongoDB
    if recording:
        out.release()

        # Upload video to MongoDB
        metadata = {
            'fieldname': 'file',
            'originalname': 'output_video.avi',
            'encoding': '7bit',
            'mimetype': 'video/avi'
        }
        upload_video_to_mongodb('output.avi', fs, metadata)

        # Delete temporary files
        recording = False
 
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
picam2.stop()
cv2.destroyAllWindows()
