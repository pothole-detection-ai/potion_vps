import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO("yolov8n-new.pt")

# Load an image
img_1 = Image.open("image-yes.jpg")
img_2 = Image.open("image-no.jpg")

# Perform object detection
results_1 = model.predict(source=img_1)
results_2 = model.predict(source=img_2)

# Print the number of detected objects
print(f"Number of detected objects: {len(results_1[0])}")
print("===================================================== START RESULTS_1 =====================================================")
print(results_1[0].boxes.xyxy)
print("===================================================== END RESULTS_1 =====================================================")
print(f"Number of detected objects: {len(results_2[0])}")
print("===================================================== START RESULTS_2 =====================================================")
print(results_2[0].boxes.xyxy)
print("===================================================== END RESULTS_2 =====================================================")