from ultralytics import YOLO
from PIL import Image
import cv2
import helpers

# Load a model
# model = YOLO("models/yolov8n.pt")
# model = YOLO("yolov8n-seg.pt")
model = YOLO("models/last_100epochs.pt")

def detect_image(image_path):
    image = Image.open(image_path)
    results = model.predict(source=image, save=True)[0]  # save plotted images
    # resize font size of predicted image

    total_objects = len(results)
    save_dir_path = results.save_dir

    return {
        "total_objects": total_objects,
        "save_dir_path": save_dir_path,
    }

def detect_image2(url):
    # image = Image.open(image_path)
    results = model.predict(url, save=True)[0]  # save plotted images

    # print(type(results))
    # print(len(results))
    # print(results)
    # print(results.save_dir)

    total_objects = len(results)
    save_dir_path = results.save_dir

    return {
        "total_objects": total_objects,
        "save_dir_path": save_dir_path
    }



