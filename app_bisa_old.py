import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import torch
import os
import requests
from flask_cors import CORS
from ultralytics import YOLO
import base64
import json
import io
from PIL import Image
import math


app = Flask(__name__, static_folder="./templates/static")
CORS(app)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, async_mode="eventlet")

# Function to load YOLOv8 model based on the provided model name
def load_model(model_name):
    model_folder = "models"
    model_path = os.path.join(model_folder, model_name)
    try:
        return YOLO(model_path)
    except Exception as e:
        print("Error loading model:", e)
        raise

# Model name
model_name = "last_100epochs.pt"

print("Mapped model name:", model_name)

# Load initial YOLOv8 model
try:
    model = load_model(model_name)
except Exception as e:
    print("Failed to load model:", e)
    model = None  # Handle this appropriately in your application

def base64_to_image(base64_string):
    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

# Function to convert image to Base64
def img_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

# def process_results(results):
    # image = results.orig_img  # Get the original image

    # for box in results.boxes:
    #     start_point = (int(box[0]), int(box[1]))
    #     end_point = (int(box[2]), int(box[3]))
    #     color = (0, 255, 0)  # Green color
    #     thickness = 2
    #     image = cv2.rectangle(image, start_point, end_point, color, thickness)

    # _, frame_encoded = cv2.imencode(".jpg", image)
    # processed_img_data = base64.b64encode(frame_encoded).decode()
    # b64_src = "data:image/jpg;base64,"
    # processed_img_data = b64_src + processed_img_data

    # return processed_img_data

def process_results(results, img):
    print("=================== START: Processing results ====================")
    print("Results:", results[0])
    # print("Number of detected objects:", len(results[0]))
    # print("BOXES:", results[0].boxes)
    if len(results[0]) == 0:
        print("No objects detected")
        image = Image.fromarray(results[0].orig_img)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")

        # Encode the bytes object to a base64 string
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # The base64 string can be used as follows:
        base64_img = f"data:image/jpeg;base64,{img_str}"

        # # Print the base64 string (for verification)
        # print(base64_img)

        return base64_img
    else:
        print("Objects detected: 1")
        # r = results[0]
        classNames = ["potholes", "no_potholes"]
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(x1, y1, x2, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Convert the image to a base64 string
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        base64_img = f"data:image/jpeg;base64,{img_str}"

        print("Processed image data:", base64_img)
        return base64_img


    print("=================== END: Processing results ====================")
    
    return None


    # if len(results[0]) == 0:
    #     return None
    # Get bounding boxes and class labels
    # boxes = results.xyxy[0].cpu().numpy()
    # class_labels = results.names

    # # Create a list to store the detected objects
    # detected_objects = []

    # # Loop through the detected objects
    # for i, box in enumerate(boxes):
    #     x1, y1, x2, y2, conf, cls_conf, cls_pred = box
    #     class_label = class_labels[int(cls_pred)]
    #     detected_objects.append({
    #       "class": class_label,
    #       "confidence": float(conf),
    #       "bbox": [int(x1), int(y1), int(x2), int(y2)]
    #     })

    # detected_objects_json = json.dumps(detected_objects)
    # detected_objects_base64 = base64.b64encode(detected_objects_json.encode()).decode()

    # return detected_objects_base64

@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})

@socketio.on("image")
def receive_image(data):
    # print("Received image")

    global model

    image = base64_to_image(data)
    if model is not None:
        results = model(image)  # Use YOLOv8 for object detection
        # print("Processed results:", results)
        processed_img_data = process_results(results, image)
        emit("processed_image", processed_img_data)
    else:
        print("Model is not loaded, skipping image processing")

@app.route("/")
def index():
    return render_template("index.html", model_name=model_name)

if __name__ == "__main__":
    socketio.run(app, debug=False, port=5000, host='0.0.0.0')
