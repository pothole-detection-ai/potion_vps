import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import base64
import torch
import os
from flask_cors import CORS
from ultralytics import YOLO
import json
import io
from PIL import Image
import math
import yolo_v8
import helpers
import custom

app = Flask(__name__, static_folder="./templates/static")
CORS(app)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="https://vps.potion.my.id")

def load_model(model_name):
    model_folder = "/root/potion/models"
    model_path = os.path.join(model_folder, model_name)
    try:
        return YOLO(model_path)
    except Exception as e:
        print("Error loading model:", e)
        raise

model_name = "last_100epochs.pt"
print("Mapped model name:", model_name)
FOLDER_PATH = "images"


try:
    model = load_model(model_name)
except Exception as e:
    print("Failed to load model:", e)
    model = None

def base64_to_image(base64_string):
    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def img_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def process_results(results, img):
    print("=================== START: Processing results ====================")
    print("Results:", results[0])
    
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
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    base64_img = f"data:image/jpeg;base64,{img_str}"
    print("Processed image data:", base64_img)
    return base64_img
        

@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})

@socketio.on("image")
def receive_image(data):
    global model
    image = base64_to_image(data)
    if model is not None:
        results = model(image)
        processed_img_data = process_results(results, image)
        emit("processed_image", processed_img_data)
    else:
        print("Model is not loaded, skipping image processing")

@app.route("/")
def index():
    return render_template("index.html", model_name=model_name)

# create route for post request and the request data contains base64 image
@app.route('/detect', methods=['POST'])
def detect():
    if 'base64_image_string' in request.json and request.json['base64_image_string']:
        # get the base64 image from the request data
        base64_image_string = request.json['base64_image_string']

        print("base64_image_string:", base64_image_string)

        # convert the base64 image to jpg
        helpers.base64_to_jpg(base64_image_string, FOLDER_PATH +  "/uploaded_object.jpg")
        
        print("Image saved successfully as uploaded_object.jpg")

        # # detect the image
        results = yolo_v8.detect_image(FOLDER_PATH + "/uploaded_object.jpg")
        base64_result = custom.detection_result(results["save_dir_path"], "uploaded_object.jpg")
        return jsonify({
            "base64_image_string": base64_result,
            "total_objects": results["total_objects"],
        })

    return jsonify({
        "error": "parameter base64_image_string tidak boleh kosong!"
    })

if __name__ == "__main__":
    socketio.run(app, debug=False, port=5000, host='0.0.0.0')
