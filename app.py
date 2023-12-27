import cv2
import numpy as np
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import base64
import torch
import os
from camera import VideoCamera  # Import your VideoCamera class
from flask_cors import CORS

app = Flask(__name__, static_folder="./templates/static")
CORS(app)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, async_mode="eventlet")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

def base64_to_image(base64_string):
    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def process_results(results):
    image = results.render()[0]  # Assuming a single image is processed
    for det in results.xyxy[0]:
        start_point = (int(det[0]), int(det[1]))
        end_point = (int(det[2]), int(det[3]))
        color = (0, 255, 0)  # Green color
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    _, frame_encoded = cv2.imencode(".jpg", image)
    processed_img_data = base64.b64encode(frame_encoded).decode()
    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data

    return processed_img_data


@socketio.on("connect")
def test_connect():
    print("Connected")
    emit("my response", {"data": "Connected"})

@socketio.on("image")
def receive_image(image):
    print("Received image")
    image = base64_to_image(image)
    results = model(image)  # Use YOLOv5 for object detection
    print("===========",results)
    processed_img_data = process_results(results)
    emit("processed_image", processed_img_data)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    socketio.run(app, debug=False, port=5000, host='0.0.0.0')
