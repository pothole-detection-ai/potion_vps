import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import torch
import os
import requests
from flask_cors import CORS

app = Flask(__name__, static_folder="./templates/static")
CORS(app)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, async_mode="eventlet")

# Function to load YOLOv5 model based on the provided model name
def load_model(model_name):
    model_folder = "models"
    model_path = os.path.join(model_folder, model_name)
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Mapping of model_name values to YOLOv5 model names
MODEL_NAME_MAPPING = {
    "BEST_1": "yolov5s",
    "MODEL_2": "yolov5n",
    "MODEL_3": "yolov5m",
}

# Get initial model name from API
api_url = "https://guimobilerobotagvswarm.my.id/api/model"
response = requests.get(api_url)
initial_model_name = response.json().get("model_name")

# Map the initial model name to YOLOv5 model name
model_name = MODEL_NAME_MAPPING.get(initial_model_name, initial_model_name)



# Load initial YOLOv5 model
model = load_model(model_name)

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
def receive_image(data):
    print("Received image")

    # Declare model_name as a global variable
    global model_name
    global model

    # Get the updated model name from the API
    updated_model_name = requests.get(api_url).json().get("model_name")

    # Map the updated model name to YOLOv5 model name
    updated_model_name = MODEL_NAME_MAPPING.get(updated_model_name, updated_model_name)

    # Reload the model if the model name has changed
    if updated_model_name != model_name:
        model_name = updated_model_name
        model = load_model(model_name)
        print("Model updated to:", model_name)

    image = base64_to_image(data)
    results = model(image)  # Use YOLOv5 for object detection
    print("===========", results)
    processed_img_data = process_results(results)
    emit("processed_image", processed_img_data)

@app.route("/")
def index():
    return render_template("index.html", model_name=model_name)

if __name__ == "__main__":
    socketio.run(app, debug=False, port=5000, host='0.0.0.0')
