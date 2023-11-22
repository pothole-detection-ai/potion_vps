import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True) 

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        results = model(image)
        a = np.squeeze(results.render())
        
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()