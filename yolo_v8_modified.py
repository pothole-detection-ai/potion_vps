from ultralytics import YOLO
import numpy as np
import os
import cv2
import shutil

class YOLOmodified:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect(self, img):
        results = self.model.predict(img)
        result = results[0] 
        boxes = []
        masks = []
        confidences = []

        if result.masks is not None:
            for mask in result.masks.xy:
                segment = np.array(mask, dtype=np.int32)
                masks.append(segment)  

            mask = result.masks.data[0].cpu().numpy().astype("uint8")
            boxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
            confidences = result.boxes.conf.cpu().numpy()

        return boxes, masks, confidences

    def detect_full(self, img):
        frame_height, frame_width = img.shape[:2]
        boxes, masks, confidences = self.detect(img)
        save_dir_path = "outputs"
        os.makedirs(save_dir_path, exist_ok=True)
        output_data = {
            "total_objects": len(boxes),
            "save_dir_path": "",
            "objects": {}
        }

        img_with_masks = img.copy()
        if masks is not None:
            cv2.fillPoly(img_with_masks, masks, (0, 128, 255))
            segmented_img = cv2.addWeighted(img, 0.5, img_with_masks, 0.5, 0)

            text_y_position = 20  # Initial Y position for the top text
            text_x_position = 10  # Initial X position for the top text
            text_line_height = 20  # Line height for text
            max_columns = 4  # Maximum number of columns before wrapping to a new row

            for i, (box, mask, confidence) in enumerate(zip(boxes, masks, confidences)):
                color = colors[i % len(colors)]
                x, y, x2, y2 = box
                box_w, box_h = x2 - x, y2 - y

                pothole_label = f"Pothole {i + 1}"
                cv2.rectangle(segmented_img, (x, y), (x2, y2), color, 3)
                add_text_with_background(segmented_img, pothole_label, (x, y - 10), color=(255, 255, 255), font_scale=0.6, thickness=2, bg_color=(0, 0, 0), bg_opacity=0.65)

                width_text = f"W: {round(((box_w * 0.0264583333) / width_scale), 2)} cm"
                height_text = f"H: {round(((box_h * 0.0264583333) / height_scale), 2)} cm"
                distance_text = f"D: {round(distance(box_w, box_h), 2)} m"
                confidence_text = f"Conf: {round(confidence * 100, 2)}%"

                # Add text to the top, wrapping to a new row after max_columns
                current_column = i % max_columns
                current_row = i // max_columns
                text_x_position = 10 + (current_column * (frame_width // max_columns))

                add_text_with_background(segmented_img, pothole_label + ":", (text_x_position, text_y_position + current_row * 5 * text_line_height), color=(255, 255, 255), font_scale=0.5, thickness=2, bg_color=(0, 0, 0), bg_opacity=0.65)
                add_text_with_background(segmented_img, distance_text, (text_x_position, text_y_position + current_row * 5 * text_line_height + text_line_height), color=(255, 255, 255), font_scale=0.5, thickness=2, bg_color=(0, 0, 0), bg_opacity=0.65)
                add_text_with_background(segmented_img, width_text, (text_x_position, text_y_position + current_row * 5 * text_line_height + 2 * text_line_height), color=(255, 255, 255), font_scale=0.5, thickness=2, bg_color=(0, 0, 0), bg_opacity=0.65)
                add_text_with_background(segmented_img, height_text, (text_x_position, text_y_position + current_row * 5 * text_line_height + 3 * text_line_height), color=(255, 255, 255), font_scale=0.5, thickness=2, bg_color=(0, 0, 0), bg_opacity=0.65)
                add_text_with_background(segmented_img, confidence_text, (text_x_position, text_y_position + current_row * 5 * text_line_height + 4 * text_line_height), color=(255, 255, 255), font_scale=0.5, thickness=2, bg_color=(0, 0, 0), bg_opacity=0.65)

                distance_val = round(distance(box_w, box_h), 2)
                pothole_data = {
                    "distance": distance_val,
                    "width": round(((box_w * 0.0264583333) / width_scale), 2),
                    "length": round(((box_h * 0.0264583333) / height_scale), 2),
                    "confidence": round(confidence * 100, 2)
                }
                output_data["objects"][f"potholes_{i + 1}"] = pothole_data

            output_image_path = os.path.join(save_dir_path, "annotated_result.jpg")
            output_data["save_dir_path"] = output_image_path
            cv2.imwrite(output_image_path, segmented_img)

        print(output_data)
        return output_data


# Depth information
def distance(box_w, box_h):
    return (((2 * 3.14 * 180) / (box_w + box_h * 360) * 1000 + 9)) * 0.0254

# Function to add text with background to an image
def add_text_with_background(image, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1.1, bg_color=(0, 0, 0), bg_opacity=0.7):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x, text_y = position
    text_w, text_h = text_size
    text_y -= text_h  # Adjust text position to be above the box

    # Draw background rectangle with opacity
    overlay = image.copy()
    cv2.rectangle(overlay, (text_x, text_y), (text_x + text_w, text_y + text_h + 5), bg_color, -1)
    alpha = bg_opacity
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw text over the background
    cv2.putText(image, text, (text_x, text_y + text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Load model
model = YOLOmodified(r'models/sgd50v8.pt')

# Path input and output
folder_path = r"images"
output_folder = "outputs"
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
    print(f"Folder '{output_folder}' has been deleted successfully.")

# os.makedirs(output_folder, exist_ok=True)

# Scale
width_scale = 0.08558701578860999
height_scale = 0.04203788529514938

# Specified color list
colors = [(0, 0, 255), (255, 0, 0), (128, 0, 128), (255, 165, 0), (165, 42, 42)]  # Blue, Red, Dark Purple, Orange, Brown

def yolo_detect_image(image_path):
    image = cv2.imread(image_path)
    results = model.detect_full(image)
    return results

img = cv2.imread('images/road (8).jpg')
result = model.detect_full(img)

cv2.destroyAllWindows()
