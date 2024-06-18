import base64
from PIL import Image
from io import BytesIO
import shutil
from PIL import Image
import pybase64
import os


def jpg_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        my_string = pybase64.b64encode(img_file.read())

    # convert bytes to string
    my_string = my_string.decode('utf-8')
    return my_string


def base64_to_jpg(base64_data, output_file):
    try:
        # Decode the base64 image data
        image_data = base64.b64decode(base64_data)
        print("Image data decoded successfully")
        print("Image data:", image_data)

        # Open the image using PIL
        img = Image.open(BytesIO(image_data))

        # Convert the image to RGB mode if it has an alpha channel
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Save the image as JPEG
        img.save(output_file, "JPEG")
        print("Image saved successfully as", output_file)
    except Exception as e:
        print("Error:", e)


def remove_dir(dir_path):
    try:
        shutil.rmtree(dir_path)
        print("Directory removed successfully:", dir_path)
    except Exception as e:
        print("Error:", e)