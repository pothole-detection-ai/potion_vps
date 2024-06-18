import helpers

def detection_result(dir_path, image_path):
    base64_image = helpers.jpg_to_base64(dir_path + "/" + image_path)
    # helpers.remove_dir(dir_path)
    return base64_image
