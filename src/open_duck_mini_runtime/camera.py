from picamzero import Camera
import cv2
import base64
import os

class Cam:
    def __init__(self):
        self.cam = Camera()

    def get_encoded_image(self):
        im = self.cam.capture_array()
        im = cv2.resize(im, (512, 512))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

        cv2.imwrite("/home/bdxv2/aze.jpg", im)

        return self.encode_image("/home/bdxv2/aze.jpg")



    # def encode_image(self, image):
    #     return base64.b64encode(image).decode("utf-8")

    def encode_image(self, image_path: str):
        # check if the image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
