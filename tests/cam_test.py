from picamzero import Camera
import cv2

print("Initializing camera ...")
cam = Camera()
# cam.still_size = (512, 512)
print("Camera initialized")

im = cam.capture_array()
im = cv2.resize(im, (512, 512))
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

cv2.imwrite("/home/bdxv2/aze.jpg", im)
# cv2.imshow("Image", im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cam.take_photo("/home/bdxv2/aze.jpg")
