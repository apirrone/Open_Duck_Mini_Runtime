import RPi.GPIO as GPIO
import time

PROJECTOR_GPIO = 25

class Projector:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(PROJECTOR_GPIO, GPIO.OUT)

        GPIO.output(PROJECTOR_GPIO, GPIO.LOW)
        self.on = False

    def switch(self):
        self.on = not self.on

        if self.on:
            GPIO.output(PROJECTOR_GPIO, GPIO.HIGH)
        else:
            GPIO.output(PROJECTOR_GPIO, GPIO.LOW)


if __name__ == "__main__":
    p = Projector()
    while True:

        p.switch()
        time.sleep(1)
