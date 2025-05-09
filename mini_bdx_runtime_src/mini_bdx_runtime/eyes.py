import RPi.GPIO as GPIO
import numpy as np
import time
from threading import Thread

LEFT_EYE_GPIO = 24
RIGHT_EYE_GPIO = 23


class Eyes:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(RIGHT_EYE_GPIO, GPIO.OUT)
        GPIO.setup(LEFT_EYE_GPIO, GPIO.OUT)

        GPIO.output(RIGHT_EYE_GPIO, GPIO.HIGH)
        GPIO.output(LEFT_EYE_GPIO, GPIO.HIGH)

        self.blink_duration = 0.1

        Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            GPIO.output(RIGHT_EYE_GPIO, GPIO.LOW)
            GPIO.output(LEFT_EYE_GPIO, GPIO.LOW)
            time.sleep(self.blink_duration)
            GPIO.output(RIGHT_EYE_GPIO, GPIO.HIGH)
            GPIO.output(LEFT_EYE_GPIO, GPIO.HIGH)

            next_blink = np.random.rand() * 4  # seconds

            time.sleep(next_blink)


if __name__ == "__main__":
	e = Eyes()
	while True:
		time.sleep(1)
