import RPi.GPIO as GPIO
import numpy as np


LEFT_FOOT_PIN = 22
RIGHT_FOOT_PIN = 27


class FeetContacts:
    def __init__(self):
        GPIO.setwarnings(False)  # Ignore warning for now
        GPIO.setmode(GPIO.BCM)  # Use physical pin numbering
        GPIO.setup(LEFT_FOOT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(RIGHT_FOOT_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def get(self):
        left = False
        right = False
        if GPIO.input(LEFT_FOOT_PIN) == GPIO.LOW:
            left = True
        if GPIO.input(RIGHT_FOOT_PIN) == GPIO.LOW:
            right = True
        return np.array([left, right])


if __name__ == "__main__":
    import time

    feet_contacts = FeetContacts()
    while True:
        print(feet_contacts.get())
        time.sleep(0.05)
