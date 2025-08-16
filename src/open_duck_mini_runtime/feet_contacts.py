import board
import digitalio
import time

LEFT_FOOT_PIN = board.D22
RIGHT_FOOT_PIN = board.D27

class FeetContacts:
    def __init__(self):
        self.left_foot = digitalio.DigitalInOut(LEFT_FOOT_PIN)
        self.left_foot.direction = digitalio.Direction.INPUT
        self.left_foot.pull = digitalio.Pull.UP

        self.right_foot = digitalio.DigitalInOut(RIGHT_FOOT_PIN)
        self.right_foot.direction = digitalio.Direction.INPUT
        self.right_foot.pull = digitalio.Pull.UP

    def get(self):
        left = not self.left_foot.value
        right = not self.right_foot.value
        return [left, right]

    def stop(self):
        self.left_foot.deinit()
        self.right_foot.deinit()

if __name__ == "__main__":
    feet_contacts = FeetContacts()
    try:
        while True:
            print(feet_contacts.get())
            time.sleep(0.05)
    finally:
        feet_contacts.stop()
