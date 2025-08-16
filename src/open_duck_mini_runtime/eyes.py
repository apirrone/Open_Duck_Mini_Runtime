import board
import digitalio
import random
import time
from threading import Thread, Event

LEFT_EYE_PIN = board.D24
RIGHT_EYE_PIN = board.D23


class Eyes:
    def __init__(self, blink_duration=0.1, min_interval=1.0, max_interval=4.0):
        self.left_eye = digitalio.DigitalInOut(LEFT_EYE_PIN)
        self.left_eye.direction = digitalio.Direction.OUTPUT

        self.right_eye = digitalio.DigitalInOut(RIGHT_EYE_PIN)
        self.right_eye.direction = digitalio.Direction.OUTPUT

        self.blink_duration = blink_duration
        self.min_interval = min_interval
        self.max_interval = max_interval

        self._stop_event = Event()
        self._thread = Thread(target=self.run, daemon=True)
        self._thread.start()

    def _set_eyes(self, state):
        self.left_eye.value = state
        self.right_eye.value = state

    def run(self):
        try:
            while not self._stop_event.is_set():
                self._set_eyes(False)
                time.sleep(self.blink_duration)
                self._set_eyes(True)
                next_blink = random.uniform(self.min_interval, self.max_interval)
                time.sleep(next_blink)
        except Exception as err:
            print(f"Error in eye thread: {err}")
            self._stop_event.set()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self._set_eyes(False)
        self.left_eye.deinit()
        self.right_eye.deinit()


if __name__ == "__main__":
    e = Eyes()
    try:
        while True:
            time.sleep(1)
    finally:
        e.stop()

