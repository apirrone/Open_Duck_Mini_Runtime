import random
import time
from threading import Thread, Event

from open_duck_mini_runtime.led_controller import get_controller


class Eyes:
    def __init__(self, blink_duration=0.1, min_interval=1.0, max_interval=4.0):
        self.ctrl = get_controller()

        self.blink_duration = blink_duration
        self.min_interval = min_interval
        self.max_interval = max_interval

        # Ensure eyes start ON to mimic previous behavior

        self.ctrl.set_eyes_color("white")

        self.ctrl.set_eyes(True)

        self._stop_event = Event()
        self._thread = Thread(target=self.run, daemon=True)
        self._thread.start()

    def _set_eyes(self, state: bool):
        self.ctrl.set_eyes(state)

    def run(self):
        try:
            while not self._stop_event.is_set():
                self._set_eyes(False)
                if self._stop_event.wait(self.blink_duration):
                    break
                self._set_eyes(True)
                next_blink = random.uniform(self.min_interval, self.max_interval)
                if self._stop_event.wait(next_blink):
                    break
        except Exception as err:
            print(f"Error in eye thread: {err}")
            self._stop_event.set()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self._set_eyes(False)
        # deinit controller
        self.ctrl.deinit()


if __name__ == "__main__":
    e = Eyes()
    try:
        while True:
            time.sleep(1)
    finally:
        e.stop()
