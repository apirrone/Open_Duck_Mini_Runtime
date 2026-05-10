import logging
import random
import time
from threading import Thread, Event

from open_duck_mini_runtime.hardware.led_controller import get_controller

logger = logging.getLogger(__name__)


class Eyes:
    def __init__(self, blink_duration=0.1, min_interval=1.0, max_interval=4.0):
        self.ctrl = get_controller()

        self.blink_duration = blink_duration
        self.min_interval = min_interval
        self.max_interval = max_interval

        self._solid = False  # when True, blink thread holds eyes on

        self.ctrl.set_eyes_color("white")
        self.ctrl.set_eyes(True)

        self._stop_event = Event()
        self._thread = Thread(target=self.run, daemon=True)
        self._thread.start()

    def _set_eyes(self, state: bool):
        self.ctrl.set_eyes(state)

    def set_color(self, color: str) -> None:
        """Change eye color without stopping the blink thread."""
        self.ctrl.set_eyes_color(color)
        self.ctrl.set_eyes(True)

    def set_solid(self, solid: bool) -> None:
        """When solid=True, suppress blinking (eyes stay on)."""
        self._solid = solid
        if solid:
            self.ctrl.set_eyes(True)

    def run(self):
        try:
            while not self._stop_event.is_set():
                if self._solid:
                    self._stop_event.wait(0.1)
                    continue
                self._set_eyes(False)
                if self._stop_event.wait(self.blink_duration):
                    break
                self._set_eyes(True)
                next_blink = random.uniform(self.min_interval, self.max_interval)
                if self._stop_event.wait(next_blink):
                    break
        except Exception as err:
            logger.error("Eye thread error: %s", err)
            self._stop_event.set()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self._set_eyes(False)
        self.ctrl.deinit()


if __name__ == "__main__":
    e = Eyes()
    try:
        while True:
            time.sleep(1)
    finally:
        e.stop()
