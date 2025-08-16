import time

from .led_controller import get_controller


class Projector:
    def __init__(self):
        self.ctrl = get_controller()
        self.on = False
        self.ctrl.set_projector(False)

    def switch(self):
        self.on = not self.on
        self.ctrl.set_projector(self.on)

    def stop(self):
        self.on = False
        self.ctrl.set_projector(False)


if __name__ == "__main__":
    p = Projector()
    try:
        while True:
            p.switch()
            time.sleep(1)
    finally:
        p.stop()
