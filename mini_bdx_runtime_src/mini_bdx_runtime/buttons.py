import time


class Button:
    def __init__(self):
        self.last_pressed_time = time.time()
        self.timeout = 0.2
        self.is_pressed = False
        self.triggered = False
        self.released = True

    def update(self, value):
        if self.is_pressed and not value:
            self.released = True
        self.is_pressed = value
        if (
            self.released
            and self.is_pressed
            and (time.time() - self.last_pressed_time > self.timeout)
        ):
            self.triggered = True
            self.last_pressed_time = time.time()
        else:
            self.triggered = False

        if self.is_pressed:
            self.released = False


class Buttons:
    def __init__(self):
        self.A = Button()
        self.B = Button()
        self.X = Button()
        self.Y = Button()
        self.LB = Button()
        self.RB = Button()
        self.dpad_up = Button()
        self.dpad_down = Button()

    def update(self, A, B, X, Y, LB, RB, dpad_up, dpad_down):
        self.A.update(A)
        self.B.update(B)
        self.X.update(X)
        self.Y.update(Y)
        self.LB.update(LB)
        self.RB.update(RB)
        self.dpad_up.update(dpad_up)
        self.dpad_down.update(dpad_down)


if __name__ == "__main__":
    from mini_bdx_runtime.xbox_controller import XBoxController

    xbox_controller = XBoxController(30)
    buttons = Buttons()
    while True:

        (
            _,
            A_pressed,
            B_pressed,
            X_pressed,
            Y_pressed,
            LB_pressed,
            RB_pressed,
            _,
            _,
            up_down,
        ) = xbox_controller.get_last_command()

        buttons.update(
            A_pressed,
            B_pressed,
            X_pressed,
            Y_pressed,
            LB_pressed,
            RB_pressed,
            up_down == 1,
            up_down == -1,
        )

        print(buttons.dpad_up.triggered, buttons.dpad_up.is_pressed, buttons.dpad_down.triggered, buttons.dpad_down.is_pressed)

        time.sleep(0.05)
