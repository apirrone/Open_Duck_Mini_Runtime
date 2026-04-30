import pygame
from threading import Thread
from queue import Queue
import time
import numpy as np
from open_duck_mini_runtime.buttons import Buttons

X_RANGE = [-0.15, 0.15]
Y_RANGE = [-0.2, 0.2]
YAW_RANGE = [-1.0, 1.0]

# rads
NECK_PITCH_RANGE = [-0.34, 1.1]
HEAD_PITCH_RANGE = [-0.78, 0.3]
HEAD_YAW_RANGE = [-0.5, 0.5]
HEAD_ROLL_RANGE = [-0.5, 0.5]


class XBoxController:
    def __init__(self, command_freq, only_head_control=False):
        self.command_freq = command_freq
        self.head_control_mode = only_head_control
        self.only_head_control = only_head_control

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_left_trigger = 0.0
        self.last_right_trigger = 0.0
        self.connected = False
        self.p1 = None
        pygame.init()
        self._try_init_joystick()
        self.cmd_queue = Queue(maxsize=1)

        self.A_pressed = False
        self.B_pressed = False
        self.X_pressed = False
        self.Y_pressed = False
        self.LB_pressed = False
        self.RB_pressed = False
        self.start_pressed = False

        self.buttons = Buttons()

        Thread(target=self.commands_worker, daemon=True).start()

    def _try_init_joystick(self) -> bool:
        pygame.joystick.quit()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("[xbox] No joystick detected")
            self.connected = False
            self.p1 = None
            return False
        self.p1 = pygame.joystick.Joystick(0)
        self.p1.init()
        print(f"[xbox] Connected: {self.p1.get_name()} ({self.p1.get_numaxes()} axes)")
        self.connected = True
        return True

    def try_reconnect(self) -> bool:
        return self._try_init_joystick()

    def commands_worker(self):
        while True:
            if self.connected:
                try:
                    self.cmd_queue.put(self.get_commands())
                except Exception as e:
                    print(f"[xbox] Controller error: {e} — disconnected")
                    self.connected = False
                    self.p1 = None
            time.sleep(1 / self.command_freq)

    def get_commands(self):
        if not self.connected or self.p1 is None:
            return (
                np.around(self.last_commands, 3),
                False, False, False, False, False, False, False,
                0.0, 0.0, 0,
            )

        last_commands = self.last_commands
        left_trigger = self.last_left_trigger
        right_trigger = self.last_right_trigger

        l_x = -1 * self.p1.get_axis(0)
        l_y = -1 * self.p1.get_axis(1)
        r_x = -1 * self.p1.get_axis(2)
        r_y = -1 * self.p1.get_axis(3)

        right_trigger = np.around((self.p1.get_axis(4) + 1) / 2, 3)
        left_trigger = np.around((self.p1.get_axis(5) + 1) / 2, 3)

        if left_trigger < 0.1:
            left_trigger = 0
        if right_trigger < 0.1:
            right_trigger = 0

        if not self.head_control_mode:
            lin_vel_y = l_x
            lin_vel_x = l_y
            ang_vel = r_x
            if lin_vel_x >= 0:
                lin_vel_x *= np.abs(X_RANGE[1])
            else:
                lin_vel_x *= np.abs(X_RANGE[0])

            if lin_vel_y >= 0:
                lin_vel_y *= np.abs(Y_RANGE[1])
            else:
                lin_vel_y *= np.abs(Y_RANGE[0])

            if ang_vel >= 0:
                ang_vel *= np.abs(YAW_RANGE[1])
            else:
                ang_vel *= np.abs(YAW_RANGE[0])

            last_commands[0] = lin_vel_x
            last_commands[1] = lin_vel_y
            last_commands[2] = ang_vel
        else:
            last_commands[0] = 0.0
            last_commands[1] = 0.0
            last_commands[2] = 0.0
            last_commands[3] = 0.0  # neck pitch 0 for now

            head_yaw = l_x
            head_pitch = l_y
            head_roll = r_x

            if head_yaw >= 0:
                head_yaw *= np.abs(HEAD_YAW_RANGE[0])
            else:
                head_yaw *= np.abs(HEAD_YAW_RANGE[1])

            if head_pitch >= 0:
                head_pitch *= np.abs(HEAD_PITCH_RANGE[0])
            else:
                head_pitch *= np.abs(HEAD_PITCH_RANGE[1])

            if head_roll >= 0:
                head_roll *= np.abs(HEAD_ROLL_RANGE[0])
            else:
                head_roll *= np.abs(HEAD_ROLL_RANGE[1])

            last_commands[4] = head_pitch
            last_commands[5] = head_yaw
            last_commands[6] = head_roll

        n_buttons = self.p1.get_numbuttons()

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:

                if n_buttons > 0 and self.p1.get_button(0):  # A button
                    self.A_pressed = True

                if n_buttons > 1 and self.p1.get_button(1):  # B button
                    self.B_pressed = True

                if n_buttons > 3 and self.p1.get_button(3):  # X button
                    self.X_pressed = True

                if n_buttons > 4 and self.p1.get_button(4):  # Y button
                    self.Y_pressed = True
                    if not self.only_head_control:
                        self.head_control_mode = not self.head_control_mode

                if n_buttons > 6 and self.p1.get_button(6):  # LB button
                    self.LB_pressed = True

                if n_buttons > 7 and self.p1.get_button(7):  # RB button
                    self.RB_pressed = True

                if n_buttons > 11 and self.p1.get_button(11):  # Start / Menu button
                    self.start_pressed = True

            if event.type == pygame.JOYBUTTONUP:
                self.A_pressed = False
                self.B_pressed = False
                self.X_pressed = False
                self.Y_pressed = False
                self.LB_pressed = False
                self.RB_pressed = False
                self.start_pressed = False

        up_down = self.p1.get_hat(0)[1] if self.p1.get_numhats() > 0 else 0
        pygame.event.pump()  # process event queue

        return (
            np.around(last_commands, 3),
            self.A_pressed,
            self.B_pressed,
            self.X_pressed,
            self.Y_pressed,
            self.LB_pressed,
            self.RB_pressed,
            self.start_pressed,
            left_trigger,
            right_trigger,
            up_down,
        )

    def get_last_command(self):
        A_pressed = False
        B_pressed = False
        X_pressed = False
        Y_pressed = False
        LB_pressed = False
        RB_pressed = False
        start_pressed = False
        up_down = 0
        try:
            (
                self.last_commands,
                A_pressed,
                B_pressed,
                X_pressed,
                Y_pressed,
                LB_pressed,
                RB_pressed,
                start_pressed,
                self.last_left_trigger,
                self.last_right_trigger,
                up_down,
            ) = self.cmd_queue.get(
                False
            )  # non blocking
        except Exception:
            pass

        self.buttons.update(
            A_pressed,
            B_pressed,
            X_pressed,
            Y_pressed,
            LB_pressed,
            RB_pressed,
            up_down == 1,
            up_down == -1,
            start=start_pressed,
        )

        return (
            self.last_commands,
            self.buttons,
            self.last_left_trigger,
            self.last_right_trigger,
        )


if __name__ == "__main__":
    controller = XBoxController(20)

    while True:
        print(controller.get_last_command())
        time.sleep(0.05)
