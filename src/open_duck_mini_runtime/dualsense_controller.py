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


class DualSenseController:
    def __init__(self, command_freq, only_head_control=False):
        self.command_freq = command_freq
        self.head_control_mode = only_head_control
        self.only_head_control = only_head_control

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_left_trigger = 0.0
        self.last_right_trigger = 0.0
        pygame.init()
        self.p1 = pygame.joystick.Joystick(0)
        self.p1.init()
        print(f"Loaded joystick with {self.p1.get_numaxes()} axes.")
        self.cmd_queue = Queue(maxsize=1)

        self.cross_pressed = False
        self.circle_pressed = False
        self.square_pressed = False
        self.triangle_pressed = False
        self.l1_pressed = False
        self.r1_pressed = False

        self.buttons = Buttons()

        Thread(target=self.commands_worker, daemon=True).start()

    def commands_worker(self):
        while True:
            self.cmd_queue.put(self.get_commands())
            time.sleep(1 / self.command_freq)

    def get_commands(self):
        last_commands = self.last_commands
        left_trigger = self.last_left_trigger
        right_trigger = self.last_right_trigger

        l_x = -1 * self.p1.get_axis(0)
        l_y = -1 * self.p1.get_axis(1)
        r_x = -1 * self.p1.get_axis(2)
        r_y = -1 * self.p1.get_axis(3)

        right_trigger = np.around((self.p1.get_axis(5) + 1) / 2, 3)
        left_trigger = np.around((self.p1.get_axis(4) + 1) / 2, 3)

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

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:

                if self.p1.get_button(0):  # Cross button
                    self.cross_pressed = True

                if self.p1.get_button(1):  # Circle button
                    self.circle_pressed = True

                if self.p1.get_button(2):  # Square button
                    self.square_pressed = True

                if self.p1.get_button(3):  # Triangle button
                    self.triangle_pressed = True
                    if not self.only_head_control:
                        self.head_control_mode = not self.head_control_mode

                if self.p1.get_button(4):  # L1 button
                    self.l1_pressed = True

                if self.p1.get_button(5):  # R1 button
                    self.r1_pressed = True

            if event.type == pygame.JOYBUTTONUP:
                self.cross_pressed = False
                self.circle_pressed = False
                self.square_pressed = False
                self.triangle_pressed = False
                self.l1_pressed = False
                self.r1_pressed = False

        up_down = self.p1.get_hat(0)[1]
        pygame.event.pump()  # process event queue

        return (
            np.around(last_commands, 3),
            self.cross_pressed,
            self.circle_pressed,
            self.square_pressed,
            self.triangle_pressed,
            self.l1_pressed,
            self.r1_pressed,
            left_trigger,
            right_trigger,
            up_down,
        )

    def get_last_command(self):
        cross_pressed = False
        circle_pressed = False
        square_pressed = False
        triangle_pressed = False
        l1_pressed = False
        r1_pressed = False
        up_down = 0
        try:
            (
                self.last_commands,
                cross_pressed,
                circle_pressed,
                square_pressed,
                triangle_pressed,
                l1_pressed,
                r1_pressed,
                self.last_left_trigger,
                self.last_right_trigger,
                up_down,
            ) = self.cmd_queue.get(
                False
            )  # non blocking
        except Exception:
            pass

        self.buttons.update(
            cross_pressed,
            circle_pressed,
            square_pressed,
            triangle_pressed,
            l1_pressed,
            r1_pressed,
            up_down == 1,
            up_down == -1,
        )

        return (
            self.last_commands,
            self.buttons,
            self.last_left_trigger,
            self.last_right_trigger,
        )

if __name__ == "__main__":
    controller = DualSenseController(20)

    while True:
        print(controller.get_last_command())
        time.sleep(0.05)
