import pygame
from threading import Thread
from queue import Queue
import time
import numpy as np


# MUJOCO
# X_RANGE = [-0.15, 0.15]
# Y_RANGE = [-0.2, 0.2]
# YAW_RANGE = [-1.0, 1.0]

# AWD
X_RANGE = [-0.3, 0.5]
Y_RANGE = [-0.3, 0.3]
YAW_RANGE = [-1.5, 1.5]

# rads
NECK_PITCH_RANGE = [-0.34, 1.1]
HEAD_PITCH_RANGE = [-0.78, 0.78]
HEAD_YAW_RANGE = [-2.7, 2.7]
HEAD_ROLL_RANGE = [-0.5, 0.5]


class XBoxController:
    def __init__(self, command_freq, standing=False):
        self.command_freq = command_freq
        self.standing = standing
        self.head_control_mode = self.standing

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_left_trigger = 0
        self.last_right_trigger = 0
        pygame.init()
        self.p1 = pygame.joystick.Joystick(0)
        self.p1.init()
        print(f"Loaded joystick with {self.p1.get_numaxes()} axes.")
        self.cmd_queue = Queue(maxsize=1)

        Thread(target=self.commands_worker, daemon=True).start()

    def commands_worker(self):
        while True:
            self.cmd_queue.put(self.get_commands())
            time.sleep(1 / self.command_freq)

    def get_commands(self):
        A_pressed = False
        X_pressed = False
        last_commands = self.last_commands
        left_trigger = self.last_left_trigger
        right_trigger = self.last_right_trigger
        for event in pygame.event.get():
            l_x = -1 * self.p1.get_axis(0)
            l_y = -1 * self.p1.get_axis(1)
            r_x = -1 * self.p1.get_axis(2)
            r_y = -1 * self.p1.get_axis(3)

            right_trigger = (self.p1.get_axis(4) + 1) / 2
            left_trigger = (self.p1.get_axis(5) + 1) / 2
            if left_trigger < 0.1:
                left_trigger = 0
            if right_trigger < 0.1:
                right_trigger = 0
            # print(f"Right trigger: {right_trigger}"
            #       f"Left trigger: {left_trigger}")

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
                last_commands[3] = 0.0 # neck pitch 0 for now

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

            if self.p1.get_button(0):  # A button
                A_pressed = True

            if self.p1.get_button(3):  # X button
                X_pressed = True

            # for i in range(10):
            #     if self.p1.get_button(i):
            #         print(f"Button {i} pressed")

            if self.p1.get_button(4):  # Y button
                self.head_control_mode = not self.head_control_mode


        pygame.event.pump()  # process event queue

        return np.around(last_commands, 3), A_pressed, X_pressed, left_trigger, right_trigger

    def get_last_command(self):
        A_pressed = False
        X_pressed = False

        try:
            self.last_commands, A_pressed, X_pressed, self.last_left_trigger, self.last_right_trigger = self.cmd_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_commands, A_pressed, X_pressed, self.last_left_trigger, self.last_right_trigger


if __name__ == "__main__":
    controller = XBoxController(20)

    while True:
        controller.get_last_command()
        time.sleep(0.05)
