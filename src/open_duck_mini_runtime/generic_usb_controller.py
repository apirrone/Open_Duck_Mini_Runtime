"""
Generic USB Gamepad Controller
-------------------------------
Works with most xinput / HID-compliant USB gamepads that SDL2 can detect.
Axis and button indices can be overridden via duck_config.json under the
"generic_usb_controller" key so you can adapt to any physical device.

Default mapping (standard Linux HID / SDL2 gamepad profile):
  Axis 0  – Left stick X   (strafe)
  Axis 1  – Left stick Y   (forward/back)
  Axis 2  – Right stick X  (yaw)
  Axis 3  – Right stick Y  (unused)
  Axis 4  – L2 / Left trigger
  Axis 5  – R2 / Right trigger

  Button 0 – A / Cross     (pause / unpause)
  Button 1 – B / Circle    (sound)
  Button 2 – X / Square    (projector)
  Button 3 – Y / Triangle  (toggle head-control mode)
  Button 4 – LB / L1       (speed boost while held)
  Button 5 – RB / R1
"""

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

# Default axis / button indices – can be overridden via duck_config.json
_DEFAULT_AXIS_MAP = {
    "left_x": 0,
    "left_y": 1,
    "right_x": 2,
    "right_y": 3,
    "left_trigger": 4,
    "right_trigger": 5,
}

_DEFAULT_BUTTON_MAP = {
    "A": 0,
    "B": 1,
    "X": 2,
    "Y": 3,
    "LB": 4,
    "RB": 5,
}


class GenericUSBController:
    """Drop-in replacement for :class:`XBoxController` / :class:`DualSenseController`
    that works with a wide variety of USB gamepads via SDL2 / pygame.

    The axis and button indices can be customised through duck_config.json under
    the ``"generic_usb_controller"`` section:

    .. code-block:: json

        "generic_usb_controller": {
            "axis_map":   { "left_x": 0, "left_y": 1, "right_x": 2, "right_y": 3,
                            "left_trigger": 4, "right_trigger": 5 },
            "button_map": { "A": 0, "B": 1, "X": 2, "Y": 3, "LB": 4, "RB": 5 },
            "joystick_index": 0
        }
    """

    def __init__(self, command_freq: float, only_head_control: bool = False, config_overrides: dict = None):
        self.command_freq = command_freq
        self.head_control_mode = only_head_control
        self.only_head_control = only_head_control

        cfg = config_overrides or {}
        self.axis_map = {**_DEFAULT_AXIS_MAP, **cfg.get("axis_map", {})}
        self.button_map = {**_DEFAULT_BUTTON_MAP, **cfg.get("button_map", {})}
        joystick_index = cfg.get("joystick_index", 0)

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.last_left_trigger = 0.0
        self.last_right_trigger = 0.0

        pygame.init()
        self.p1 = pygame.joystick.Joystick(joystick_index)
        self.p1.init()
        print(
            f"[GenericUSBController] Loaded '{self.p1.get_name()}' "
            f"with {self.p1.get_numaxes()} axes and {self.p1.get_numbuttons()} buttons."
        )

        self.cmd_queue: Queue = Queue(maxsize=1)

        self._A_pressed = False
        self._B_pressed = False
        self._X_pressed = False
        self._Y_pressed = False
        self._LB_pressed = False
        self._RB_pressed = False

        self.buttons = Buttons()

        Thread(target=self._commands_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _commands_worker(self) -> None:
        while True:
            self.cmd_queue.put(self._get_commands())
            time.sleep(1 / self.command_freq)

    def _read_axis(self, name: str) -> float:
        idx = self.axis_map[name]
        if idx < self.p1.get_numaxes():
            return self.p1.get_axis(idx)
        return 0.0

    def _read_button(self, name: str) -> bool:
        idx = self.button_map[name]
        if idx < self.p1.get_numbuttons():
            return bool(self.p1.get_button(idx))
        return False

    def _get_commands(self):
        last_commands = list(self.last_commands)
        left_trigger = self.last_left_trigger
        right_trigger = self.last_right_trigger

        l_x = -1 * self._read_axis("left_x")
        l_y = -1 * self._read_axis("left_y")
        r_x = -1 * self._read_axis("right_x")

        # Triggers: SDL2 reports them in [-1, 1]; remap to [0, 1]
        right_trigger = np.around((self._read_axis("right_trigger") + 1) / 2, 3)
        left_trigger = np.around((self._read_axis("left_trigger") + 1) / 2, 3)

        if left_trigger < 0.1:
            left_trigger = 0.0
        if right_trigger < 0.1:
            right_trigger = 0.0

        if not self.head_control_mode:
            lin_vel_y = l_x
            lin_vel_x = l_y
            ang_vel = r_x

            lin_vel_x *= np.abs(X_RANGE[1]) if lin_vel_x >= 0 else np.abs(X_RANGE[0])
            lin_vel_y *= np.abs(Y_RANGE[1]) if lin_vel_y >= 0 else np.abs(Y_RANGE[0])
            ang_vel *= np.abs(YAW_RANGE[1]) if ang_vel >= 0 else np.abs(YAW_RANGE[0])

            last_commands[0] = lin_vel_x
            last_commands[1] = lin_vel_y
            last_commands[2] = ang_vel
        else:
            last_commands[0] = 0.0
            last_commands[1] = 0.0
            last_commands[2] = 0.0
            last_commands[3] = 0.0  # neck pitch

            head_yaw = l_x
            head_pitch = l_y
            head_roll = r_x

            head_yaw *= np.abs(HEAD_YAW_RANGE[0]) if head_yaw >= 0 else np.abs(HEAD_YAW_RANGE[1])
            head_pitch *= np.abs(HEAD_PITCH_RANGE[0]) if head_pitch >= 0 else np.abs(HEAD_PITCH_RANGE[1])
            head_roll *= np.abs(HEAD_ROLL_RANGE[0]) if head_roll >= 0 else np.abs(HEAD_ROLL_RANGE[1])

            last_commands[4] = head_pitch
            last_commands[5] = head_yaw
            last_commands[6] = head_roll

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if self._read_button("A"):
                    self._A_pressed = True
                if self._read_button("B"):
                    self._B_pressed = True
                if self._read_button("X"):
                    self._X_pressed = True
                if self._read_button("Y"):
                    self._Y_pressed = True
                    if not self.only_head_control:
                        self.head_control_mode = not self.head_control_mode
                if self._read_button("LB"):
                    self._LB_pressed = True
                if self._read_button("RB"):
                    self._RB_pressed = True

            if event.type == pygame.JOYBUTTONUP:
                self._A_pressed = False
                self._B_pressed = False
                self._X_pressed = False
                self._Y_pressed = False
                self._LB_pressed = False
                self._RB_pressed = False

        up_down = self.p1.get_hat(0)[1] if self.p1.get_numhats() > 0 else 0
        pygame.event.pump()

        return (
            np.around(last_commands, 3),
            self._A_pressed,
            self._B_pressed,
            self._X_pressed,
            self._Y_pressed,
            self._LB_pressed,
            self._RB_pressed,
            left_trigger,
            right_trigger,
            up_down,
        )

    # ------------------------------------------------------------------
    # Public API (matches XBoxController / DualSenseController)
    # ------------------------------------------------------------------

    def get_last_command(self):
        """Return ``(last_commands, buttons, left_trigger, right_trigger)``."""
        A_pressed = False
        B_pressed = False
        X_pressed = False
        Y_pressed = False
        LB_pressed = False
        RB_pressed = False
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
                self.last_left_trigger,
                self.last_right_trigger,
                up_down,
            ) = self.cmd_queue.get(False)
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
        )

        return (
            self.last_commands,
            self.buttons,
            self.last_left_trigger,
            self.last_right_trigger,
        )


if __name__ == "__main__":
    controller = GenericUSBController(20)
    while True:
        print(controller.get_last_command())
        time.sleep(0.05)
