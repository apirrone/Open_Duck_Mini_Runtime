#!/usr/bin/env python3
import os
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import pygame
from threading import Thread
from queue import Queue
import numpy as np
from mini_bdx_runtime.buttons import Buttons


X_RANGE = [-0.15, 0.15]
Y_RANGE = [-0.2, 0.2]
YAW_RANGE = [-1.0, 1.0]

# rads
NECK_PITCH_RANGE = [-0.34, 1.1]
HEAD_PITCH_RANGE = [-0.78, 0.3]
HEAD_YAW_RANGE = [-0.5, 0.5]
HEAD_ROLL_RANGE = [-0.5, 0.5]


# ============================
# Calibration + Mapping Utils
# ============================

THRESH_AXIS_MOVE = 0.40     # decisive movement needed on the intended axis
CROSS_AXIS_TOL  = 0.20     # allowed leakage on other axes during detection
BASELINE_SETTLE_S = 0.6    # time to let device report steady values
QUIET_TOL = 0.08           # max movement allowed during quiet window
QUIET_WINDOW_S = 0.4       # require this long of quiet before arming detection
CONSECUTIVE_SAMPLES = 4    # require N consistent samples on same axis to accept
AXIS_PAUSE_S = 3.0         # seconds between axis detections (stick prompts)
POLL_DT = 1.0 / 120.0
TIMEOUT_S = 15.0

def _default_mapping_path() -> str:
    """Always use ~/.config/gamepad_mapping.json, ensure directory exists."""
    home = os.path.expanduser("~")
    cfg_dir = os.path.join(home, ".config")
    os.makedirs(cfg_dir, exist_ok=True)
    return os.path.join(cfg_dir, "gamepad_mapping.json")

@dataclass
class AxisMapping:
    index: int
    invert: bool = False
    deadzone: float = 0.05
    type: str = "axis"   # "axis" or "button" for triggers (some pads map triggers as buttons)
    button_index: Optional[int] = None
    normalize01: bool = False   # True if raw axis is in [-1,1] but represents [0,1] trigger

class GamepadMapper:
    """
    Loads a JSON mapping and provides canonical accessors:
      lx, ly, rx, ry in [-1, 1] (right/up positive)
      lt, rt in [0, 1] (analog) or {0,1} if digital
    Also offers button(label) and dpad_up/down helpers.
    """
    def __init__(self, mapping: Dict[str, Any]):
        self.m = mapping

    @staticmethod
    def _apply_axis(raw: float, invert: bool, deadzone: float, normalize01: bool) -> float:
        if normalize01:
            raw = (raw + 1.0) * 0.5  # [-1,1] -> [0,1]
        if invert:
            raw = -raw
        if abs(raw) < deadzone:
            return 0.0
        return raw

    def _axis(self, joy, key: str) -> float:
        spec = self.m["axes"][key]
        if spec["type"] == "button":
            pressed = joy.get_button(spec["button_index"])
            return 1.0 if pressed else 0.0
        raw = joy.get_axis(spec["index"])
        return self._apply_axis(raw,
                                bool(spec.get("invert", False)),
                                float(spec.get("deadzone", 0.05)),
                                bool(spec.get("normalize01", False)))

    # Canonical getters
    def lx(self, joy) -> float: return float(self._axis(joy, "lx"))
    def ly(self, joy) -> float: return float(self._axis(joy, "ly"))
    def rx(self, joy) -> float: return float(self._axis(joy, "rx"))
    def ry(self, joy) -> float: return float(self._axis(joy, "ry"))
    def lt(self, joy) -> float: return float(self._axis(joy, "lt"))
    def rt(self, joy) -> float: return float(self._axis(joy, "rt"))

    def button_index(self, label: str) -> int:
        return int(self.m["buttons"][label])

    def button(self, joy, label: str) -> bool:
        return bool(joy.get_button(self.button_index(label)))

    def dpad_up(self, joy) -> bool:
        idx = int(self.m["hat"].get("index_up", -1))
        if idx < 0: return False
        return tuple(joy.get_hat(idx)) == tuple(self.m["hat"]["up"])

    def dpad_down(self, joy) -> bool:
        idx = int(self.m["hat"].get("index_down", -1))
        if idx < 0: return False
        return tuple(joy.get_hat(idx)) == tuple(self.m["hat"]["down"])


def _wait_for_axis_motion(joy: pygame.joystick.Joystick, prompt: str,
                          expect_positive: Optional[bool] = None,
                          allow_buttons: bool = False,
                          excluded_axes: Optional[set] = None,
                          require_quiet_start: bool = True,
                          require_release_cycle: bool = False) -> Tuple[str, AxisMapping]:
    """
    Robustly detect a single axis by CHANGE from a settled baseline.
    - Settles a baseline, then waits for a quiet period before arming detection.
    - Requires the same axis to dominate for CONSECUTIVE_SAMPLES (debounce).
    - If `require_release_cycle` is True (good for triggers), requires a pull-and-release
      on that axis before finalizing.
    - Respects `excluded_axes`.
    Note: we no longer rely on expect_positive for sticks; invert is computed later from sign.
    """
    if excluded_axes is None:
        excluded_axes = set()

    print(f"\n{prompt}")
    print("   (Move it decisively. Press ESC to abort.)")

    # 1) Baseline settle
    t_end = time.time() + BASELINE_SETTLE_S
    while time.time() < t_end:
        pygame.event.pump()
        time.sleep(POLL_DT)

    baseline = [joy.get_axis(i) for i in range(joy.get_numaxes())]

    # 2) Optional quiet window to avoid instant detections
    if require_quiet_start:
        quiet_start = None
        while True:
            pygame.event.pump()
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt()
                if allow_buttons and ev.type == pygame.JOYBUTTONDOWN:
                    bi = ev.button
                    while joy.get_button(bi):
                        pygame.event.pump()
                        time.sleep(POLL_DT)
                    print(f"   ✔ Detected button {bi}")
                    return ("button", AxisMapping(index=-1, type="button", button_index=bi))
            ok = True
            for ai in range(joy.get_numaxes()):
                if ai in excluded_axes:
                    continue
                if abs(joy.get_axis(ai) - baseline[ai]) > QUIET_TOL:
                    ok = False
                    break
            if ok:
                if quiet_start is None:
                    quiet_start = time.time()
                elif (time.time() - quiet_start) >= QUIET_WINDOW_S:
                    break
            else:
                quiet_start = None
            time.sleep(POLL_DT)

    # 3) Debounced dominant motion + optional release cycle
    same_axis_count = 0
    last_axis_idx = None
    motion_started = False
    peak_ai = None

    start_time = time.time()
    while time.time() - start_time < TIMEOUT_S:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt()
            if allow_buttons and ev.type == pygame.JOYBUTTONDOWN:
                bi = ev.button
                while joy.get_button(bi):
                    pygame.event.pump()
                    time.sleep(POLL_DT)
                print(f"   ✔ Detected button {bi}")
                return ("button", AxisMapping(index=-1, type="button", button_index=bi))

        deltas = []
        for ai in range(joy.get_numaxes()):
            if ai in excluded_axes:
                continue
            v = joy.get_axis(ai)
            d = abs(v - baseline[ai])
            deltas.append((d, ai, v))

        if deltas:
            deltas.sort(reverse=True)
            best_d, best_ai, best_v = deltas[0]
            runner_up_d = deltas[1][0] if len(deltas) > 1 else 0.0

            if best_d >= THRESH_AXIS_MOVE and runner_up_d <= CROSS_AXIS_TOL:
                if last_axis_idx is None or best_ai == last_axis_idx:
                    same_axis_count += 1
                else:
                    same_axis_count = 1
                last_axis_idx = best_ai

                if same_axis_count >= CONSECUTIVE_SAMPLES:
                    # We do not set invert here; run_calibration_wizard will deduce it from sign.
                    print(f"   ✔ Identified axis {best_ai} (value {best_v:.2f})")
                    if not require_release_cycle:
                        return ("axis", AxisMapping(index=best_ai))
                    else:
                        if not motion_started:
                            motion_started = True
                            peak_ai = best_ai
                        else:
                            if abs(joy.get_axis(peak_ai) - baseline[peak_ai]) <= QUIET_TOL:
                                print(f"   ✔ Identified axis {peak_ai} (trigger, with release)")
                                return ("axis", AxisMapping(index=peak_ai))

            else:
                same_axis_count = 0
                last_axis_idx = None

        time.sleep(POLL_DT)

    raise TimeoutError("Timed out waiting for input.")


def _wait_for_button(joy: pygame.joystick.Joystick, label: str) -> int:
    print(f"\nPress and release [{label}]")
    print("   (Press ESC to abort.)")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt()
            if event.type == pygame.JOYBUTTONDOWN:
                bi = event.button
                while joy.get_button(bi):
                    pygame.event.pump()
                    time.sleep(POLL_DT)
                print(f"   ✔ Button {label} = index {bi}")
                return bi
        pygame.event.pump()
        time.sleep(POLL_DT)


def _wait_for_hat(joy: pygame.joystick.Joystick, label: str, expect: Tuple[int, int]) -> Tuple[int, Tuple[int, int]]:
    if joy.get_numhats() == 0:
        print("\n   ⚠ No hats reported; will skip D-Pad mapping.")
        return (-1, (0, 0))
    print(f"\nPress D-Pad {label}")
    print("   (Release after detection. Press ESC to abort.)")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt()
        for hi in range(joy.get_numhats()):
            state = joy.get_hat(hi)
            if state == expect:
                while joy.get_hat(hi) == expect:
                    pygame.event.pump()
                    time.sleep(POLL_DT)
                print(f"   ✔ Hat {label} = index {hi}, value {expect}")
                return (hi, expect)
        pygame.event.pump()
        time.sleep(POLL_DT)


def _infer_invert_from_sign(joy: pygame.joystick.Joystick, axis_index: int, motion_for: str) -> bool:
    """
    Sample the axis briefly to determine sign of the commanded motion.
    motion_for: 'left' or 'up'
    Canonical: right/up should be positive.
      - If user moved LEFT: we expect negative; if positive, invert=True
      - If user moved UP:   we expect positive; if negative, invert=True
    """
    t_end = time.time() + 0.3
    peak = 0.0
    while time.time() < t_end:
        pygame.event.pump()
        v = joy.get_axis(axis_index)
        if abs(v) > abs(peak):
            peak = v
        time.sleep(POLL_DT)
    if motion_for == 'left':
        return peak > 0.0
    elif motion_for == 'up':
        return peak < 0.0
    return False


def run_calibration_wizard(joy: pygame.joystick.Joystick, outfile: str, device_tag: str = "") -> Dict[str, Any]:
    print("======================================")
    print("  Gamepad Calibration Wizard")
    print("======================================")
    name = joy.get_name()
    print(f"Device: {name}")
    print(f"Axes: {joy.get_numaxes()} | Buttons: {joy.get_numbuttons()} | Hats: {joy.get_numhats()}")
    print("Follow the instructions. Move decisively. You can press ESC to abort.")
    time.sleep(0.8)

    mapping: Dict[str, Any] = {
        "device_name": name,
        "device_tag": device_tag,
        "axes": {},
        "buttons": {},
        "hat": {}
    }

    # Sticks with axis exclusion
    identified_axes = set()

    kind, ax = _wait_for_axis_motion(joy, "Move the LEFT STICK to the LEFT ⬅️",
                                     excluded_axes=identified_axes)
    if kind != "axis":
        raise RuntimeError("Left stick X cannot be a button; please retry.")
    ax.invert = _infer_invert_from_sign(joy, ax.index, 'left')
    mapping["axes"]["lx"] = ax.__dict__
    identified_axes.add(ax.index)
    print(f"   ✅ Left stick X identified. Hold still for {AXIS_PAUSE_S:.0f} seconds…")
    time.sleep(AXIS_PAUSE_S)

    kind, ay = _wait_for_axis_motion(joy, "Move the LEFT STICK UP ⬆️",
                                     excluded_axes=identified_axes)
    if kind != "axis":
        raise RuntimeError("Left stick Y cannot be a button; please retry.")
    ay.invert = _infer_invert_from_sign(joy, ay.index, 'up')
    mapping["axes"]["ly"] = ay.__dict__
    identified_axes.add(ay.index)
    print(f"   ✅ Left stick Y identified. Hold still for {AXIS_PAUSE_S:.0f} seconds…")
    time.sleep(AXIS_PAUSE_S)

    kind, ax2 = _wait_for_axis_motion(joy, "Move the RIGHT STICK to the LEFT ⬅️",
                                      excluded_axes=identified_axes)
    if kind != "axis":
        raise RuntimeError("Right stick X cannot be a button; please retry.")
    ax2.invert = _infer_invert_from_sign(joy, ax2.index, 'left')
    mapping["axes"]["rx"] = ax2.__dict__
    identified_axes.add(ax2.index)
    print(f"   ✅ Right stick X identified. Hold still for {AXIS_PAUSE_S:.0f} seconds…")
    time.sleep(AXIS_PAUSE_S)

    kind, ay2 = _wait_for_axis_motion(joy, "Move the RIGHT STICK UP ⬆️",
                                      excluded_axes=identified_axes)
    if kind != "axis":
        raise RuntimeError("Right stick Y cannot be a button; please retry.")
    ay2.invert = _infer_invert_from_sign(joy, ay2.index, 'up')
    mapping["axes"]["ry"] = ay2.__dict__
    identified_axes.add(ay2.index)
    print(f"   ✅ Right stick Y identified. Hold still for {AXIS_PAUSE_S:.0f} seconds…")
    time.sleep(AXIS_PAUSE_S)

    # Triggers (accept axis or button) with exclusions and release-cycle
    for lab, key in [("LEFT TRIGGER (LT)", "lt"), ("RIGHT TRIGGER (RT)", "rt")]:
        kind, tmap = _wait_for_axis_motion(
            joy,
            f"Pull and release {lab}",
            allow_buttons=True,
            excluded_axes=identified_axes,
            require_quiet_start=True,
            require_release_cycle=True,
        )
        if kind == "axis":
            read = joy.get_axis(tmap.index)
            print(f"   Observed resting value: {read:.2f}. Fully pull and release once more…")
            time.sleep(0.5)
            peak = read
            t_end = time.time() + 1.5
            while time.time() < t_end:
                pygame.event.pump()
                v = joy.get_axis(tmap.index)
                if abs(v) > abs(peak):
                    peak = v
                time.sleep(POLL_DT)
            tmap.normalize01 = abs(read) > 0.5 and peak > read
            identified_axes.add(tmap.index)
        mapping["axes"][key] = tmap.__dict__
        print(f"   ✅ {lab} identified. Hold still for {AXIS_PAUSE_S:.0f} seconds…")
        time.sleep(AXIS_PAUSE_S)

    # Face buttons
    for label in ["A", "B", "X", "Y", "LB", "RB"]:
        bi = _wait_for_button(joy, label)
        mapping["buttons"][label] = bi

    # D-Pad (hat)
    if joy.get_numhats() > 0:
        hi, _ = _wait_for_hat(joy, "UP", (0, 1))
        hi2, _ = _wait_for_hat(joy, "DOWN", (0, -1))
        mapping["hat"] = {
            "index_up": hi,
            "index_down": hi2,
            "up": [0, 1],
            "down": [0, -1]
        }
    else:
        mapping["hat"] = {"index_up": -1, "index_down": -1, "up": [0, 0], "down": [0, 0]}

    with open(outfile, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n✅ Saved mapping to {outfile}")
    return mapping


def ensure_mapping(joy: pygame.joystick.Joystick, mapping_path: Optional[str]) -> GamepadMapper:
    # Resolve default path under user's home if not provided
    if not mapping_path:
        mapping_path = _default_mapping_path()
    else:
        # If caller passed a basename like "gamepad_mapping.json", still put it under ~/.config
        if os.path.basename(mapping_path) == mapping_path:
            mapping_path = os.path.join(os.path.expanduser("~"), ".config", mapping_path)
        # Ensure parent exists
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        return GamepadMapper(mapping)

    print(f"\nNo mapping file found at '{mapping_path}'. Starting calibration wizard...")
    mapping = run_calibration_wizard(joy, mapping_path)
    return GamepadMapper(mapping)


class XBoxController:
    def __init__(self, command_freq, only_head_control=False, mapping_path: Optional[str] = None):
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

        self.A_pressed = False
        self.B_pressed = False
        self.X_pressed = False
        self.Y_pressed = False
        self.LB_pressed = False
        self.RB_pressed = False

        # Load or create universal mapping (always under ~/.config by default)
        self.mapper = ensure_mapping(self.p1, mapping_path)

        # Cache button indices for event checks
        self._btn_idx = {
            "A": self.mapper.button_index("A"),
            "B": self.mapper.button_index("B"),
            "X": self.mapper.button_index("X"),
            "Y": self.mapper.button_index("Y"),
            "LB": self.mapper.button_index("LB"),
            "RB": self.mapper.button_index("RB"),
        }

        # Hat info
        self._hat_up_idx = int(self.mapper.m["hat"].get("index_up", -1))
        self._hat_down_idx = int(self.mapper.m["hat"].get("index_down", -1))
        self._hat_up_val = tuple(self.mapper.m["hat"].get("up", (0, 1)))
        self._hat_down_val = tuple(self.mapper.m["hat"].get("down", (0, -1)))

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

        # Use universal mapper (right/up positive)
        l_x = -self.mapper.lx(self.p1)
        l_y = self.mapper.ly(self.p1)
        r_x = -self.mapper.rx(self.p1)
        r_y = self.mapper.ry(self.p1)

        right_trigger = np.around(self.mapper.rt(self.p1), 3)
        left_trigger  = np.around(self.mapper.lt(self.p1), 3)

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

                if self.p1.get_button(self._btn_idx["A"]):  # A button
                    self.A_pressed = True

                if self.p1.get_button(self._btn_idx["B"]):  # B button
                    self.B_pressed = True

                if self.p1.get_button(self._btn_idx["X"]):  # X button
                    self.X_pressed = True

                if self.p1.get_button(self._btn_idx["Y"]):  # Y button
                    self.Y_pressed = True
                    if not self.only_head_control:
                        self.head_control_mode = not self.head_control_mode

                if self.p1.get_button(self._btn_idx["LB"]):  # LB button
                    self.LB_pressed = True

                if self.p1.get_button(self._btn_idx["RB"]):  # RB button
                    self.RB_pressed = True

            if event.type == pygame.JOYBUTTONUP:
                self.A_pressed = False
                self.B_pressed = False
                self.X_pressed = False
                self.Y_pressed = False
                self.LB_pressed = False
                self.RB_pressed = False

            # for i in range(10):
            #     if self.p1.get_button(i):
            #         print(f"Button {i} pressed")

        # D-pad up/down via mapped hat (fallback to 0 if none)
        up_down = 0
        if self._hat_up_idx >= 0 and tuple(self.p1.get_hat(self._hat_up_idx)) == self._hat_up_val:
            up_down = 1
        elif self._hat_down_idx >= 0 and tuple(self.p1.get_hat(self._hat_down_idx)) == self._hat_down_val:
            up_down = -1

        pygame.event.pump()  # process event queue

        return (
            np.around(last_commands, 3),
            self.A_pressed,
            self.B_pressed,
            self.X_pressed,
            self.Y_pressed,
            self.LB_pressed,
            self.RB_pressed,
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
        )

        return (
            self.last_commands,
            self.buttons,
            self.last_left_trigger,
            self.last_right_trigger,
        )


if __name__ == "__main__":
    # Default: mapping saved/loaded under ~/.config/gamepad_mapping.json
    controller = XBoxController(20)

    while True:
        print(controller.get_last_command())
        time.sleep(0.05)
