"""
Keyboard Controller
--------------------
Enables controlling the robot via WASD + Q/E keys over an SSH terminal
(no display required – uses raw stdin via :mod:`termios`).

Key bindings
~~~~~~~~~~~~
  W            – Walk forward
  S            – Walk backward
  A            – Yaw / turn left
  D            – Yaw / turn right
  Q            – Strafe left
  E            – Strafe right
  Space        – Toggle pause (maps to the "A" button)
  L            – Speed boost while held (maps to LB)
  P            – Toggle head-control mode (maps to Y button)
  X            – Toggle projector (maps to X button)
  B            – Play random sound (maps to B button)
  Ctrl-C / ESC – Exit

Movement keys are "held" by tracking the last-seen timestamp; if no new
event arrives within ``key_timeout`` seconds the key is considered released.
"""

from __future__ import annotations

import os
import sys
import select
import termios
import tty
import time
from threading import Thread, Lock
from queue import Queue
import numpy as np

from open_duck_mini_runtime.buttons import Buttons

X_RANGE = [-0.15, 0.15]
Y_RANGE = [-0.2, 0.2]
YAW_RANGE = [-1.0, 1.0]

# How long (seconds) without seeing a keypress before we treat the key as released.
_KEY_HOLD_TIMEOUT = 0.15


class _KeyState:
    """Tracks whether a key is considered "held" based on recent events."""

    def __init__(self):
        self._last_seen: float = 0.0
        self._lock = Lock()

    def press(self) -> None:
        with self._lock:
            self._last_seen = time.monotonic()

    def is_held(self, timeout: float = _KEY_HOLD_TIMEOUT) -> bool:
        with self._lock:
            return (time.monotonic() - self._last_seen) < timeout


class KeyboardController:
    """Drop-in replacement for XBoxController / DualSenseController using
    the keyboard over a standard SSH terminal session.

    Returns the same ``(commands, buttons, left_trigger, right_trigger)``
    tuple from :meth:`get_last_command` so it plugs straight into
    :class:`~open_duck_mini_runtime.walk.RLWalk`.
    """

    def __init__(self, command_freq: float, only_head_control: bool = False):
        self.command_freq = command_freq
        self.head_control_mode = only_head_control
        self.only_head_control = only_head_control

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.buttons = Buttons()

        # Key state for held movement keys (all lowercase – reader thread lowercases input)
        self._keys: dict[str, _KeyState] = {k: _KeyState() for k in "wasdeqxblp "}
        # Enter / Return is handled as a one-shot START event (raw mode sends \r)

        # Queued one-shot button events (triggered on edge – key just pressed)
        self._event_queue: Queue[str] = Queue()

        # Alive-flag used by the reader thread
        self._running = True

        self._reader_thread = Thread(target=self._read_stdin, daemon=True)
        self._reader_thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_stdin(self) -> None:
        """Background thread: put raw key characters into the event queue
        and update held-key timestamps.  Works over SSH (no display needed)."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd, termios.TCSANOW)
            while self._running:
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not readable:
                    continue
                ch = os.read(fd, 1)
                if not ch:
                    continue
                key = ch.decode("utf-8", errors="ignore").lower()

                # Escape / Ctrl-C  →  signal exit to caller
                if key in ("\x03", "\x1b"):
                    self._event_queue.put("__quit__")
                    break

                # Track held-key state for movement keys
                if key in self._keys:
                    self._keys[key].press()

                # Enqueue one-shot events for button-trigger keys
                if key in (" ", "x", "b", "p", "\r"):
                    self._event_queue.put(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # ------------------------------------------------------------------
    # Public API (matches XBoxController / DualSenseController)
    # ------------------------------------------------------------------

    def get_last_command(self):
        """Compute the current command vector from keyboard state and return
        ``(commands, buttons, left_trigger, right_trigger)``."""

        # --- Movement commands from held keys ---
        lin_vel_x = 0.0
        lin_vel_y = 0.0
        ang_vel = 0.0

        if self._keys["w"].is_held():
            lin_vel_x += X_RANGE[1]  # forward
        if self._keys["s"].is_held():
            lin_vel_x += X_RANGE[0]  # backward
        if self._keys["q"].is_held():
            lin_vel_y += Y_RANGE[1]  # strafe left
        if self._keys["e"].is_held():
            lin_vel_y += Y_RANGE[0]  # strafe right
        if self._keys["a"].is_held():
            ang_vel += YAW_RANGE[1]  # turn left
        if self._keys["d"].is_held():
            ang_vel += YAW_RANGE[0]  # turn right

        # Clamp to valid ranges
        lin_vel_x = float(np.clip(lin_vel_x, X_RANGE[0], X_RANGE[1]))
        lin_vel_y = float(np.clip(lin_vel_y, Y_RANGE[0], Y_RANGE[1]))
        ang_vel = float(np.clip(ang_vel, YAW_RANGE[0], YAW_RANGE[1]))

        self.last_commands[0] = round(lin_vel_x, 3)
        self.last_commands[1] = round(lin_vel_y, 3)
        self.last_commands[2] = round(ang_vel, 3)

        # --- One-shot / edge-triggered button events ---
        A_pressed = False
        X_pressed = False
        B_pressed = False
        Y_pressed = False  # head-mode toggle

        while not self._event_queue.empty():
            ev = self._event_queue.get_nowait()
            if ev == " ":
                A_pressed = True
            elif ev == "x":
                X_pressed = True
            elif ev == "b":
                B_pressed = True
            elif ev == "p":
                Y_pressed = True

        # L key held → LB (speed boost)
        LB_pressed = self._keys["l"].is_held()

        self.buttons.update(
            A_pressed,
            B_pressed,
            X_pressed,
            Y_pressed,
            LB_pressed,
            False,  # RB – not mapped
            False,  # dpad_up
            False,  # dpad_down
        )

        return (
            list(self.last_commands),
            self.buttons,
            0.0,  # left_trigger  (not used with keyboard)
            0.0,  # right_trigger (not used with keyboard)
        )

    def stop(self) -> None:
        """Signal the reader thread to exit and restore terminal state."""
        self._running = False

    @staticmethod
    def print_controls() -> None:
        print(
            "\n"
            "┌──────────────────────────────────┐\n"
            "│       Keyboard Walk Controls      │\n"
            "├──────────────────────────────────┤\n"
            "│  W / S       – Forward / Backward │\n"
            "│  A / D       – Turn left / right  │\n"
            "│  Q / E       – Strafe left / right│\n"
            "│  L (hold)    – Speed boost (LB)   │\n"
            "│  Space       – Pause / Unpause    │\n"
            "│  X           – Toggle projector   │\n"
            "│  B           – Play random sound  │\n"
            "│  P           – Head-control mode  │\n"
            "│  Enter       – Toggle motors on/off│\n"
            "│  Ctrl-C / ESC – Exit              │\n"
            "└──────────────────────────────────┘\n"
        )


if __name__ == "__main__":
    ctrl = KeyboardController(20)
    KeyboardController.print_controls()
    try:
        while True:
            cmds, buttons, lt, rt = ctrl.get_last_command()
            print(
                f"\r  lin_x={cmds[0]:+.3f}  lin_y={cmds[1]:+.3f}  yaw={cmds[2]:+.3f}"
                f"  LB={'Y' if buttons.LB.is_pressed else 'N'}"
                f"  paused={'Y' if buttons.A.triggered else 'N'}    ",
                end="",
            )
            time.sleep(1 / 20)
    except KeyboardInterrupt:
        ctrl.stop()
        print("\nDone.")
