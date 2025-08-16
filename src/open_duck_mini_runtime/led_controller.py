"""
Shared controller for a single 3-LED NeoPixel strip used by eyes (2 LEDs) and projector (1 LED).

Design:
- Index 0: left eye
- Index 1: right eye
- Index 2: projector

This module lazily imports hardware libraries to avoid import errors on non-hardware hosts.
"""
from __future__ import annotations

from threading import Lock
from typing import Tuple, Optional


class LedController:
    def __init__(
        self,
        data_pin_name: str = "D23",  # default GPIO for NeoPixel data
        num_pixels: int = 3,
        brightness: float = 0.3,
    ) -> None:
        self._lock = Lock()
        self._pixels = None  # type: ignore
        self._deinited = False

        # Lazy import to prevent issues on dev machines/CI without hardware
        # Importing here avoids name collision with our package module names.
        import importlib

        board = importlib.import_module("board")
        neopixel = importlib.import_module("neopixel")

        data_pin = getattr(board, data_pin_name)

        # Initialize NeoPixel strip
        self._pixels = neopixel.NeoPixel(
            data_pin,
            num_pixels,
            brightness=brightness,
            auto_write=False,
            pixel_order=getattr(neopixel, "GRB", None) or getattr(neopixel, "RGB"),
        )

        # Cache simple color tuples
        self.OFF = (0, 0, 0)
        self.WHITE = (255, 255, 255)

        # Track on/off states (for toggling and consistent show)
        self.eye_left_on = True
        self.eye_right_on = True
        self.projector_on = False

        # Ensure an initial known state
        self._apply()

    # Internal helper to apply current states to pixels
    def _apply(
        self,
        left_color: Optional[Tuple[int, int, int]] = None,
        right_color: Optional[Tuple[int, int, int]] = None,
        proj_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        if self._pixels is None or self._deinited:
            return
        with self._lock:
            # Determine colors from boolean states when not explicitly specified
            if left_color is None:
                left_color = self.WHITE if self.eye_left_on else self.OFF
            if right_color is None:
                right_color = self.WHITE if self.eye_right_on else self.OFF
            if proj_color is None:
                proj_color = self.WHITE if self.projector_on else self.OFF

            # Assign indices: 0-left, 1-right, 2-projector
            self._pixels[0] = left_color
            self._pixels[1] = right_color
            self._pixels[2] = proj_color
            self._pixels.show()

    # Eyes API
    def set_eyes(self, on: bool) -> None:
        self.eye_left_on = on
        self.eye_right_on = on
        self._apply()

    def set_left_eye(self, on: bool) -> None:
        self.eye_left_on = on
        self._apply()

    def set_right_eye(self, on: bool) -> None:
        self.eye_right_on = on
        self._apply()

    # Projector API
    def set_projector(self, on: bool) -> None:
        self.projector_on = on
        self._apply()

    # Utilities
    def all_off(self) -> None:
        self.eye_left_on = False
        self.eye_right_on = False
        self.projector_on = False
        self._apply()

    def deinit(self) -> None:
        if self._deinited or self._pixels is None:
            return
        with self._lock:
            try:
                self.all_off()
            except Exception:
                # Ignore if showing fails on teardown
                pass
            try:
                self._pixels.deinit()
            finally:
                self._deinited = True
                self._pixels = None


_controller: Optional[LedController] = None


def get_controller() -> LedController:
    global _controller
    if _controller is None:
        _controller = LedController()
    return _controller
