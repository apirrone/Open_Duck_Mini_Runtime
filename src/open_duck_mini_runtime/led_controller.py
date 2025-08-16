"""
Shared controller for a single 3-LED NeoPixel strip used by eyes (2 LEDs) and projector (1 LED).

Design:
- Index 0: projector
- Index 1: right eye
- Index 2: left eye
"""
from __future__ import annotations

from threading import Lock
from typing import Tuple, Optional, Union, Dict

# Direct hardware imports (we assume we're running on-device)
import board
import neopixel

# Pin and pixel configuration
PIXEL_PIN = board.D10
NUM_PIXELS = 3

# The order of the pixel colors - RGB or GRB. Some NeoPixels have red and green reversed!
# For RGBW NeoPixels, simply change the ORDER to RGBW or GRBW.
ORDER = neopixel.RGBW

# Brightness and a single shared NeoPixel instance
BRIGHTNESS = 1
pixels = neopixel.NeoPixel(
    PIXEL_PIN, NUM_PIXELS, brightness=BRIGHTNESS, auto_write=False, pixel_order=ORDER
)

class LedController:
    def __init__(self) -> None:
        self._lock = Lock()
        self._pixels = None  # type: ignore
        self._deinited = False

        # Use the module-level NeoPixel configured above
        self._pixels = pixels

        # Cache simple color tuples (use 4-tuple for RGBW strips)
        # Colors are stored in logical (R, G, B, W) regardless of ORDER.
        self.OFF = (0, 0, 0, 0)
        self.WHITE = (0, 0, 0, 255)
        self.RED = (255, 0, 0, 0)
        self.GREEN = (0, 255, 0, 0)
        self.BLUE = (0, 0, 255, 0)

        self._named_colors = {
            "off": self.OFF,
            "white": self.WHITE,
            "red": self.RED,
            "green": self.GREEN,
            "blue": self.BLUE,
        }

        # Track on/off states (for toggling and consistent show)
        self.eye_left_on = True
        self.eye_right_on = True
        self.projector_on = False

        # Track current colors for each pixel (default WHITE)
        self.left_color = self.WHITE
        self.right_color = self.WHITE
        self.proj_color = self.WHITE

        # Ensure an initial known state
        self._apply()

    # Internal helper to apply current states to pixels
    def _apply(
        self,
        left_color: Optional[Tuple[int, int, int, int]] = None,
        right_color: Optional[Tuple[int, int, int, int]] = None,
        proj_color: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        if self._pixels is None or self._deinited:
            return
        with self._lock:
            # Determine colors from boolean states when not explicitly specified
            if left_color is None:
                left_color = self.left_color if self.eye_left_on else self.OFF
            if right_color is None:
                right_color = self.right_color if self.eye_right_on else self.OFF
            if proj_color is None:
                proj_color = self.proj_color if self.projector_on else self.OFF

            # Assign indices: 2-left, 1-right, 0-projector
            self._pixels[0] = self._to_order(proj_color)
            self._pixels[1] = self._to_order(right_color)
            self._pixels[2] = self._to_order(left_color)
            self._pixels.show()

    def _to_order(self, color_rgba: Tuple[int, int, int, int]):
        """
        Convert logical (R,G,B,W) into the configured NeoPixel ORDER tuple.
        For RGB strips (no W), we will drop the W channel.
        """
        r, g, b, w = color_rgba
        try:
            # neopixel library generally accepts (r,g,b) or (r,g,b,w) depending on pixel type
            # We map to the declared ORDER for clarity when direct indexing is used.
            if ORDER in (neopixel.RGB, neopixel.GRB):
                mapping = {
                    neopixel.RGB: (r, g, b),
                    neopixel.GRB: (g, r, b),
                }
                return mapping[ORDER]
            else:
                # RGBW variants
                mapping = {
                    neopixel.RGBW: (r, g, b, w),
                    neopixel.GRBW: (g, r, b, w),
                }
                return mapping.get(ORDER, (r, g, b, w))
        except Exception:
            # Fallback: return RGBA or RGB as-is
            return (r, g, b, w)

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

    # Color API (accepts name or tuple)
    def _norm_color(self, color: Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]
                   ) -> Tuple[int, int, int, int]:
        if isinstance(color, str):
            c = self._named_colors.get(color.lower())
            if c is None:
                raise ValueError(f"Unknown color name: {color}")
            return c
        # If 3-tuple provided, assume RGB on RGBW strip -> map to (r,g,b,0)
        if isinstance(color, tuple) and len(color) == 3:
            r, g, b = color
            return (r, g, b, 0)
        if isinstance(color, tuple) and len(color) == 4:
            return color  # already RGBA(W)
        raise ValueError("Color must be a name or RGB/RGBW tuple")

    def set_left_eye_color(self, color: Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]):
        self.left_color = self._norm_color(color)
        self._apply()

    def set_right_eye_color(self, color: Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]):
        self.right_color = self._norm_color(color)
        self._apply()

    def set_projector_color(self, color: Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]):
        self.proj_color = self._norm_color(color)
        self._apply()

    def set_eyes_color(self, color: Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]):
        norm = self._norm_color(color)
        self.left_color = norm
        self.right_color = norm
        self._apply()

    def set_all_color(self, color: Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]):
        norm = self._norm_color(color)
        self.left_color = norm
        self.right_color = norm
        self.proj_color = norm
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
