"""
todo: add a "standby" mode for the original GPIO eyes (left/right alternate blink)
or something like that.

2-todo. split into a separate thread so that the blink loop doesn't block the main thread?

3-todo. maybe split the original eye code and neopixel code into separate files or classes so that the main Eyes class is just a wrapper that picks which one to use based on the hardware.
"""

import logging
import random
import time
from threading import Thread, Event

logger = logging.getLogger(__name__)

# todo: did i even include support for these in our new runtime? check this.
# GPIO pins for the original single-colour LED eyes (used when neopixels=False)
_LEFT_EYE_PIN_NAME = "D24"
_RIGHT_EYE_PIN_NAME = "D23"


# oooo
# this is an S-O-S
# don't wanna second guess
# this is the bottom line it's trueeee

# SOS timing (short = dot, long = dash)
_SOS_DOT = 0.12
_SOS_DASH = 0.36
_SOS_ELEM_GAP = 0.12  # gap between elements within a letter
_SOS_LETTER_GAP = 0.36  # gap between letters
_SOS_CYCLE_GAP = 1.0  # pause after full S-O-S before repeating

# SOS pattern: S = . . .   O = - - -   S = . . .
_SOS_PATTERN = (
    [_SOS_DOT, _SOS_DOT, _SOS_DOT],  # S
    [_SOS_DASH, _SOS_DASH, _SOS_DASH],  # O
    [_SOS_DOT, _SOS_DOT, _SOS_DOT],  # S
)


class Eyes:
    """
    Manages the duck's eye LEDs.

    Parameters
    ----------
    blink_duration : float
        How long (seconds) eyes stay closed during a random blink.
    min_interval / max_interval : float
        Random blink interval range (seconds).
    neopixels : bool
        ``True``  — use the NeoPixel LED controller (default, new hardware).
        ``False`` — use original single-colour GPIO eyes (old hardware).
    """

    def __init__(
        self,
        blink_duration: float = 0.1,
        min_interval: float = 1.0,
        max_interval: float = 4.0,
        neopixels: bool = True,
    ):
        self.blink_duration = blink_duration
        self.min_interval = min_interval
        self.max_interval = max_interval
        self._neopixels = neopixels

        # _solid: when True the blink thread keeps eyes on (neopixel) or runs SOS (original)
        self._solid = False
        # _standby: when True (original mode only) run alternating L/R blink instead of
        #           synchronised random blink; ignored in neopixel mode
        self._standby = False

        if neopixels:
            self._init_neopixel()
        else:
            self._init_gpio()

        self._stop_event = Event()
        self._thread = Thread(target=self.run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Hardware initialisation
    # ------------------------------------------------------------------

    def _init_neopixel(self) -> None:
        from open_duck_mini_runtime.hardware.led_controller import get_controller

        self.ctrl = get_controller()
        self.ctrl.set_eyes_color("white")
        self.ctrl.set_eyes(True)

    def _init_gpio(self) -> None:
        import board
        import digitalio

        left_pin = getattr(board, _LEFT_EYE_PIN_NAME)
        right_pin = getattr(board, _RIGHT_EYE_PIN_NAME)
        self._left_eye = digitalio.DigitalInOut(left_pin)
        self._left_eye.direction = digitalio.Direction.OUTPUT
        self._right_eye = digitalio.DigitalInOut(right_pin)
        self._right_eye.direction = digitalio.Direction.OUTPUT
        self._left_eye.value = True
        self._right_eye.value = True

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _set_eyes(self, state: bool) -> None:
        """Turn both eyes on/off."""
        if self._neopixels:
            self.ctrl.set_eyes(state)
        else:
            self._left_eye.value = state
            self._right_eye.value = state

    def _set_left(self, state: bool) -> None:
        """Original GPIO only — set left eye independently."""
        self._left_eye.value = state

    def _set_right(self, state: bool) -> None:
        """Original GPIO only — set right eye independently."""
        self._right_eye.value = state

    # ------------------------------------------------------------------
    # Public API (mirrors the original neopixel interface)
    # ------------------------------------------------------------------

    def set_color(self, color) -> None:
        """Change eye colour (neopixel mode only; no-op for original GPIO eyes)."""
        if not self._neopixels:
            return
        self.ctrl.set_eyes_color(color)
        self.ctrl.set_eyes(True)

    def set_solid(self, solid: bool) -> None:
        """
        Neopixel mode  — suppress blinking (eyes stay on) when ``solid=True``.
        Original mode  — trigger SOS blink pattern when ``solid=True`` (motors off).
        """
        self._solid = solid
        if self._neopixels and solid:
            self.ctrl.set_eyes(True)

    def set_standby(self, standby: bool) -> None:
        """
        Original GPIO mode only — switch between normal random blink (``False``)
        and alternating left/right blink (``True``, used while paused/standby).
        This is a no-op in neopixel mode.
        """
        self._standby = standby

    # ------------------------------------------------------------------
    # Blink thread
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._neopixels:
                    self._run_neopixel_tick()
                else:
                    self._run_gpio_tick()
        except Exception as err:
            logger.error("Eye thread error: %s", err)
            self._stop_event.set()

    def _run_neopixel_tick(self) -> None:
        """One iteration of the neopixel blink loop (unchanged behaviour)."""
        if self._solid:
            self._stop_event.wait(0.1)
            return
        self._set_eyes(False)
        if self._stop_event.wait(self.blink_duration):
            return
        self._set_eyes(True)
        next_blink = random.uniform(self.min_interval, self.max_interval)
        self._stop_event.wait(next_blink)

    def _run_gpio_tick(self) -> None:
        """One iteration of the original GPIO blink loop."""
        if self._solid:
            # Motors off → SOS distress signal
            self._blink_sos()
        elif self._standby:
            # Paused / standby → alternate left-right
            self._blink_alternate()
        else:
            # Walking → synchronised random blink (original behaviour)
            self._set_eyes(False)
            if self._stop_event.wait(self.blink_duration):
                return
            self._set_eyes(True)
            next_blink = random.uniform(self.min_interval, self.max_interval)
            self._stop_event.wait(next_blink)

    def _blink_sos(self) -> None:
        """Blink SOS in Morse code using both eyes together."""
        for letter_idx, durations in enumerate(_SOS_PATTERN):
            for elem_idx, dur in enumerate(durations):
                if self._stop_event.is_set() or not self._solid:
                    self._set_eyes(True)
                    return
                # Flash on
                self._set_eyes(True)
                if self._stop_event.wait(dur):
                    return
                # Flash off (element gap — skip after last element)
                if elem_idx < len(durations) - 1:
                    self._set_eyes(False)
                    if self._stop_event.wait(_SOS_ELEM_GAP):
                        self._set_eyes(True)
                        return
            # Letter gap (skip after last letter)
            if letter_idx < len(_SOS_PATTERN) - 1:
                self._set_eyes(False)
                if self._stop_event.wait(_SOS_LETTER_GAP):
                    self._set_eyes(True)
                    return
        # End-of-SOS pause before repeating
        self._set_eyes(False)
        self._stop_event.wait(_SOS_CYCLE_GAP)
        self._set_eyes(True)

    def _blink_alternate(self) -> None:
        """Blink left eye, then right eye, alternating — standby mode for GPIO eyes."""
        for eye_fn in (self._set_left, self._set_right):
            if self._stop_event.is_set() or self._solid or not self._standby:
                self._set_eyes(True)
                return
            self._set_eyes(False)
            eye_fn(True)  # turn on just one eye
            if self._stop_event.wait(self.blink_duration * 3):
                self._set_eyes(True)
                return
            self._set_eyes(False)
            if self._stop_event.wait(self.blink_duration):
                self._set_eyes(True)
                return
        self._set_eyes(True)
        next_alt = random.uniform(self.min_interval * 0.5, self.max_interval * 0.5)
        self._stop_event.wait(next_alt)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()
        self._set_eyes(False)
        if self._neopixels:
            self.ctrl.deinit()
        else:
            self._left_eye.deinit()
            self._right_eye.deinit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-neopixels",
        action="store_true",
        default=False,
        help="Use original single-colour GPIO eyes instead of NeoPixels",
    )
    args = parser.parse_args()

    e = Eyes(neopixels=not args.no_neopixels)
    try:
        while True:
            time.sleep(1)
    finally:
        e.stop()
