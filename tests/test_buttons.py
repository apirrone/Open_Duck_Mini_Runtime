"""
Tests for the Button / Buttons state machine.
"""

import time
import pytest
from unittest.mock import patch
from open_duck_mini_runtime.controller.buttons import Button, Buttons

# ---------------------------------------------------------------------------
# Button – single button state machine
# ---------------------------------------------------------------------------


class TestButton:
    def test_initial_state(self):
        btn = Button()
        assert btn.is_pressed is False
        assert btn.triggered is False
        assert btn.released is True

    def test_press_triggers(self):
        btn = Button()
        # Last pressed was a long time ago so timeout is cleared
        btn.last_pressed_time = time.time() - 1.0
        btn.update(True)
        assert btn.is_pressed is True
        assert btn.triggered is True

    def test_hold_does_not_retrigger_within_timeout(self):
        btn = Button()
        btn.last_pressed_time = time.time() - 1.0
        btn.update(True)
        assert btn.triggered is True
        # Keep holding – should not trigger again within timeout
        btn.update(True)
        assert btn.triggered is False

    def test_release_sets_released_flag(self):
        btn = Button()
        btn.last_pressed_time = time.time() - 1.0
        btn.update(True)  # press
        btn.update(False)  # release
        assert btn.is_pressed is False
        assert btn.released is True

    def test_retrigger_after_release(self):
        btn = Button()
        btn.last_pressed_time = time.time() - 1.0
        btn.update(True)  # press 1
        btn.update(False)  # release

        # Simulate enough time has passed for another trigger
        btn.last_pressed_time = time.time() - 1.0
        btn.update(True)  # press 2
        assert btn.triggered is True

    def test_not_triggered_when_never_released(self):
        btn = Button()
        btn.last_pressed_time = time.time() - 1.0
        btn.update(True)  # first press → triggers
        btn.update(True)  # still held → released=False, should NOT trigger
        assert btn.triggered is False

    def test_not_pressed_initially(self):
        btn = Button()
        btn.update(False)
        assert btn.is_pressed is False
        assert btn.triggered is False


# ---------------------------------------------------------------------------
# Buttons – aggregate
# ---------------------------------------------------------------------------


class TestButtons:
    def _make_buttons(self, **pressed):
        """Helper: build Buttons with given keys set to True."""
        defaults = dict(
            A=False,
            B=False,
            X=False,
            Y=False,
            LB=False,
            RB=False,
            dpad_up=False,
            dpad_down=False,
        )
        defaults.update(pressed)
        b = Buttons()
        # Age the last_pressed_time so triggers fire
        for attr in ["A", "B", "X", "Y", "LB", "RB", "dpad_up", "dpad_down"]:
            getattr(b, attr).last_pressed_time = time.time() - 1.0
        b.update(
            defaults["A"],
            defaults["B"],
            defaults["X"],
            defaults["Y"],
            defaults["LB"],
            defaults["RB"],
            defaults["dpad_up"],
            defaults["dpad_down"],
        )
        return b

    def test_all_released_initially(self):
        b = Buttons()
        assert b.A.is_pressed is False
        assert b.B.is_pressed is False
        assert b.X.is_pressed is False
        assert b.Y.is_pressed is False
        assert b.LB.is_pressed is False
        assert b.RB.is_pressed is False
        assert b.dpad_up.is_pressed is False
        assert b.dpad_down.is_pressed is False

    def test_A_pressed(self):
        b = self._make_buttons(A=True)
        assert b.A.is_pressed is True
        assert b.A.triggered is True
        assert b.B.is_pressed is False

    def test_LB_held_not_retrigger(self):
        b = Buttons()
        b.LB.last_pressed_time = time.time() - 1.0
        b.update(False, False, False, False, True, False, False, False)
        assert b.LB.is_pressed is True
        assert b.LB.triggered is True
        # Hold without releasing
        b.update(False, False, False, False, True, False, False, False)
        assert b.LB.is_pressed is True
        assert b.LB.triggered is False  # no re-trigger

    def test_dpad_up_and_down_independent(self):
        b = self._make_buttons(dpad_up=True)
        assert b.dpad_up.is_pressed is True
        assert b.dpad_down.is_pressed is False

    def test_multiple_buttons_simultaneously(self):
        b = self._make_buttons(A=True, LB=True)
        assert b.A.is_pressed is True
        assert b.LB.is_pressed is True
        assert b.B.is_pressed is False
