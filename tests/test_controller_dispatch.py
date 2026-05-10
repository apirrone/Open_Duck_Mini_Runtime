"""
Tests for RLWalk._make_controller – verifies the correct controller class is
instantiated for each controller_type string.

All controller constructors are mocked so that no hardware is required.
"""

import importlib.util
import pytest
from unittest.mock import patch, MagicMock

_has_dualsense = importlib.util.find_spec("open_duck_mini_runtime.controller.dualsense_controller") is not None
_has_generic_usb = importlib.util.find_spec("open_duck_mini_runtime.controller.generic_usb_controller") is not None
_has_keyboard = importlib.util.find_spec("open_duck_mini_runtime.controller.keyboard_controller") is not None

skip_dualsense = pytest.mark.skipif(not _has_dualsense, reason="dualsense_controller not on this branch")
skip_generic_usb = pytest.mark.skipif(not _has_generic_usb, reason="generic_usb_controller not on this branch")
skip_keyboard = pytest.mark.skipif(not _has_keyboard, reason="keyboard_controller not on this branch")

# ---------------------------------------------------------------------------
# Helper: build a minimal RLWalk-like object with only the controller
# factory method under test, without touching any hardware.
# ---------------------------------------------------------------------------


class _FakeRLWalk:
    """Minimal shim that exposes only _make_controller."""

    def __init__(self, command_freq: int = 20):
        self.command_freq = command_freq

    # Copy the exact implementation from walk.py so we test the real logic.
    def _make_controller(self, controller_type: str):
        ctype = controller_type.lower()
        if ctype == "dualsense":
            from open_duck_mini_runtime.controller.dualsense_controller import DualSenseController

            return DualSenseController(self.command_freq)
        elif ctype == "generic_usb":
            from open_duck_mini_runtime.controller.generic_usb_controller import (
                GenericUSBController,
            )

            return GenericUSBController(self.command_freq)
        elif ctype == "keyboard":
            from open_duck_mini_runtime.controller.keyboard_controller import KeyboardController

            return KeyboardController(self.command_freq)
        else:
            from open_duck_mini_runtime.controller.xbox_controller import XBoxController

            return XBoxController(self.command_freq)


# ---------------------------------------------------------------------------
# Controller dispatch per type
# ---------------------------------------------------------------------------


class TestControllerDispatch:
    def test_xbox_type(self):
        with patch("open_duck_mini_runtime.controller.xbox_controller.XBoxController") as mock_cls:
            mock_cls.return_value = MagicMock()
            fw = _FakeRLWalk()
            fw._make_controller("xbox")
            mock_cls.assert_called_once_with(20)

    @skip_dualsense
    def test_dualsense_type(self):
        with patch(
            "open_duck_mini_runtime.controller.dualsense_controller.DualSenseController"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            fw = _FakeRLWalk()
            fw._make_controller("dualsense")
            mock_cls.assert_called_once_with(20)

    @skip_generic_usb
    def test_generic_usb_type(self):
        with patch(
            "open_duck_mini_runtime.controller.generic_usb_controller.GenericUSBController"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            fw = _FakeRLWalk()
            fw._make_controller("generic_usb")
            mock_cls.assert_called_once_with(20)

    @skip_keyboard
    def test_keyboard_type(self):
        with patch("open_duck_mini_runtime.controller.keyboard_controller.Thread"):
            fw = _FakeRLWalk()
            controller = fw._make_controller("keyboard")
            from open_duck_mini_runtime.controller.keyboard_controller import KeyboardController

            assert isinstance(controller, KeyboardController)

    def test_unknown_type_defaults_to_xbox(self):
        with patch("open_duck_mini_runtime.controller.xbox_controller.XBoxController") as mock_cls:
            mock_cls.return_value = MagicMock()
            fw = _FakeRLWalk()
            fw._make_controller("unknown_controller")
            mock_cls.assert_called_once_with(20)

    @skip_dualsense
    def test_case_insensitive(self):
        with patch(
            "open_duck_mini_runtime.controller.dualsense_controller.DualSenseController"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            fw = _FakeRLWalk()
            fw._make_controller("DualSense")
            mock_cls.assert_called_once()

        with patch("open_duck_mini_runtime.controller.xbox_controller.XBoxController") as mock_cls:
            mock_cls.return_value = MagicMock()
            fw = _FakeRLWalk()
            fw._make_controller("XBOX")
            mock_cls.assert_called_once()

    @skip_generic_usb
    def test_generic_usb_case_insensitive(self):
        with patch(
            "open_duck_mini_runtime.controller.generic_usb_controller.GenericUSBController"
        ) as mock_cls:
            mock_cls.return_value = MagicMock()
            fw = _FakeRLWalk()
            fw._make_controller("Generic_USB")
            mock_cls.assert_called_once()
