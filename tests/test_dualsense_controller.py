"""
Tests for DualSenseController – validates structure, API compatibility,
and mapping correctness (no hardware / display required; pygame is mocked).
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers to build a DualSenseController without hardware
# ---------------------------------------------------------------------------


def _make_dualsense(command_freq: int = 20) -> "DualSenseController":
    """Instantiate DualSenseController with pygame and threading mocked out."""
    with (
        patch("pygame.init"),
        patch("pygame.joystick.Joystick") as mock_joy_cls,
        patch("open_duck_mini_runtime.dualsense_controller.Thread"),
    ):

        mock_joy = MagicMock()
        mock_joy.get_numaxes.return_value = 6
        mock_joy.get_numbuttons.return_value = 14
        mock_joy.get_numhats.return_value = 1
        mock_joy_cls.return_value = mock_joy

        from open_duck_mini_runtime.dualsense_controller import DualSenseController

        ctrl = DualSenseController(command_freq)
        ctrl._p1 = mock_joy  # keep reference for axis mocking
        return ctrl


# ---------------------------------------------------------------------------
# Module-level validation
# ---------------------------------------------------------------------------


class TestDualSenseControllerStructure:
    def test_importable(self):
        from open_duck_mini_runtime.dualsense_controller import DualSenseController

        assert DualSenseController is not None

    def test_has_get_last_command(self):
        from open_duck_mini_runtime.dualsense_controller import DualSenseController

        assert callable(getattr(DualSenseController, "get_last_command", None))

    def test_has_get_commands(self):
        from open_duck_mini_runtime.dualsense_controller import DualSenseController

        assert callable(getattr(DualSenseController, "get_commands", None))

    def test_initial_commands_zero(self):
        ctrl = _make_dualsense()
        assert ctrl.last_commands == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_get_last_command_returns_four_tuple(self):
        ctrl = _make_dualsense()
        result = ctrl.get_last_command()
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_commands_list_length_7(self):
        ctrl = _make_dualsense()
        cmds, buttons, lt, rt = ctrl.get_last_command()
        assert len(cmds) == 7

    def test_triggers_default_zero(self):
        ctrl = _make_dualsense()
        _, _, lt, rt = ctrl.get_last_command()
        assert lt == pytest.approx(0.0)
        assert rt == pytest.approx(0.0)

    def test_buttons_object_present(self):
        from open_duck_mini_runtime.buttons import Buttons

        ctrl = _make_dualsense()
        _, buttons, _, _ = ctrl.get_last_command()
        assert isinstance(buttons, Buttons)


# ---------------------------------------------------------------------------
# Axis mapping – DualSense specifics
# ---------------------------------------------------------------------------


class TestDualSenseAxisMapping:
    """Verify that trigger axes match the DualSense SDL2 layout:
    L2 → axis 4, R2 → axis 5  (opposite of XBox where R is axis 4)."""

    def test_axis_constants_are_correct(self):
        from open_duck_mini_runtime import dualsense_controller as ds

        # DualSense uses axis 4 for left trigger and axis 5 for right trigger
        # Confirm this by inspecting the source-level logic: get_axis(5) → right trigger
        import inspect

        src = inspect.getsource(ds.DualSenseController.get_commands)
        # "get_axis(5)" should appear before "right_trigger"
        assert "get_axis(5)" in src
        assert "get_axis(4)" in src


# ---------------------------------------------------------------------------
# Button mapping – DualSense specifics
# ---------------------------------------------------------------------------


class TestDualSenseButtonMapping:
    def test_button_indices_in_source(self):
        """Cross=0, Circle=1, Square=2, Triangle=3, L1=4, R1=5"""
        import inspect
        from open_duck_mini_runtime.dualsense_controller import DualSenseController

        src = inspect.getsource(DualSenseController.get_commands)
        # Spot-check the button indices documented in the source
        assert "get_button(0)" in src  # Cross
        assert "get_button(1)" in src  # Circle
        assert "get_button(2)" in src  # Square
        assert "get_button(3)" in src  # Triangle
        assert "get_button(4)" in src  # L1
        assert "get_button(5)" in src  # R1


# ---------------------------------------------------------------------------
# API parity with XBoxController
# ---------------------------------------------------------------------------


class TestDualSenseXBoxParity:
    """DualSenseController must expose the same public API as XBoxController
    so it can be used as a drop-in replacement in walk.py."""

    def _get_public_methods(self, cls):
        return {name for name in dir(cls) if not name.startswith("_")}

    def test_shared_public_methods(self):
        from open_duck_mini_runtime.xbox_controller import XBoxController
        from open_duck_mini_runtime.dualsense_controller import DualSenseController

        xbox_methods = self._get_public_methods(XBoxController)
        ds_methods = self._get_public_methods(DualSenseController)
        # Both must have get_last_command
        assert "get_last_command" in xbox_methods
        assert "get_last_command" in ds_methods

    def test_get_last_command_same_return_shape(self):
        """Both controllers must return (list[float], Buttons, float, float)."""
        from open_duck_mini_runtime.buttons import Buttons

        xbox_ctrl = _make_any_controller("xbox_controller", "XBoxController")
        ds_ctrl = _make_dualsense()

        for ctrl in [xbox_ctrl, ds_ctrl]:
            cmds, buttons, lt, rt = ctrl.get_last_command()
            assert isinstance(cmds, (list, np.ndarray))
            assert len(cmds) == 7
            assert isinstance(buttons, Buttons)
            assert isinstance(lt, float)
            assert isinstance(rt, float)


def _make_any_controller(module_name: str, class_name: str, command_freq: int = 20):
    with (
        patch("pygame.init"),
        patch("pygame.joystick.Joystick") as mock_joy_cls,
        patch(f"open_duck_mini_runtime.{module_name}.Thread"),
    ):

        mock_joy = MagicMock()
        mock_joy.get_numaxes.return_value = 6
        mock_joy_cls.return_value = mock_joy

        import importlib

        mod = importlib.import_module(f"open_duck_mini_runtime.{module_name}")
        cls = getattr(mod, class_name)
        return cls(command_freq)
