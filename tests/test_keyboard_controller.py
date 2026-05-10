"""
Tests for KeyboardController – command generation and key-state logic.

The stdin reader thread is mocked so that no real terminal access occurs
during tests.
"""

import time
import pytest
from unittest.mock import patch, MagicMock

_mod = pytest.importorskip("open_duck_mini_runtime.controller.keyboard_controller")
KeyboardController = _mod.KeyboardController
_KeyState = _mod._KeyState
X_RANGE = _mod.X_RANGE
Y_RANGE = _mod.Y_RANGE
YAW_RANGE = _mod.YAW_RANGE

# ---------------------------------------------------------------------------
# _KeyState – per-key hold tracking
# ---------------------------------------------------------------------------


class TestKeyState:
    def test_not_held_before_press(self):
        ks = _KeyState()
        assert ks.is_held() is False

    def test_held_immediately_after_press(self):
        ks = _KeyState()
        ks.press()
        assert ks.is_held() is True

    def test_expires_after_timeout(self):
        ks = _KeyState()
        ks.press()
        # Manually backdate the timestamp
        ks._last_seen = time.monotonic() - 1.0
        assert ks.is_held() is False

    def test_custom_timeout(self):
        ks = _KeyState()
        ks.press()
        ks._last_seen = time.monotonic() - 0.05
        assert ks.is_held(timeout=0.10) is True
        assert ks.is_held(timeout=0.01) is False


# ---------------------------------------------------------------------------
# KeyboardController – initialise with mocked thread
# ---------------------------------------------------------------------------


@pytest.fixture
def keyboard_ctrl():
    """Return a KeyboardController whose stdin thread is mocked out.
    All button last_pressed_time values are pre-aged so trigger checks pass immediately.
    """
    with patch("open_duck_mini_runtime.controller.keyboard_controller.Thread") as mock_thread_cls:
        mock_thread_cls.return_value = MagicMock()
        ctrl = KeyboardController(command_freq=20)
    # Pre-age all buttons so the debounce timeout doesn't block triggering in tests
    for attr in ["A", "B", "X", "Y", "LB", "RB", "dpad_up", "dpad_down"]:
        getattr(ctrl.buttons, attr).last_pressed_time = time.time() - 1.0
    return ctrl


# ---------------------------------------------------------------------------
# Movement command generation
# ---------------------------------------------------------------------------


class TestKeyboardControllerCommands:
    def test_idle_commands_are_zero(self, keyboard_ctrl):
        cmds, buttons, lt, rt = keyboard_ctrl.get_last_command()
        assert cmds[0] == pytest.approx(0.0)  # lin_vel_x
        assert cmds[1] == pytest.approx(0.0)  # lin_vel_y
        assert cmds[2] == pytest.approx(0.0)  # ang_vel
        assert lt == pytest.approx(0.0)
        assert rt == pytest.approx(0.0)

    def test_forward_w(self, keyboard_ctrl):
        keyboard_ctrl._keys["w"].press()
        cmds, _, _, _ = keyboard_ctrl.get_last_command()
        assert cmds[0] == pytest.approx(X_RANGE[1])

    def test_backward_s(self, keyboard_ctrl):
        keyboard_ctrl._keys["s"].press()
        cmds, _, _, _ = keyboard_ctrl.get_last_command()
        assert cmds[0] == pytest.approx(X_RANGE[0])

    def test_strafe_left_q(self, keyboard_ctrl):
        keyboard_ctrl._keys["q"].press()
        cmds, _, _, _ = keyboard_ctrl.get_last_command()
        assert cmds[1] == pytest.approx(Y_RANGE[1])

    def test_strafe_right_e(self, keyboard_ctrl):
        keyboard_ctrl._keys["e"].press()
        cmds, _, _, _ = keyboard_ctrl.get_last_command()
        assert cmds[1] == pytest.approx(Y_RANGE[0])

    def test_turn_left_a(self, keyboard_ctrl):
        keyboard_ctrl._keys["a"].press()
        cmds, _, _, _ = keyboard_ctrl.get_last_command()
        assert cmds[2] == pytest.approx(YAW_RANGE[1])

    def test_turn_right_d(self, keyboard_ctrl):
        keyboard_ctrl._keys["d"].press()
        cmds, _, _, _ = keyboard_ctrl.get_last_command()
        assert cmds[2] == pytest.approx(YAW_RANGE[0])

    def test_forward_backward_cancel(self, keyboard_ctrl):
        """W+S simultaneously should clamp to net-zero (or at range boundary)."""
        keyboard_ctrl._keys["w"].press()
        keyboard_ctrl._keys["s"].press()
        cmds, _, _, _ = keyboard_ctrl.get_last_command()
        # X_RANGE[1] + X_RANGE[0] = 0.15 + (-0.15) = 0.0
        assert cmds[0] == pytest.approx(0.0)

    def test_expired_key_stops_motion(self, keyboard_ctrl):
        keyboard_ctrl._keys["w"].press()
        keyboard_ctrl._keys["w"]._last_seen = time.monotonic() - 1.0  # expire
        cmds, _, _, _ = keyboard_ctrl.get_last_command()
        assert cmds[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Button events via event queue
# ---------------------------------------------------------------------------


class TestKeyboardControllerButtons:
    def test_space_triggers_A_button(self, keyboard_ctrl):
        keyboard_ctrl._event_queue.put(" ")
        _, buttons, _, _ = keyboard_ctrl.get_last_command()
        assert buttons.A.triggered is True

    def test_x_key_triggers_X_button(self, keyboard_ctrl):
        keyboard_ctrl._event_queue.put("x")
        _, buttons, _, _ = keyboard_ctrl.get_last_command()
        assert buttons.X.triggered is True

    def test_b_key_triggers_B_button(self, keyboard_ctrl):
        keyboard_ctrl._event_queue.put("b")
        _, buttons, _, _ = keyboard_ctrl.get_last_command()
        assert buttons.B.triggered is True

    def test_p_key_triggers_Y_button(self, keyboard_ctrl):
        keyboard_ctrl._event_queue.put("p")
        _, buttons, _, _ = keyboard_ctrl.get_last_command()
        assert buttons.Y.triggered is True

    def test_l_held_activates_LB(self, keyboard_ctrl):
        keyboard_ctrl._keys["l"].press()
        _, buttons, _, _ = keyboard_ctrl.get_last_command()
        assert buttons.LB.is_pressed is True

    def test_l_expired_deactivates_LB(self, keyboard_ctrl):
        keyboard_ctrl._keys["l"].press()
        keyboard_ctrl._keys["l"]._last_seen = time.monotonic() - 1.0
        _, buttons, _, _ = keyboard_ctrl.get_last_command()
        assert buttons.LB.is_pressed is False


# ---------------------------------------------------------------------------
# Commands list is always length 7
# ---------------------------------------------------------------------------


def test_commands_length(keyboard_ctrl):
    cmds, _, _, _ = keyboard_ctrl.get_last_command()
    assert len(cmds) == 7


# ---------------------------------------------------------------------------
# print_controls – smoke test
# ---------------------------------------------------------------------------


def test_print_controls_does_not_raise(capsys):
    KeyboardController.print_controls()
    captured = capsys.readouterr()
    assert "WASD" in captured.out or "W" in captured.out
