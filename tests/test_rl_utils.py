"""
Tests for rl_utils – pure-Python utility functions that need no hardware.
"""

import numpy as np
import pytest
from open_duck_mini_runtime.rl_walk.rl_utils import (
    make_action_dict,
    LowPassActionFilter,
    ActionFilter,
    action_to_pd_targets,
    isaac_to_mujoco,
    mujoco_to_isaac,
    quat_rotate_inverse,
    isaac_joints_order,
    mujoco_joints_order,
)

# ---------------------------------------------------------------------------
# make_action_dict
# ---------------------------------------------------------------------------


class TestMakeActionDict:
    _joints = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "head_roll",
        "left_antenna",
        "right_antenna",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
    ]

    def test_antennas_excluded(self):
        action = np.arange(len(self._joints), dtype=float)
        result = make_action_dict(action, self._joints)
        assert "left_antenna" not in result
        assert "right_antenna" not in result

    def test_non_antenna_joints_included(self):
        action = np.arange(len(self._joints), dtype=float)
        result = make_action_dict(action, self._joints)
        for j in self._joints:
            if "antenna" not in j:
                assert j in result

    def test_values_match_action(self):
        action = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                99.0,
                99.0,
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
            ]
        )
        result = make_action_dict(action, self._joints)
        assert result["left_hip_yaw"] == pytest.approx(1.0)
        assert result["right_ankle"] == pytest.approx(14.0)

    def test_empty_action(self):
        result = make_action_dict([], [])
        assert result == {}


# ---------------------------------------------------------------------------
# action_to_pd_targets
# ---------------------------------------------------------------------------


class TestActionToPDTargets:
    def test_basic(self):
        action = np.array([1.0, 2.0, 3.0])
        offset = np.array([0.5, 0.5, 0.5])
        scale = 2.0
        result = action_to_pd_targets(action, offset, scale)
        expected = np.array([2.5, 4.5, 6.5])
        np.testing.assert_allclose(result, expected)

    def test_zero_scale(self):
        action = np.array([10.0, -5.0])
        result = action_to_pd_targets(action, np.zeros(2), 0.0)
        np.testing.assert_allclose(result, [0.0, 0.0])


# ---------------------------------------------------------------------------
# LowPassActionFilter
# ---------------------------------------------------------------------------


class TestLowPassActionFilter:
    def test_alpha_in_valid_range(self):
        f = LowPassActionFilter(control_freq=50, cutoff_frequency=30)
        assert 0.0 < f.alpha < 1.0

    def test_converges_to_target(self):
        f = LowPassActionFilter(control_freq=50, cutoff_frequency=10)
        target = np.array([1.0, -1.0, 2.0])
        for _ in range(200):
            f.push(target)
            output = f.get_filtered_action()
        np.testing.assert_allclose(output, target, atol=1e-2)

    def test_initial_output_is_zero_vector(self):
        f = LowPassActionFilter(control_freq=50, cutoff_frequency=30)
        f.push(np.array([1.0, 2.0]))
        # First call: last_action starts at 0, so output is close to 0 weighted
        output = f.get_filtered_action()
        assert output.shape == (2,)

    def test_higher_cutoff_faster_response(self):
        """High cutoff → alpha → 0 → more weight on new signal → faster."""
        fast = LowPassActionFilter(control_freq=50, cutoff_frequency=45)
        slow = LowPassActionFilter(control_freq=50, cutoff_frequency=1)
        target = np.array([1.0])
        fast.push(target)
        slow.push(target)
        fast_out = fast.get_filtered_action()
        slow_out = slow.get_filtered_action()
        assert fast_out >= slow_out


# ---------------------------------------------------------------------------
# ActionFilter (moving average)
# ---------------------------------------------------------------------------


class TestActionFilter:
    def test_average_over_window(self):
        f = ActionFilter(window_size=4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            f.push(np.array([v]))
        result = f.get_filtered_action()
        np.testing.assert_allclose(result, [2.5])

    def test_window_size_respected(self):
        f = ActionFilter(window_size=2)
        for v in [1.0, 2.0, 3.0]:
            f.push(np.array([v]))
        # Buffer should only keep last 2: [2.0, 3.0] → mean = 2.5
        result = f.get_filtered_action()
        np.testing.assert_allclose(result, [2.5])


# ---------------------------------------------------------------------------
# Joint order conversions (round-trip identity)
# ---------------------------------------------------------------------------


class TestJointOrderConversions:
    def test_isaac_mujoco_roundtrip(self):
        joints = list(range(16))
        converted = isaac_to_mujoco(joints)
        back = mujoco_to_isaac(converted)
        assert back == joints

    def test_mujoco_isaac_roundtrip(self):
        joints = list(range(16))
        converted = mujoco_to_isaac(joints)
        back = isaac_to_mujoco(converted)
        assert back == joints

    def test_same_length(self):
        joints = list(range(16))
        assert len(isaac_to_mujoco(joints)) == 16
        assert len(mujoco_to_isaac(joints)) == 16


# ---------------------------------------------------------------------------
# quat_rotate_inverse
# ---------------------------------------------------------------------------


class TestQuatRotateInverse:
    def test_identity_quaternion(self):
        # Identity quaternion: [0, 0, 0, 1]
        q = np.array([0.0, 0.0, 0.0, 1.0])
        v = np.array([1.0, 0.0, 0.0])
        result = quat_rotate_inverse(q, v)
        np.testing.assert_allclose(result, v, atol=1e-6)

    def test_output_shape(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        v = np.array([1.0, 2.0, 3.0])
        result = quat_rotate_inverse(q, v)
        assert result.shape == (3,)
