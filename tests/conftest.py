"""
pytest configuration and shared fixtures for Open Duck Mini Runtime tests.
"""

import json
import os
import pytest
import tempfile


@pytest.fixture
def tmp_duck_config(tmp_path):
    """Write a minimal valid duck_config.json to a temp directory and return its path."""

    cfg = {
        "start_paused": False,
        "imu_upside_down": False,
        "phase_frequency_factor_offset": 0.0,
        "controller_type": "xbox",
        "expression_features": {
            "eyes": False,
            "projector": False,
            "antennas": False,
            "speaker": False,
            "microphone": False,
            "camera": False,
        },
        "joints_offsets": {
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": 0.0,
            "left_knee": 0.0,
            "left_ankle": 0.0,
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": 0.0,
            "right_knee": 0.0,
            "right_ankle": 0.0,
        },
    }
    config_path = tmp_path / "duck_config.json"
    config_path.write_text(json.dumps(cfg))
    return str(config_path)


@pytest.fixture
def full_duck_config(tmp_path):
    """Return path to a duck_config with all expression features enabled."""

    cfg = {
        "start_paused": True,
        "imu_upside_down": True,
        "phase_frequency_factor_offset": 0.5,
        "controller_type": "dualsense",
        "expression_features": {
            "eyes": True,
            "projector": True,
            "antennas": True,
            "speaker": True,
            "microphone": True,
            "camera": True,
        },
        "joints_offsets": {
            "left_hip_yaw": 0.1,
            "left_hip_roll": -0.1,
            "left_hip_pitch": 0.2,
            "left_knee": -0.2,
            "left_ankle": 0.05,
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "right_hip_yaw": -0.1,
            "right_hip_roll": 0.1,
            "right_hip_pitch": -0.2,
            "right_knee": 0.2,
            "right_ankle": -0.05,
        },
    }
    config_path = tmp_path / "full_duck_config.json"
    config_path.write_text(json.dumps(cfg))
    return str(config_path)
