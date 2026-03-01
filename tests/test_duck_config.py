"""
Tests for DuckConfig – configuration parsing and defaults.
"""

import json
import pytest
from open_duck_mini_runtime.duck_config import DuckConfig

# ---------------------------------------------------------------------------
# Basic defaults
# ---------------------------------------------------------------------------


def test_duck_config_defaults(tmp_duck_config):
    cfg = DuckConfig(config_json_path=tmp_duck_config, ignore_default=True)
    assert cfg.start_paused is False
    assert cfg.imu_upside_down is False
    assert cfg.phase_frequency_factor_offset == 0.0
    assert cfg.controller_type == "xbox"
    assert cfg.eyes is False
    assert cfg.projector is False
    assert cfg.antennas is False
    assert cfg.speaker is False
    assert cfg.microphone is False
    assert cfg.camera is False


def test_duck_config_full(full_duck_config):
    cfg = DuckConfig(config_json_path=full_duck_config, ignore_default=False)
    assert cfg.start_paused is True
    assert cfg.imu_upside_down is True
    assert cfg.phase_frequency_factor_offset == pytest.approx(0.5)
    assert cfg.controller_type == "dualsense"
    assert cfg.eyes is True
    assert cfg.projector is True
    assert cfg.antennas is True
    assert cfg.speaker is True
    assert cfg.microphone is True
    assert cfg.camera is True


# ---------------------------------------------------------------------------
# controller_type field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctype", ["xbox", "dualsense", "generic_usb", "keyboard"])
def test_controller_type_parsed(tmp_path, ctype):
    cfg_data = {"controller_type": ctype}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg_data))
    cfg = DuckConfig(config_json_path=str(p), ignore_default=True)
    assert cfg.controller_type == ctype


def test_controller_type_defaults_to_xbox(tmp_path):
    """If controller_type is absent from JSON the default is 'xbox'."""
    p = tmp_path / "cfg.json"
    p.write_text("{}")
    cfg = DuckConfig(config_json_path=str(p), ignore_default=True)
    assert cfg.controller_type == "xbox"


# ---------------------------------------------------------------------------
# Missing config file → defaults
# ---------------------------------------------------------------------------


def test_duck_config_missing_file(tmp_path):
    nonexistent = str(tmp_path / "does_not_exist.json")
    cfg = DuckConfig(config_json_path=nonexistent, ignore_default=True)
    # Should not raise; should use defaults
    assert cfg.controller_type == "xbox"
    assert cfg.start_paused is False


# ---------------------------------------------------------------------------
# joints_offsets
# ---------------------------------------------------------------------------


def test_joints_offsets_parsed(full_duck_config):
    cfg = DuckConfig(config_json_path=full_duck_config, ignore_default=False)
    assert cfg.joints_offset["left_hip_yaw"] == pytest.approx(0.1)
    assert cfg.joints_offset["left_hip_roll"] == pytest.approx(-0.1)


def test_joints_offsets_default_zero(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text("{}")
    cfg = DuckConfig(config_json_path=str(p), ignore_default=True)
    for joint, value in cfg.joints_offset.items():
        assert value == pytest.approx(0.0), f"{joint} should default to 0.0"


# ---------------------------------------------------------------------------
# Expression features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "feature", ["eyes", "projector", "antennas", "speaker", "microphone", "camera"]
)
def test_expression_feature_individually(tmp_path, feature):
    cfg_data = {"expression_features": {feature: True}}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg_data))
    cfg = DuckConfig(config_json_path=str(p), ignore_default=True)
    assert getattr(cfg, feature) is True
    # Other features should remain False
    for other in ["eyes", "projector", "antennas", "speaker", "microphone", "camera"]:
        if other != feature:
            assert getattr(cfg, other) is False
