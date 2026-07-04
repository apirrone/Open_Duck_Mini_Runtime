"""
Hardware integration tests for motor EEPROM configuration.

Verifies that all 14 motors match the expected config (v2 source of truth).
If a motor fails, run tools/batch_reconfigure.py to apply the correct values.

Run with:
    uv run pytest -m hardware
"""

import pytest
from pypot.feetech import FeetechSTS3215IO

JOINTS = [
    ("left_hip_yaw", 20),
    ("left_hip_roll", 21),
    ("left_hip_pitch", 22),
    ("left_knee", 23),
    ("left_ankle", 24),
    ("neck_pitch", 30),
    ("head_pitch", 31),
    ("head_yaw", 32),
    ("head_roll", 33),
    ("right_hip_yaw", 10),
    ("right_hip_roll", 11),
    ("right_hip_pitch", 12),
    ("right_knee", 13),
    ("right_ankle", 14),
]

EXPECTED = {
    "return_delay_time": 0,  # delays > 0 break SYNC_READ
    "response_status_level": 1,  # 2 = respond to all commands, breaks SYNC_WRITE
    "mode": 0,
    "maximum_acceleration": 0,
    "acceleration": 0,
    "P_coefficient": 32,
    "I_coefficient": 0,
    "D_coefficient": 0,
}

USB_PORT = "/dev/ttyACM0"
BAUD_RATE = 1_000_000


@pytest.fixture(scope="module")
def motor_io():
    try:
        io = FeetechSTS3215IO(USB_PORT, baudrate=BAUD_RATE, use_sync_read=False)
    except Exception as exc:
        pytest.skip(f"Cannot open {USB_PORT}: {exc}", allow_module_level=True)
    return io


def _read_config(io, motor_id):
    return {
        "return_delay_time": io.get_return_delay_time([motor_id])[0],
        "response_status_level": io.get_response_status_level([motor_id])[0],
        "mode": io.get_mode([motor_id])[0],
        "maximum_acceleration": io.get_maximum_acceleration([motor_id])[0],
        "acceleration": io.get_acceleration([motor_id])[0],
        "P_coefficient": io.get_P_coefficient([motor_id])[0],
        "I_coefficient": io.get_I_coefficient([motor_id])[0],
        "D_coefficient": io.get_D_coefficient([motor_id])[0],
    }


@pytest.mark.hardware
@pytest.mark.parametrize("joint_name,motor_id", JOINTS)
def test_motor_config(motor_io, joint_name, motor_id):
    actual = _read_config(motor_io, motor_id)
    mismatches = {
        reg: f"got {actual[reg]}, expected {exp}"
        for reg, exp in EXPECTED.items()
        if actual[reg] != exp
    }
    assert not mismatches, (
        f"{joint_name} (ID {motor_id}) config mismatch:\n"
        + "\n".join(f"  {reg}: {detail}" for reg, detail in mismatches.items())
        + "\nRun tools/batch_reconfigure.py to fix."
    )
