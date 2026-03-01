"""
Hardware integration tests for servo presence.

These tests require a physically connected robot.

Run with:
    uv run pytest -m hardware

Skip during normal CI:
    uv run pytest -m "not hardware"
"""

import pytest

# Joint name → servo ID, matching HWI.joints order
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

USB_PORT = "/dev/ttyACM0"
BAUD_RATE = 1_000_000


@pytest.fixture(scope="module")
def servo_io():
    """Open a rustypot feetech connection, skip the module if hardware is absent."""
    try:
        import rustypot
    except ImportError:
        pytest.skip("rustypot not installed", allow_module_level=True)

    try:
        io = rustypot.feetech(USB_PORT, BAUD_RATE)
    except Exception as exc:
        pytest.skip(
            f"Cannot open {USB_PORT} at {BAUD_RATE} baud: {exc}",
            allow_module_level=True,
        )

    return io


@pytest.mark.hardware
@pytest.mark.parametrize("joint_name,joint_id", JOINTS)
def test_servo_responds(servo_io, joint_name: str, joint_id: int):
    """Each servo should return a numeric present-position when polled."""
    try:
        result = servo_io.read_present_position([joint_id])
    except Exception as exc:
        pytest.fail(
            f"Servo {joint_name!r} (ID {joint_id}) raised an exception: {exc}"
        )

    assert result is not None, (
        f"Servo {joint_name!r} (ID {joint_id}) returned None"
    )
    assert len(result) == 1, (
        f"Expected 1 position value for {joint_name!r} (ID {joint_id}), got {len(result)}"
    )
    position = float(result[0])
    # A sane present-position is roughly in [-2π, 2π] radians for these joints.
    assert -10.0 <= position <= 10.0, (
        f"Servo {joint_name!r} (ID {joint_id}) returned implausible position {position:.4f} rad"
    )


@pytest.mark.hardware
def test_all_servos_present(servo_io):
    """Bulk check: all 14 servos respond without error."""
    all_ids = [joint_id for _, joint_id in JOINTS]
    failed = []
    for joint_name, joint_id in JOINTS:
        try:
            result = servo_io.read_present_position([joint_id])
            if result is None or len(result) == 0:
                failed.append(f"{joint_name} (ID {joint_id}): no response")
        except Exception as exc:
            failed.append(f"{joint_name} (ID {joint_id}): {exc}")

    assert not failed, (
        f"{len(failed)}/{len(all_ids)} servo(s) did not respond:\n"
        + "\n".join(f"  - {f}" for f in failed)
    )
