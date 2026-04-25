"""
Visual hardware integration tests for Open Duck Mini.

Moves each motor, cycles eye colors, tests the projector, and plays sounds
so a user can confirm everything works by watching/listening.

Run with:
    uv run pytest tests/test_hardware_visual.py -m hardware -s -v -p no:randomly

The -s flag is required so you see the prompts between test steps.
All tests are marked @pytest.mark.hardware and are skipped during normal CI.
"""

import time
import pytest

MOVE_DELTA_RAD = 0.15   # how far to nudge each joint from init (radians)
MOVE_HOLD_S = 0.5       # how long to hold the nudged position
HOME_SETTLE_S = 2.0     # time to wait after homing all joints
ANTENNA_HOLD_S = 0.7    # how long to hold each antenna position

# Joint names in the same order HWI uses — used for parametrize
JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

EYE_COLORS = ["red", "green", "blue", "yellow", "white"]

# Single repeatable sound used for all audio tests
SOUND_INDEX = 0  # beep1.wav (first alphabetically)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hwi():
    """Open the HWI (applies duck_config offsets), home to init, yield, then off."""
    try:
        from open_duck_mini_runtime.hwi import HWI
        from open_duck_mini_runtime.duck_config import DuckConfig
    except ImportError as exc:
        pytest.skip(f"HWI/DuckConfig not importable: {exc}")

    try:
        dc = DuckConfig()
        h = HWI(dc)
    except Exception as exc:
        pytest.skip(f"Cannot open HWI: {exc}")

    print("\n  Moving to init position (low kp)...")
    try:
        h.turn_on()
        time.sleep(HOME_SETTLE_S)
        print("  Homed.")
    except Exception as exc:
        pytest.skip(f"Could not home motors: {exc}")

    yield h

    # Return to init, then release torque
    try:
        h.set_position_all(h.init_pos)
        time.sleep(HOME_SETTLE_S)
    except Exception:
        pass
    try:
        h.turn_off()
        print("\n  Motors torque disabled.")
    except Exception:
        pass


@pytest.fixture(scope="module")
def led_ctrl():
    try:
        from open_duck_mini_runtime.led_controller import get_controller
        ctrl = get_controller()
    except Exception as exc:
        pytest.skip(f"LED controller unavailable: {exc}")

    yield ctrl

    ctrl.set_eyes_color("white")
    ctrl.set_eyes(True)
    ctrl.set_projector(False)


@pytest.fixture(scope="module")
def antennas():
    try:
        from open_duck_mini_runtime.antennas import Antennas
        ant = Antennas()
    except Exception as exc:
        pytest.skip(f"Antennas unavailable: {exc}")

    yield ant

    ant.stop()


@pytest.fixture(scope="module")
def sounds():
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        from open_duck_mini_runtime.sounds import Sounds
        snd = Sounds()
    except Exception as exc:
        pytest.skip(f"Sound system unavailable: {exc}")

    assert snd.wav_files, "No WAV files found in assets/"
    yield snd


# ---------------------------------------------------------------------------
# Motor tests
# ---------------------------------------------------------------------------


@pytest.mark.hardware
@pytest.mark.parametrize("joint_name", JOINT_NAMES)
def test_motor_moves(hwi, joint_name: str):
    """Nudge each joint +MOVE_DELTA_RAD from its calibrated init position and back."""
    init = hwi.init_pos[joint_name]
    target = init + MOVE_DELTA_RAD

    try:
        hwi.set_position(joint_name, target)
        time.sleep(MOVE_HOLD_S)
        hwi.set_position(joint_name, init)
        time.sleep(MOVE_HOLD_S)
    except Exception as exc:
        pytest.fail(f"{joint_name}: move command failed — {exc}")

    positions = hwi.get_present_positions()
    if positions is None:
        pytest.fail(f"{joint_name}: could not read present positions")

    joint_idx = list(hwi.joints.keys()).index(joint_name)
    final = float(positions[joint_idx])

    assert abs(final - init) < 0.12, (
        f"{joint_name}: did not return to init. "
        f"expected≈{init:.3f} actual={final:.3f}"
    )


# ---------------------------------------------------------------------------
# Eye tests
# ---------------------------------------------------------------------------


@pytest.mark.hardware
@pytest.mark.parametrize("color", EYE_COLORS)
def test_eye_color(led_ctrl, color: str):
    """Set both eyes to each color for 0.5 s — visually verify on the duck."""
    led_ctrl.set_eyes_color(color)
    led_ctrl.set_eyes(True)
    time.sleep(0.5)


@pytest.mark.hardware
def test_eye_blink(led_ctrl):
    """Cycle eyes on/off 4 times to verify the blink mechanism."""
    for _ in range(4):
        led_ctrl.set_eyes(False)
        time.sleep(0.15)
        led_ctrl.set_eyes(True)
        time.sleep(0.25)


@pytest.mark.hardware
def test_eye_standby_color(led_ctrl):
    """Show YELLOW (standby) for 1 s, then WHITE (normal)."""
    led_ctrl.set_eyes_color("yellow")
    led_ctrl.set_eyes(True)
    time.sleep(1.0)
    led_ctrl.set_eyes_color("white")
    led_ctrl.set_eyes(True)


@pytest.mark.hardware
def test_eye_stop_color(led_ctrl):
    """Show RED (stopped/off) for 1 s, then WHITE (normal)."""
    led_ctrl.set_eyes_color("red")
    led_ctrl.set_eyes(True)
    time.sleep(1.0)
    led_ctrl.set_eyes_color("white")
    led_ctrl.set_eyes(True)


# ---------------------------------------------------------------------------
# Projector test
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_projector(led_ctrl):
    """Flash the projector LED on/off three times."""
    for _ in range(3):
        led_ctrl.set_projector(True)
        time.sleep(0.4)
        led_ctrl.set_projector(False)
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# Antenna tests
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_antennas_sweep(antennas):
    """Sweep antennas through min, max, and neutral to confirm full range."""
    # Neutral
    antennas.set_position_left(0.0)
    antennas.set_position_right(0.0)
    time.sleep(ANTENNA_HOLD_S)

    # Max deflection
    antennas.set_position_left(1.0)
    antennas.set_position_right(1.0)
    time.sleep(ANTENNA_HOLD_S)

    # Min deflection
    antennas.set_position_left(-1.0)
    antennas.set_position_right(-1.0)
    time.sleep(ANTENNA_HOLD_S)

    # Opposite directions
    antennas.set_position_left(1.0)
    antennas.set_position_right(-1.0)
    time.sleep(ANTENNA_HOLD_S)

    antennas.set_position_left(-1.0)
    antennas.set_position_right(1.0)
    time.sleep(ANTENNA_HOLD_S)

    # Return to neutral
    antennas.set_position_left(0.0)
    antennas.set_position_right(0.0)
    time.sleep(ANTENNA_HOLD_S)


# ---------------------------------------------------------------------------
# Sound tests
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_sound_plays(sounds):
    """Play one repeatable sound (blocking) to confirm audio output."""
    from open_duck_mini_runtime.sounds import play_sound as _play_blocking
    path = sounds.wav_files[SOUND_INDEX]
    print(f"\n  Playing: {path.name}")
    _play_blocking(path, volume=sounds.volume)


@pytest.mark.hardware
def test_sound_non_blocking(sounds):
    """Confirm play_sound returns immediately (< 0.1 s) without blocking."""
    from open_duck_mini_runtime.sounds import play_sound_async
    path = sounds.wav_files[SOUND_INDEX]
    t0 = time.monotonic()
    play_sound_async(path, volume=sounds.volume)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1, f"play_sound_async blocked for {elapsed:.3f} s"
    time.sleep(1.0)  # let the async sound finish before test teardown
