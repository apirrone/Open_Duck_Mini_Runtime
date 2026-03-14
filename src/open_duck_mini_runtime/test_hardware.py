"""
Interactive hardware validation CLI for Open Duck Mini Runtime.

Exercises every hardware subsystem with verbose output.  Run manually on any
target board to confirm all subsystems are wired, powered, and functional.

    uv run test-hardware                        # full suite, no servo motion
    uv run test-hardware --no-move              # explicit safe mode (same as default)
    uv run test-hardware --skip-sounds          # skip sounds if no speaker
    uv run test-hardware --skip-servos          # skip servos if waveshare not connected
    uv run test-hardware --serial-port /dev/ttyUSB0

Exit 0 = all enabled tests pass; 1 = any failure.
"""

import argparse
import math
import os
import sys
import time
from typing import Optional

PASS = "[ PASS ]"
FAIL = "[ FAIL ]"
SKIP = "[ SKIP ]"

# All known servo IDs: right leg 10-14, left leg 20-24, head 30-33
EXPECTED_SERVO_IDS = list(range(10, 15)) + list(range(20, 25)) + list(range(30, 34))


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"  {msg}")


def result_line(label: str, status: str, detail: str = "") -> None:
    msg = f"{status} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# ---------------------------------------------------------------------------
# Test: NeoPixels
# ---------------------------------------------------------------------------


def test_neopixels() -> bool:
    print("\n--- NeoPixel LEDs ---")
    try:
        from open_duck_mini_runtime.led_controller import LedController

        leds = LedController()
        try:
            steps = [
                ("RED",   leds.RED),
                ("GREEN", leds.GREEN),
                ("BLUE",  leds.BLUE),
                ("WHITE", leds.WHITE),
                ("OFF",   leds.OFF),
            ]
            for name, color in steps:
                log(f"Setting all pixels to {name}")
                leds.left_color = color
                leds.right_color = color
                leds.proj_color = color
                leds.eye_left_on = True
                leds.eye_right_on = True
                leds.projector_on = True
                leds._apply()  # noqa: SLF001
                time.sleep(0.5)
        finally:
            leds.deinit()
    except Exception as e:
        result_line("NeoPixels", FAIL, str(e))
        return False
    result_line("NeoPixels", PASS)
    return True


# ---------------------------------------------------------------------------
# Test: MG90s antennas
# ---------------------------------------------------------------------------


def test_mg90s_antennas() -> bool:
    print("\n--- MG90s Antennas ---")
    try:
        from open_duck_mini_runtime.antennas import Antennas

        antennas = Antennas()
        try:
            start = time.monotonic()
            now = start
            steps = 0
            while now - start < 3.0:
                value = math.sin(2 * math.pi * 1 * now)
                antennas.set_position_left(value)
                antennas.set_position_right(value)
                if steps % 25 == 0:
                    log(f"t={now - start:.1f}s  pos={value:.3f}")
                time.sleep(1 / 50)
                now = time.monotonic()
                steps += 1
        finally:
            antennas.stop()
    except Exception as e:
        result_line("MG90s antennas", FAIL, str(e))
        return False
    result_line("MG90s antennas", PASS)
    return True


# ---------------------------------------------------------------------------
# Test: IMU
# ---------------------------------------------------------------------------


def test_imu(num_samples: int) -> bool:
    print("\n--- IMU (BNO055) ---")
    try:
        from open_duck_mini_runtime import raw_imu

        imu = raw_imu.Imu(sampling_freq=50, upside_down=False)
        # Give the background thread time to collect first readings
        time.sleep(0.5)

        gyro_samples = []
        accel_samples = []
        for i in range(num_samples):
            data = imu.get_data()
            g = data.get("gyro")
            a = data.get("accelero")
            if g is None or a is None:
                result_line("IMU", FAIL, f"None reading at sample {i}")
                return False
            gyro_samples.append(list(g))
            accel_samples.append(list(a))
            time.sleep(0.02)

        import numpy as np

        gyro_arr  = np.array(gyro_samples)
        accel_arr = np.array(accel_samples)

        log(f"Gyro  min={gyro_arr.min():.3f}  max={gyro_arr.max():.3f}  "
            f"mean={gyro_arr.mean():.3f} rad/s")
        log(f"Accel min={accel_arr.min():.3f}  max={accel_arr.max():.3f}  "
            f"mean={accel_arr.mean():.3f} m/s²")

        # Gravity check: at least one accel axis should have |val| > 5 m/s²
        if not (np.abs(accel_arr).max(axis=0) > 5).any():
            result_line("IMU", FAIL, "no gravity signal — is the IMU connected?")
            return False

        # Sanity check: gyro must not be saturated
        if (np.abs(gyro_arr) > 50).any():
            result_line("IMU", FAIL, "gyro values out of range (>50 rad/s)")
            return False

    except Exception as e:
        result_line("IMU", FAIL, str(e))
        return False

    result_line("IMU", PASS, f"{num_samples} samples collected")
    return True


# ---------------------------------------------------------------------------
# Test: Sounds
# ---------------------------------------------------------------------------


def test_sounds(sound_index: Optional[int]) -> bool:
    print("\n--- Sounds ---")
    try:
        from open_duck_mini_runtime.sounds import Sounds

        sounds = Sounds()
        if not sounds.wav_files:
            result_line("Sounds", FAIL, "no .wav files found in assets/")
            return False

        if sound_index is not None:
            if sound_index >= len(sounds.wav_files):
                result_line("Sounds", FAIL,
                            f"index {sound_index} out of range (0–{len(sounds.wav_files) - 1})")
                return False
            log(f"Playing [{sound_index}]: {sounds.wav_files[sound_index].name}")
            sounds.play_sound(sound_index)
        else:
            for i, wav in enumerate(sounds.wav_files):
                log(f"Playing [{i}/{len(sounds.wav_files) - 1}]: {wav.name}")
                sounds.play_sound(i)

    except Exception as e:
        result_line("Sounds", FAIL, str(e))
        return False

    result_line("Sounds", PASS)
    return True


# ---------------------------------------------------------------------------
# Test: Feetech servos
# ---------------------------------------------------------------------------


def test_feetech_servos(serial_port: str, move: bool) -> bool:
    print("\n--- Feetech Servos ---")
    try:
        import rustypot

        io = rustypot.feetech(serial_port, 1_000_000)

        responding = []
        missing = []
        for sid in EXPECTED_SERVO_IDS:
            try:
                io.read_present_position([sid])
                responding.append(sid)
                log(f"ID {sid:2d}: responds")
            except Exception:
                missing.append(sid)
                log(f"ID {sid:2d}: no response")

        if missing:
            log(f"Missing IDs: {missing}")
        log(f"Responding: {len(responding)}/{len(EXPECTED_SERVO_IDS)} servos")

        if not responding:
            result_line("Feetech servos", FAIL, "no servos responded")
            return False

        if move:
            log("Moving servos ±0.1 rad from current position (torque on)...")
            try:
                io.disable_torque(responding)
                io.set_kps(responding, [2] * len(responding))

                for sid in responding:
                    try:
                        pos_list = io.read_present_position([sid])
                        original = pos_list[0]
                        log(f"  ID {sid:2d}: current pos={original:.3f} rad")
                        io.write_goal_position([sid], [original + 0.1])
                        time.sleep(0.5)
                        io.write_goal_position([sid], [original - 0.1])
                        time.sleep(0.5)
                        io.write_goal_position([sid], [original])
                        time.sleep(0.3)
                    except Exception as e:
                        log(f"  ID {sid:2d}: motion error — {e}")
            finally:
                io.disable_torque(responding)
                log("Torque disabled on all servos.")
        else:
            log("--no-move set; skipping motion commands.")

    except Exception as e:
        result_line("Feetech servos", FAIL, str(e))
        return False

    result_line("Feetech servos", PASS,
                f"{len(responding)}/{len(EXPECTED_SERVO_IDS)} responded")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive hardware validation for Open Duck Mini Runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--serial-port",
        default=os.environ.get("DUCK_SERIAL_PORT", "/dev/ttyACM0"),
        help="Serial port for Feetech motor controller",
    )
    parser.add_argument("--skip-leds",     action="store_true", help="Skip NeoPixel test")
    parser.add_argument("--skip-antennas", action="store_true", help="Skip antenna test")
    parser.add_argument("--skip-imu",      action="store_true", help="Skip IMU test")
    parser.add_argument("--skip-sounds",   action="store_true", help="Skip sounds test")
    parser.add_argument("--skip-servos",   action="store_true", help="Skip servo test")
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="Report servo positions without commanding motion (safe default)",
    )
    parser.add_argument(
        "--sound-index",
        type=int,
        default=None,
        metavar="INT",
        help="Play a single sound by index instead of all",
    )
    parser.add_argument(
        "--imu-samples",
        type=int,
        default=20,
        metavar="INT",
        help="Number of IMU samples to collect for validation",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Open Duck Mini — interactive hardware test")
    print("=" * 60)
    print(f"  Serial port : {args.serial_port}")
    print(f"  No-move     : {args.no_move}")
    print(f"  IMU samples : {args.imu_samples}")
    print("=" * 60)

    results: dict[str, Optional[bool]] = {
        "NeoPixels":       None,
        "MG90s antennas":  None,
        "IMU":             None,
        "Sounds":          None,
        "Feetech servos":  None,
    }

    if args.skip_leds:
        results["NeoPixels"] = None
    else:
        results["NeoPixels"] = test_neopixels()

    if args.skip_antennas:
        results["MG90s antennas"] = None
    else:
        results["MG90s antennas"] = test_mg90s_antennas()

    if args.skip_imu:
        results["IMU"] = None
    else:
        results["IMU"] = test_imu(args.imu_samples)

    if args.skip_sounds:
        results["Sounds"] = None
    else:
        results["Sounds"] = test_sounds(args.sound_index)

    if args.skip_servos:
        results["Feetech servos"] = None
    else:
        results["Feetech servos"] = test_feetech_servos(args.serial_port, not args.no_move)

    # Summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        if ok is None:
            status = SKIP
        elif ok:
            status = PASS
        else:
            status = FAIL
            all_pass = False
        print(f"  {status} {name}")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
