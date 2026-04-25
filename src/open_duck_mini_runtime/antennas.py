import math
import time

import pigpio

LEFT_GPIO = 13   # board.D13
RIGHT_GPIO = 12  # board.D12
LEFT_SIGN = 1
RIGHT_SIGN = -1
MIN_UPDATE_INTERVAL = 1 / 50  # 20ms

# Servo pulse widths in microseconds (standard hobby servo range)
PULSE_MIN_US = 1000
PULSE_MAX_US = 2000
PULSE_NEUTRAL_US = 1500

# Input deadband: values below this magnitude are clamped to zero
DEADBAND = 0.05
# Secondary deadband on the smoothed value: snaps near-zero to exactly 0
SMOOTH_DEADBAND = 0.02
# EMA smoothing factor: lower = smoother but slower response
EMA_ALPHA = 0.15


def value_to_pulse_us(v: float) -> int:
    """Map [-1, 1] to [PULSE_MIN_US, PULSE_MAX_US]."""
    us = PULSE_NEUTRAL_US + v * (PULSE_MAX_US - PULSE_MIN_US) / 2
    return int(min(max(us, PULSE_MIN_US), PULSE_MAX_US))


class Antennas:
    def __init__(self):
        self._pi = pigpio.pi()
        if not self._pi.connected:
            raise RuntimeError(
                "Cannot connect to pigpiod. Run: sudo systemctl start pigpiod"
            )

        self._smooth_left = 0.0
        self._smooth_right = 0.0
        self._last_us_left = PULSE_NEUTRAL_US
        self._last_us_right = PULSE_NEUTRAL_US

        self._pi.set_servo_pulsewidth(LEFT_GPIO, PULSE_NEUTRAL_US)
        self._pi.set_servo_pulsewidth(RIGHT_GPIO, PULSE_NEUTRAL_US)

    def set_position_left(self, position: float) -> None:
        if abs(position) < DEADBAND:
            position = 0.0
        self._smooth_left = EMA_ALPHA * position + (1 - EMA_ALPHA) * self._smooth_left
        if abs(self._smooth_left) < SMOOTH_DEADBAND:
            self._smooth_left = 0.0
        self._write(LEFT_GPIO, self._smooth_left, LEFT_SIGN, "_last_us_left")

    def set_position_right(self, position: float) -> None:
        if abs(position) < DEADBAND:
            position = 0.0
        self._smooth_right = EMA_ALPHA * position + (1 - EMA_ALPHA) * self._smooth_right
        if abs(self._smooth_right) < SMOOTH_DEADBAND:
            self._smooth_right = 0.0
        self._write(RIGHT_GPIO, self._smooth_right, RIGHT_SIGN, "_last_us_right")

    def _write(self, gpio: int, value: float, sign: int, cache_attr: str) -> None:
        if not (-1 <= value <= 1):
            print("Invalid input! Enter a value between -1 and 1.")
            return
        us = value_to_pulse_us(value * sign)
        if us != getattr(self, cache_attr):
            self._pi.set_servo_pulsewidth(gpio, us)
            setattr(self, cache_attr, us)

    def stop(self) -> None:
        time.sleep(MIN_UPDATE_INTERVAL)
        self._pi.set_servo_pulsewidth(LEFT_GPIO, PULSE_NEUTRAL_US)
        self._pi.set_servo_pulsewidth(RIGHT_GPIO, PULSE_NEUTRAL_US)
        time.sleep(MIN_UPDATE_INTERVAL)
        # Pulse width 0 releases the servo (no signal = no hold force)
        self._pi.set_servo_pulsewidth(LEFT_GPIO, 0)
        self._pi.set_servo_pulsewidth(RIGHT_GPIO, 0)
        self._pi.stop()


if __name__ == "__main__":
    antennas = Antennas()

    try:
        start_time = time.monotonic()
        current_time = start_time

        while current_time - start_time < 5:
            value = math.sin(2 * math.pi * 1 * current_time)
            antennas.set_position_left(value)
            antennas.set_position_right(value)
            time.sleep(MIN_UPDATE_INTERVAL)
            current_time = time.monotonic()

    finally:
        antennas.stop()
