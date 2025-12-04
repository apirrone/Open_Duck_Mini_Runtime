import adafruit_bno055
import board
import busio
import numpy as np
import os
import pickle
import math

from collections import deque
from queue import Queue, Full
from threading import Thread
import time


# ---------- Filtering helpers ----------
class HampelFilter1D:
    """
    Streaming Hampel filter for spike removal.
    Keeps a rolling window; if the incoming value deviates from the median by
    > n_sigmas * 1.4826 * MAD, it is replaced with the median (or clamped).
    """
    def __init__(self, window_size=11, n_sigmas=3.5, clamp=False):
        assert window_size % 2 == 1, "Hampel window_size must be odd"
        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.clamp = clamp
        self.buf = deque(maxlen=window_size)

    def _median(self, seq):
        a = sorted(seq)
        m = len(a) // 2
        return a[m]

    def filter(self, x):
        x = float(x)
        self.buf.append(x)
        # pass-through until the buffer is filled
        if len(self.buf) < self.window_size:
            return x

        med = self._median(self.buf)
        abs_dev = [abs(v - med) for v in self.buf]
        mad = self._median(abs_dev)
        if mad == 0.0:
            mad = 1e-9  # avoid zero
        threshold = self.n_sigmas * 1.4826 * mad

        if abs(x - med) > threshold:
            if self.clamp:
                # clamp toward median at exactly the threshold
                return med + math.copysign(threshold, x - med)
            else:
                # replace the spike outright
                return med
        return x


class EMA1D:
    """ Simple exponential moving average for gentle smoothing. """
    def __init__(self, alpha=0.12):
        self.alpha = alpha
        self.y = None

    def filter(self, x):
        x = float(x)
        if self.y is None:
            self.y = x
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y


def _valid_vec(v):
    """Return True iff v is a length-3 vector of finite numbers."""
    if v is None:
        return False
    a = np.array(v, dtype=float).reshape(-1)
    if a.shape[0] != 3:
        return False
    return np.all(np.isfinite(a))


class Imu:
    def __init__(
        self,
        sampling_freq,
        user_pitch_bias=0,
        calibrate=False,
        upside_down=True,
        enable_spike_filter=True,
        hampel_window_size=5,      # small window for minimal latency
        hampel_sigmas=3.5,
        smooth_gyro=False,         # NO smoothing on gyro by default (fast response for balance)
        smooth_accel=True,         # light smoothing on accel is okay (used for drift correction)
        ema_alpha_gyro=0.4,        # only used if smooth_gyro=True
        ema_alpha_acc=0.3,         # moderate smoothing for accel
        clamp_spikes=False,
    ):
        self.sampling_freq = float(sampling_freq)
        self.calibrate = calibrate
        self.user_pitch_bias = user_pitch_bias  # kept for compatibility
        self.enable_spike_filter = enable_spike_filter
        self.smooth_gyro = smooth_gyro
        self.smooth_accel = smooth_accel

        # --- IMU init ---
        i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = adafruit_bno055.BNO055_I2C(i2c)

        # Choose fusion mode
        self.imu.mode = adafruit_bno055.NDOF_MODE

        # Orientation remap
        if upside_down:
            self.imu.axis_remap = (
                adafruit_bno055.AXIS_REMAP_Y,
                adafruit_bno055.AXIS_REMAP_X,
                adafruit_bno055.AXIS_REMAP_Z,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
            )
        else:
            self.imu.axis_remap = (
                adafruit_bno055.AXIS_REMAP_Y,
                adafruit_bno055.AXIS_REMAP_X,
                adafruit_bno055.AXIS_REMAP_Z,
                adafruit_bno055.AXIS_REMAP_NEGATIVE,
                adafruit_bno055.AXIS_REMAP_POSITIVE,
                adafruit_bno055.AXIS_REMAP_POSITIVE,
            )

        # Optional calibration routine
        if self.calibrate:
            self.imu.mode = adafruit_bno055.NDOF_MODE
            calibrated = self.imu.calibrated
            while not calibrated:
                print("Calibration status: ", self.imu.calibration_status)
                print("Calibrated : ", self.imu.calibrated)
                calibrated = self.imu.calibrated
                time.sleep(0.1)
            print("CALIBRATION DONE")
            offsets_accelerometer = self.imu.offsets_accelerometer
            offsets_gyroscope = self.imu.offsets_gyroscope
            offsets_magnetometer = self.imu.offsets_magnetometer

            imu_calib_data = {
                "offsets_accelerometer": offsets_accelerometer,
                "offsets_gyroscope": offsets_gyroscope,
                "offsets_magnetometer": offsets_magnetometer,
            }
            for k, v in imu_calib_data.items():
                print(k, v)

            pickle.dump(imu_calib_data, open("imu_calib_data.pkl", "wb"))
            print("Saved", "imu_calib_data.pkl")
            # Exit so you can restart in normal mode
            raise SystemExit(0)

        # Load persisted calibration if present
        if os.path.exists("imu_calib_data.pkl"):
            imu_calib_data = pickle.load(open("imu_calib_data.pkl", "rb"))
            self.imu.mode = adafruit_bno055.CONFIG_MODE
            time.sleep(0.1)
            self.imu.offsets_accelerometer = imu_calib_data["offsets_accelerometer"]
            self.imu.offsets_gyroscope = imu_calib_data["offsets_gyroscope"]
            self.imu.offsets_magnetometer = imu_calib_data["offsets_magnetometer"]
            self.imu.mode = adafruit_bno055.NDOF_MODE
            time.sleep(0.1)
        else:
            print("imu_calib_data.pkl not found")
            print("Imu is running uncalibrated")

        # X-axis tare (disabled by default)
        self.x_offset = 0.0
        # self.tare_x()

        self.last_imu_data = {"gyro": np.zeros(3), "accelero": np.zeros(3)}

        # --- spike filters (per axis) ---
        # Asymmetric filtering strategy for balance robots:
        #   - Gyro: Hampel only (spike removal) - needs fast response for balance
        #   - Accel: Hampel + EMA - can be slower (used for drift correction)
        if self.enable_spike_filter:
            self.gyro_hampel = [
                HampelFilter1D(window_size=hampel_window_size, n_sigmas=hampel_sigmas, clamp=clamp_spikes),
                HampelFilter1D(window_size=hampel_window_size, n_sigmas=hampel_sigmas, clamp=clamp_spikes),
                HampelFilter1D(window_size=hampel_window_size, n_sigmas=hampel_sigmas, clamp=clamp_spikes),
            ]
            self.acc_hampel = [
                HampelFilter1D(window_size=hampel_window_size, n_sigmas=hampel_sigmas, clamp=clamp_spikes),
                HampelFilter1D(window_size=hampel_window_size, n_sigmas=hampel_sigmas, clamp=clamp_spikes),
                HampelFilter1D(window_size=hampel_window_size, n_sigmas=hampel_sigmas, clamp=clamp_spikes),
            ]
            if self.smooth_gyro:
                self.gyro_ema = [EMA1D(alpha=ema_alpha_gyro), EMA1D(alpha=ema_alpha_gyro), EMA1D(alpha=ema_alpha_gyro)]
            if self.smooth_accel:
                self.acc_ema = [EMA1D(alpha=ema_alpha_acc), EMA1D(alpha=ema_alpha_acc), EMA1D(alpha=ema_alpha_acc)]

        # Single-slot queue: always keep only the newest sample
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def tare_x(self):
        print("Taring x ...")
        x_values = []
        num_values = 100
        ok = False
        while not ok:
            try:
                a = self.imu.acceleration
            except Exception as e:
                print("[IMU][tare]:", e)
                time.sleep(0.01)
                continue

            if not _valid_vec(a):
                time.sleep(0.01)
                continue

            x_values.append(float(np.array(a)[0]))
            x_values = x_values[-num_values:]

            if len(x_values) == num_values:
                mean = float(np.mean(x_values))
                std = float(np.std(x_values))
                if std < 0.05:
                    ok = True
                    self.x_offset = mean
                    print("Tare x done")
                else:
                    print("Tare std:", std)

            time.sleep(0.01)

    def imu_worker(self):
        Ts = 1.0 / self.sampling_freq if self.sampling_freq > 0 else 0.02
        while True:
            s = time.time()
            try:
                gyro = np.array(self.imu.gyro, dtype=float).copy()
                accelero = np.array(self.imu.acceleration, dtype=float).copy()
            except Exception as e:
                print("[IMU]:", e)
                time.sleep(Ts)
                continue

            if not (_valid_vec(gyro) and _valid_vec(accelero)):
                time.sleep(Ts)
                continue

            # Offset correction
            accelero[0] -= self.x_offset

            # Asymmetric filtering: fast gyro, smoother accel
            if self.enable_spike_filter:
                for i in range(3):
                    # Gyro: Hampel spike removal (+ optional EMA if smooth_gyro=True)
                    gyro[i] = self.gyro_hampel[i].filter(gyro[i])
                    if self.smooth_gyro:
                        gyro[i] = self.gyro_ema[i].filter(gyro[i])

                    # Accel: Hampel spike removal (+ optional EMA if smooth_accel=True)
                    accelero[i] = self.acc_hampel[i].filter(accelero[i])
                    if self.smooth_accel:
                        accelero[i] = self.acc_ema[i].filter(accelero[i])

            data = {"gyro": gyro, "accelero": accelero}

            # Non-blocking queue: keep only the newest
            try:
                if self.imu_queue.full():
                    _ = self.imu_queue.get_nowait()
                self.imu_queue.put_nowait(data)
            except Full:
                pass

            took = time.time() - s
            time.sleep(max(0.0, Ts - took))

    def get_data(self):
        # Return latest if available; otherwise last known
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non-blocking
        except Exception:
            pass
        return self.last_imu_data


if __name__ == "__main__":
    imu = Imu(
        sampling_freq=50,
        calibrate=False,
        upside_down=False,
        enable_spike_filter=True,
        hampel_window_size=5,
        hampel_sigmas=3.5,
        smooth_gyro=False,      # Fast gyro for balance
        smooth_accel=True,      # Smoother accel for drift correction
        ema_alpha_acc=0.3,
        clamp_spikes=False,
    )
    print("IMU initialized with asymmetric filtering:")
    print("  - Gyro: Hampel only (fast response)")
    print("  - Accel: Hampel + EMA (smoother)")
    while True:
        data = imu.get_data()
        print("gyro", np.around(data["gyro"], 3))
        print("accelero", np.around(data["accelero"], 3))
        print("---")
        time.sleep(1 / 25)
