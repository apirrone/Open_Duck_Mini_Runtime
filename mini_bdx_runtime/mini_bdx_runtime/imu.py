import adafruit_bno055
import board
import busio
import numpy as np
import pickle
import os

# import serial

from queue import Queue
from threading import Thread
import time
from scipy.spatial.transform import Rotation as R


# TODO filter spikes
class Imu:
    def __init__(
        self, sampling_freq, user_pitch_bias=0, calibrate=False, upside_down=True
    ):
        self.sampling_freq = sampling_freq
        self.user_pitch_bias = user_pitch_bias
        self.nominal_pitch_bias = 0
        self.calibrate = calibrate

        # self.uart = serial.Serial("/dev/ttyS0", baudrate=9600)
        # self.imu = adafruit_bno055.BNO055_UART(self.uart)

        i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = adafruit_bno055.BNO055_I2C(i2c)

        self.imu.mode = adafruit_bno055.IMUPLUS_MODE
        # self.imu.mode = adafruit_bno055.ACCGYRO_MODE
        # self.imu.mode = adafruit_bno055.GYRONLY_MODE
        # self.imu.mode = adafruit_bno055.NDOF_MODE
        # self.imu.mode = adafruit_bno055.NDOF_FMC_OFF_MODE

        self.imu.axis_remap = (
            adafruit_bno055.AXIS_REMAP_Y,
            adafruit_bno055.AXIS_REMAP_X,
            adafruit_bno055.AXIS_REMAP_Z,
            adafruit_bno055.AXIS_REMAP_NEGATIVE,
            adafruit_bno055.AXIS_REMAP_POSITIVE,
            adafruit_bno055.AXIS_REMAP_POSITIVE,
        # if upside_down:
            # self.imu.axis_remap = (
                # adafruit_bno055.AXIS_REMAP_Y,
                # adafruit_bno055.AXIS_REMAP_X,
                # adafruit_bno055.AXIS_REMAP_Z,
                # adafruit_bno055.AXIS_REMAP_NEGATIVE,
                # adafruit_bno055.AXIS_REMAP_NEGATIVE,
                # adafruit_bno055.AXIS_REMAP_NEGATIVE,
            # )
        # else:
            # self.imu.axis_remap = (
                # adafruit_bno055.AXIS_REMAP_Y,
                # adafruit_bno055.AXIS_REMAP_X,
                # adafruit_bno055.AXIS_REMAP_Z,
                # adafruit_bno055.AXIS_REMAP_NEGATIVE,
                # adafruit_bno055.AXIS_REMAP_POSITIVE,
                # adafruit_bno055.AXIS_REMAP_POSITIVE,
            # )

        self.pitch_bias = self.nominal_pitch_bias + self.user_pitch_bias

        if self.calibrate:
            self.imu.mode = adafruit_bno055.IMUPLUS_MODE
            calibrated = self.imu.calibrated
            while not calibrated:
                print("Calibration status: ", self.imu.calibration_status)
                print("Calibrated : ", self.imu.calibrated)
                calibrated = self.imu.calibrated
                time.sleep(0.1)
            print("CALIBRATION DONE")
            offsets_accelerometer = self.imu.offsets_accelerometer
            offsets_gyroscope = self.imu.offsets_gyroscope
            # No magnetometer in IMUPLUS_MODE

            imu_calib_data = {
                "offsets_accelerometer": offsets_accelerometer,
                "offsets_gyroscope": offsets_gyroscope,
            }
            for k, v in imu_calib_data.items():
                print(k, v)

            pickle.dump(imu_calib_data, open("imu_calib_data.pkl", "wb"))

            print("Saved", "imu_calib_data.pkl")
            exit()

        if False and os.path.exists("imu_calib_data.pkl"):
            imu_calib_data = pickle.load(open("imu_calib_data.pkl", "rb"))
            self.imu.mode = adafruit_bno055.CONFIG_MODE
            time.sleep(0.1)
            self.imu.offsets_accelerometer = imu_calib_data["offsets_accelerometer"]
            self.imu.offsets_gyroscope = imu_calib_data["offsets_gyroscope"]
            # No magnetometer in IMUPLUS_MODE (skip loading if not present)
            self.imu.mode = adafruit_bno055.IMUPLUS_MODE
            time.sleep(0.1)
        else:
            print("imu_calib_data.pkl not found")
            print("Imu is running uncalibrated")

        self.last_imu_data = {
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # [w, x, y, z]
            "gyro": np.zeros(3, dtype=np.float32),
        }
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def imu_worker(self):
        while True:
            s = time.time()
            try:
                # imu returns scalar first [w, x, y, z]
                raw_orientation = np.array(self.imu.quaternion).copy()
                gyro = np.array(self.imu.gyro, dtype=np.float32).copy()
                euler = (
                    R.from_quat(raw_orientation, scalar_first=True)
                    .as_euler("xyz")
                    .copy()
                )
            except Exception as e:
                print("[IMU]:", e)
                continue

            euler[1] -= np.deg2rad(self.pitch_bias)

            # Convert back to quaternion in scalar-first [w, x, y, z] format
            quat_scalar_last = R.from_euler("xyz", euler).as_quat()  # [x, y, z, w]
            quat_scalar_first = np.array(
                [quat_scalar_last[3], quat_scalar_last[0], quat_scalar_last[1], quat_scalar_last[2]],
                dtype=np.float32,
            )

            self.imu_queue.put({"quaternion": quat_scalar_first, "gyro": gyro})
            took = time.time() - s
            time.sleep(max(0, 1 / self.sampling_freq - took))

    def get_data(self):
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_imu_data


if __name__ == "__main__":
    imu = Imu(50, calibrate=True, upside_down=False)
    # imu = Imu(50, upside_down=False)
    while True:
        data = imu.get_data()
        print("gyro", np.around(data["gyro"], 3))
        print("quaternion", np.around(data["quaternion"], 3))
        print("---")
        time.sleep(1 / 25)
