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

        self.pitch_bias = self.nominal_pitch_bias + self.user_pitch_bias

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
            exit()

        if os.path.exists("imu_calib_data.pkl"):
            imu_calib_data = pickle.load(open("imu_calib_data.pkl", "rb"))
            self.imu.mode = adafruit_bno055.CONFIG_MODE
            time.sleep(0.1)
            self.imu.offsets_accelerometer = imu_calib_data["offsets_accelerometer"]
            self.imu.offsets_gyroscope = imu_calib_data["offsets_gyroscope"]
            self.imu.offsets_magnetometer = imu_calib_data["offsets_magnetometer"]
            self.imu.mode = adafruit_bno055.IMUPLUS_MODE
            time.sleep(0.1)
        else:
            print("imu_calib_data.pkl not found")
            print("Imu is running uncalibrated")

        self.last_imu_data = {"orientation": [0, 0, 0, 0], "gyro": [0, 0, 0]}
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

    def convert_axes(self, euler):
        euler = [np.pi + euler[1], euler[0], euler[2]]
        return euler

    def imu_worker(self):
        while True:
            s = time.time()
            try:
                # imu returns scalar first
                raw_orientation = np.array(self.imu.quaternion).copy()  # quat
                gyro = np.array(self.imu.gyro).copy()  # gyro
                euler = (
                    R.from_quat(raw_orientation, scalar_first=True)
                    .as_euler("xyz")
                    .copy()
                )
            except Exception as e:
                print("[IMU]:", e)
                continue

            if gyro is None or raw_orientation is None:
                continue

            if gyro.any() is None or raw_orientation.any() is None:
                continue

            # Converting to correct axes
            # euler = self.convert_axes(euler)
            euler[1] -= np.deg2rad(self.pitch_bias)
            # euler[2] = 0  # ignoring yaw

            # gives scalar last, which is what isaac wants
            final_orientation_quat = R.from_euler("xyz", euler).as_quat()

            self.imu_queue.put(
                {"orientation": final_orientation_quat.copy(), "gyro": gyro.copy()}
            )
            took = time.time() - s
            time.sleep(max(0, 1 / self.sampling_freq - took))

    def get_data(self, as_euler=False, as_mat=False):
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        try:
            if as_euler:
                euler = R.from_quat(self.last_imu_data["orientation"]).as_euler("xyz")
                self.last_imu_data["orientation"] = euler
            elif as_mat:
                if self.last_imu_data["orientation"].shape != (3, 3):
                    mat = R.from_quat(self.last_imu_data["orientation"]).as_matrix()
                    self.last_imu_data["orientation"] = mat
            
            return self.last_imu_data

        except Exception as e:
            print("[IMU]: ", e)
            return None


if __name__ == "__main__":
    imu = Imu(50, calibrate=False, upside_down=False)
    # imu = Imu(50, upside_down=False)
    while True:
        data = imu.get_data(as_mat=True)
        if data is None:
            continue
        # print(data)
        # print("gyro", np.around(data["gyro"], 3))
        # print("orientation", np.around(data["orientation"], 3))
        mat = data["orientation"]


        # gravity = np.array(data.site_xmat[self.get_site_id_from_name("imu")]).reshape(
        #     (3, 3)
        # ).T @ np.array([0, 0, -1])

        gravity = mat @ np.array([0, 0, -1])
        print("gravity", np.around(gravity, 3))
        print("---")
        time.sleep(1 / 25)
