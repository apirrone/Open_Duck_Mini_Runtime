
from mini_bdx_runtime.raw_imu import Imu

if __name__ == "__main__":
    imu = Imu(50, calibrate=True, upside_down=False)