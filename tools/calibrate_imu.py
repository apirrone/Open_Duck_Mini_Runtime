from open_duck_mini_runtime.hardware.raw_imu import Imu

if __name__ == "__main__":
    imu = Imu(50, calibrate=True, upside_down=False)
