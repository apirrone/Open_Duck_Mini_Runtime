import adafruit_bno055
import time
import serial
from scipy.spatial.transform import Rotation as R
import pickle

uart = serial.Serial("/dev/ttyS0")  # , baudrate=115200)
imu = adafruit_bno055.BNO055_UART(uart)
data = []
print("Starting to record. Press Ctrl+C to stop.")
try: 
    while True:
        try:
            raw_orientation = imu.quaternion  # quat
            data.append(raw_orientation)
            euler = R.from_quat(raw_orientation).as_euler("xyz")
            print(euler)
        except Exception as e:
            print(e)
            continue

        time.sleep(1 / 30)
except KeyboardInterrupt:
    print("Saving data...")
    with open("imu_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print("Data saved.")
    exit(0)


