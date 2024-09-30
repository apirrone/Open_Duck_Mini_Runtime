from mini_bdx_runtime.hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer
import time
import adafruit_bno055
import serial
import numpy as np

hwi = HWI("/dev/ttyUSB0")


uart = serial.Serial("/dev/ttyS0")  # , baudrate=115200)
imu = adafruit_bno055.BNO055_UART(uart)
imu.mode = adafruit_bno055.IMUPLUS_MODE

hwi.turn_on()
hwi.set_pid_all([1100, 0, 0])
hwi.set_pid([500, 0, 0], "neck_pitch")
hwi.set_pid([500, 0, 0], "head_pitch")
hwi.set_pid([500, 0, 0], "head_yaw")

time.sleep(1)

init_pos = hwi.init_pos
times = {}
times["set_pos_all"] = []
times["get_pos_all"] = []
times["get_vel_all"] = []
times["get_imu"] = []
times["full_loop"] = []

freq = 60
for i in range(1000):
    start = time.time()

    s = time.time()
    hwi.set_position_all(init_pos)
    took = time.time() - s
    times["set_pos_all"].append(took)

    s = time.time()
    dof_pos = hwi.get_present_positions()  # rad
    took = time.time() - s
    times["get_pos_all"].append(took)

    s = time.time()
    dof_vel = hwi.get_present_velocities()  # rad/s
    took = time.time() - s
    times["get_vel_all"].append(took)

    s = time.time()
    raw_orientation = imu.quaternion  # quat
    took = time.time() - s
    times["get_imu"].append(took)

    took = time.time() - start
    times["full_loop"].append(took)

report = {}
report["set_pos_all_mean"] = np.mean(times["set_pos_all"])
report["set_pos_all_std"] = np.std(times["set_pos_all"])
report["get_pos_all_mean"] = np.mean(times["get_pos_all"])
report["get_pos_all_std"] = np.std(times["get_pos_all"])
report["get_vel_all_mean"] = np.mean(times["get_vel_all"])
report["get_vel_all_std"] = np.std(times["get_vel_all"])
report["get_imu_mean"] = np.mean(times["get_imu"])
report["get_imu_std"] = np.std(times["get_imu"])
report["full_loop_mean"] = np.mean(times["full_loop"])
report["full_loop_std"] = np.std(times["full_loop"])

print("Report:")
for key, value in report.items():
    if "mean" in key:
        print(f"{key}: {value:.6f} seconds ({1.0 / value:.2f} Hz)")
    else:
        print(f"{key}: {value:.6f} seconds")
