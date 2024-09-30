from mini_bdx_runtime.hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer
import time
import adafruit_bno055
import serial

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
    print(f"Loop {i} took {took:.3f}s")
    print("target time was", 1 / freq)
