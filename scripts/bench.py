from mini_bdx_runtime.hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer
import time
import adafruit_bno055
import serial
import numpy as np
from threading import Thread

hwi = HWI("/dev/ttyUSB0")


def imu_worker():
    uart = serial.Serial("/dev/ttyS0")  # , baudrate=115200)
    imu = adafruit_bno055.BNO055_UART(uart)
    imu.mode = adafruit_bno055.IMUPLUS_MODE
    while True:
        raw_orientation = imu.quaternion  # quat
        time.sleep(60 / 2)


Thread(target=imu_worker, daemon=True).start()
policy = OnnxInfer("/home/bdx/ONNX.onnx")
adaptation_module = OnnxInfer("/home/bdx/ADAPTATION.onnx", "obs_history")

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
times["rma_inference"] = []
times["policy_inference"] = []
times["full_loop"] = []

freq = 60
for i in range(500):
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
    # raw_orientation = imu.quaternion  # quat
    took = time.time() - s
    times["get_imu"].append(took)

    s = time.time()
    latent = adaptation_module.infer(np.zeros(765))
    took = time.time() - s
    times["rma_inference"].append(took)

    s = time.time()
    policy.infer(np.zeros(51 + 18))
    took = time.time() - s
    times["policy_inference"].append(took)

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
report["rma_inference_mean"] = np.mean(times["rma_inference"])
report["rma_inference_std"] = np.std(times["rma_inference"])
report["policy_inference_mean"] = np.mean(times["policy_inference"])
report["policy_inference_std"] = np.std(times["policy_inference"])
report["full_loop_mean"] = np.mean(times["full_loop"])
report["full_loop_std"] = np.std(times["full_loop"])

print("Report:")
for key_suffix in [
    "set_pos_all",
    "get_pos_all",
    "get_vel_all",
    "get_imu",
    "rma_inference",
    "policy_inference",
    "full_loop",
]:
    mean_key = f"{key_suffix}_mean"
    std_key = f"{key_suffix}_std"
    mean_val = report[mean_key]
    std_val = report[std_key]
    print(
        f"{mean_key}: {mean_val:.6f} s ({1.0 / mean_val:.2f} Hz), {std_key}: {std_val:.6f} s"
    )
