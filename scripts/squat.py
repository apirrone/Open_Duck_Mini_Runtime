from mini_bdx_runtime.hwi import HWI
import pickle
import numpy as np
import time

hwi = HWI("/dev/ttyUSB0")

pid = [2500, 0, 0]
hwi.turn_on()
hwi.set_pid_all(pid)
# hwi.set_pid_all([2500, 0, 1000])
hwi.set_pid([500, 0, 0], "neck_pitch")
hwi.set_pid([500, 0, 0], "head_pitch")
hwi.set_pid([500, 0, 0], "head_yaw")

init_pos = hwi.init_pos

a = 0.3
f = 0.2
control_freq = 60
record_time = 15

data = {}
data["dofs"] = list(hwi.joints.keys())
data["control_freq"] = control_freq
data["pid"] = pid
data["target_positions"] = []
data["present_positions"] = []
data["velocities"] = []
data["current"] = []
data["voltage"] = []

start = time.time()
while True:
    target_pos = init_pos.copy()
    target_pos["left_hip_pitch"] += a * np.sin(2 * np.pi * f * time.time())
    target_pos["left_knee"] -= a * np.sin(2 * np.pi * f * time.time())
    target_pos["right_hip_pitch"] += a * np.sin(2 * np.pi * f * time.time())
    target_pos["right_knee"] -= a * np.sin(2 * np.pi * f * time.time())

    hwi.set_position_all(target_pos)

    pos = hwi.get_present_positions()
    vel = hwi.get_present_velocities()
    cur = hwi.get_current_all()
    volt = hwi.get_voltage_all()

    data["target_positions"].append(list(target_pos.values()))
    data["present_positions"].append(pos)
    data["velocities"].append(vel)
    data["current"].append(cur)
    data["voltage"].append(volt)

    time.sleep(1 / control_freq)

    if time.time() - start > record_time:
        break

pid_str = "_".join([str(p) for p in pid])
with open(f"data_{pid_str}.pkl", "wb") as file:
    pickle.dump(data, file)
