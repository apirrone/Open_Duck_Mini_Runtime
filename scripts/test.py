from mini_bdx_runtime.hwi import HWI
import time

hwi = HWI("/dev/ttyUSB0")

hwi.set_pid_all([1100, 0, 0])
hwi.set_pid([500, 0, 0], "neck_pitch")
hwi.set_pid([500, 0, 0], "head_pitch")
hwi.set_pid([500, 0, 0], "head_yaw")

record = {}
record["normal_vel"] = []
record["dual_vel"] = []
record["normal_pos"] = []
record["dual_pos"] = []
index = 10
try:
    while True:
        pos = hwi.get_present_positions()
        left_hip_pitch_pos = pos[index]
        vel = hwi.get_present_velocities()
        left_hip_pitch_vel = vel[index]

        record["normal_vel"].append(left_hip_pitch_vel)
        record["normal_pos"].append(left_hip_pitch_pos)

        pos_vel = hwi.get_present_positions_and_velocities()
        left_hip_pitch_pos = pos_vel[0][index]
        left_hip_pitch_vel = pos_vel[1][index]
        print(left_hip_pitch_pos, left_hip_pitch_vel)

        record["dual_vel"].append(left_hip_pitch_vel)
        record["dual_pos"].append(left_hip_pitch_pos)
        print(left_hip_pitch_pos, left_hip_pitch_vel)
        print("=")
        time.sleep(0.01)
except KeyboardInterrupt:
    import matplotlib.pyplot as plt

    # Plotting normal velocity versus dual velocity
    plt.figure(figsize=(10, 5))
    plt.plot(record["normal_vel"], label="Normal Velocity")
    plt.plot(record["dual_vel"], label="Dual Velocity", linestyle="--")

    plt.title("Normal Velocity vs Dual Velocity")
    plt.xlabel("Sample Index")
    plt.ylabel("Velocity")
    plt.ylim(-15, 15)  # Limit the y-axis for velocity
    plt.legend()
    plt.grid(True)

    plt.show()

    # Plotting normal position versus dual position
    plt.figure(figsize=(10, 5))
    plt.plot(record["normal_pos"], label="Normal Position")
    plt.plot(record["dual_pos"], label="Dual Position", linestyle="--")

    plt.title("Normal Position vs Dual Position")
    plt.xlabel("Sample Index")
    plt.ylabel("Position")
    plt.ylim(-3.14159, 3.14159)  # Limit the y-axis for position using pi approximation
    plt.legend()
    plt.grid(True)

    plt.show()
