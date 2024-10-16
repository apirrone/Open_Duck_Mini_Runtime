import pickle
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process data from a pkl file")
parser.add_argument("file_path", type=str, help="Path to the data.pkl file")
args = parser.parse_args()

# Load the data from the specified file path
data = pickle.load(open(args.file_path, "rb"))

# data = pickle.load(open("data.pkl", "rb"))
control_freq = data["control_freq"]
record_time = len(data["target_positions"]) * 1 / control_freq
pid = data["pid"]
selected_dof = "left_knee"

import matplotlib.pyplot as plt

# Extracting data for the selected DOF
target_positions = [
    pos[data["dofs"].index(selected_dof)] for pos in data["target_positions"]
]
present_positions = [
    pos[data["dofs"].index(selected_dof)] for pos in data["present_positions"]
]
velocities = [vel[data["dofs"].index(selected_dof)] for vel in data["velocities"]]
currents = [cur[data["dofs"].index(selected_dof)] for cur in data["current"]]
voltages = [volt[data["dofs"].index(selected_dof)] for volt in data["voltage"]]

# Time vector for the x-axis
time_vector = np.linspace(0, record_time, num=len(target_positions))

# Creating a figure with subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
fig.suptitle(f"Data for {selected_dof} (PID: {pid})")

# Plotting present position, target position and velocity with dual y-axes
ax_pos = axs[0]
ax_vel = ax_pos.twinx()
ax_pos.plot(time_vector, target_positions, label="Target Position", color="b")
ax_pos.plot(time_vector, present_positions, label="Present Position", color="g")
ax_vel.plot(time_vector, velocities, label="Velocity", color="r", linestyle="--")
ax_pos.set_ylabel("Position (rad)")
ax_vel.set_ylabel("Velocity (rad/s)")
ax_pos.legend(loc="upper left")
ax_vel.legend(loc="upper right")

# Plotting current and voltage with dual y-axes
ax_cur = axs[1]
ax_volt = ax_cur.twinx()
ax_cur.plot(time_vector, currents, label="Current", color="c")
ax_volt.plot(time_vector, voltages, label="Voltage", color="m", linestyle="--")
ax_cur.set_ylabel("Current (mA)")
ax_volt.set_ylabel("Voltage (V)")
ax_cur.set_xlabel("Time (s)")
ax_cur.legend(loc="upper left")
ax_volt.legend(loc="upper right")

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
