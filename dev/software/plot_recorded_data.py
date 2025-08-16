import pickle
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
args = parser.parse_args()

data = pickle.load(open(args.file, "rb"))


plt.figure()

plt.subplot(4, 1, 1)
plt.plot(data["times"], data["positions"], label="positions (rad)")
plt.plot(data["times"], data["goal_positions"], label="goal_positions (rad)")
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(data["times"], data["speeds"], label="speed (rad/s)")
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(data["times"], data["loads"], label="load")
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(data["times"], data["currents"], label="current (mA)")
plt.legend()

# plt.suptitle(f"Acceleration: {data['acceleration']}, KP: {data['kp']}, KD: {data['kd']}")
plt.suptitle(f"KP: {data['kp']}, firmware: {data['fw_version']}")
plt.tight_layout()

plt.show()
