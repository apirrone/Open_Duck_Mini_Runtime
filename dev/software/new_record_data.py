from pypot.feetech import FeetechSTS3215IO
import pickle
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description="Record data from Feetech motor")
parser.add_argument(
    "--new_firmware",
    action="store_true",
    help="Use new firmware for the motor",
    default=False,
)
args = parser.parse_args()
io = FeetechSTS3215IO("/dev/ttyACM0")


def convert_load(raw_load):
    sign = -1
    if raw_load > 1023:
        raw_load -= 1024
        sign = 1
    return sign * raw_load * 0.001


id = 1 if args.new_firmware else 11
kp = 32
kd = 0
acceleration = 0
maximum_acceleration = 0
maximum_velocity = 0
goal_speed = 0

time.sleep(1)
io.set_maximum_acceleration({id: maximum_acceleration})
io.set_acceleration({id: acceleration})
io.set_maximum_velocity({id: maximum_velocity})
io.set_goal_speed({id: goal_speed})
io.set_P_coefficient({id: kp})
io.set_D_coefficient({id: kd})
time.sleep(1)

goal_position = 90

io.set_goal_position({id: 0})
time.sleep(3)

exit()

times = []
positions = []
goal_positions = []
speeds = []
loads = []
currents = []

io.set_goal_position({id: goal_position})

s = time.time()
set = False
while True:
    t = time.time() - s
    # goal_position = np.rad2deg(np.sin(t**2))
    io.set_goal_position({id: goal_position})
    present_position = np.deg2rad(io.get_present_position([id])[0])
    present_speed = np.deg2rad(io.get_present_speed([id])[0])
    present_load = convert_load(io.get_present_load([id])[0])
    present_current = io.get_present_current([id])[0]

    times.append(t)
    positions.append(present_position)
    goal_positions.append(np.deg2rad(goal_position))
    speeds.append(present_speed)
    loads.append(present_load)
    currents.append(present_current)

    if t > 3:
        break

    time.sleep(0.01)

fw_version = "new" if args.new_firmware else "old"
data = {
    "maximum_acceleration": maximum_acceleration,
    "acceleration": acceleration,
    "maximum_velocity": maximum_velocity,
    "kp": kp,
    "kd": kd,
    "times": times,
    "positions": positions,
    "goal_positions": goal_positions,
    "speeds": speeds,
    "loads": loads,
    "currents": currents,
    "fw_version": fw_version,
}

pickle.dump(
    data,
    open(
        f"data_KP_{kp}_{fw_version}.pkl",
        "wb",
    ),
)
