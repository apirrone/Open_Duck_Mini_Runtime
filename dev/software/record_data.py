from pypot.feetech import FeetechSTS3215IO
import pickle
import numpy as np
import time

io = FeetechSTS3215IO("/dev/ttyACM0")

# accelerations = [0, 10, 50, 100, 200, 255]
accelerations = [0]
switch=False
# kps = [4, 8, 16, 32]
# kds = [0, 4, 8, 16, 32]

# accelerations = [0]
kps = [32]
kds = [0]

for acceleration in accelerations:
    for kp in kps:
        for kd in kds:
            print(f"acceleration: {acceleration}, kp: {kp}, kd: {kd}")

            # acceleration = 50
            # kp = 32
            # kd = 0

            # io.set_mode({1: 0})
            # io.set_lock({1: 1})
            # time.sleep(1)
            # io.set_maximum_acceleration({1: acceleration})
            # io.set_acceleration({1: acceleration})
            # time.sleep(1)
            io.set_P_coefficient({1: kp})
            io.set_D_coefficient({1: kd})

            # time.sleep(1)
            goal_position = 90

            io.set_goal_position({1: 0})
            time.sleep(3)

            times = []
            positions = []
            goal_positions = []
            speeds = []
            loads = []
            currents = []

            def convert_load(raw_load):
                sign = -1
                if raw_load > 1023:
                    raw_load -= 1024
                    sign = 1
                return sign * raw_load * 0.001

            io.set_goal_position({1: goal_position})
            s = time.time()
            set = False
            while True:
                t = time.time() - s
                # goal_position = np.rad2deg(np.sin(t**2))
                io.set_goal_position({1: goal_position})
                present_position = np.deg2rad(io.get_present_position([1])[0])
                present_speed = np.deg2rad(io.get_present_speed([1])[0])
                present_load = convert_load(io.get_present_load([1])[0])
                present_current = io.get_present_current([1])[0]

                times.append(t)
                positions.append(present_position)
                goal_positions.append(np.deg2rad(goal_position))
                speeds.append(present_speed)
                loads.append(present_load)
                currents.append(present_current)

                # if switch and t > 0.2 and not set:
                #     goal_position = -goal_position
                #     io.set_goal_position({1: goal_position})
                #     set = True

                if t > 3:
                    break

                time.sleep(0.01)

            data = {
                "acceleration": acceleration,
                "kp": kp,
                "kd": kd,
                "times": times,
                "positions": positions,
                "goal_positions": goal_positions,
                "speeds": speeds,
                "loads": loads,
                "currents": currents,
            }

            pickle.dump(
                data,
                open(f"data_{'switch_' if switch else ''}acceleration_{acceleration}_kp_{kp}_kd_{kd}.pkl", "wb"),
            )
