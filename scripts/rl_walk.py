import pickle
import time
from queue import Queue
from threading import Thread

import adafruit_bno055
import FramesViewer.utils as fv_utils
import numpy as np
import serial
from scipy.spatial.transform import Rotation as R

from mini_bdx_runtime.hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer
from mini_bdx_runtime.rl_utils import (
    ActionFilter,
    isaac_to_mujoco,
    make_action_dict,
    mujoco_joints_order,
    mujoco_to_isaac,
)


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        serial_port: str = "/dev/ttyUSB0",
        control_freq: float = 30,
        debug_no_imu: bool = False,
        action_scale=0.1,
    ):
        self.debug_no_imu = debug_no_imu
        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path)
        self.hwi = HWI(serial_port)
        self.action_filter = ActionFilter(window_size=10)
        if not self.debug_no_imu:
            self.uart = serial.Serial("/dev/ttyS0")  # , baudrate=115200)
            self.imu = adafruit_bno055.BNO055_UART(self.uart)
            # self.imu.mode = adafruit_bno055.NDOF_MODE
            # self.imu.mode = adafruit_bno055.GYRONLY_MODE
            self.imu.mode = adafruit_bno055.IMUPLUS_MODE
            self.last_imu_data = ([0, 0, 0, 0], [0, 0, 0])
            self.imu_queue = Queue()
            Thread(target=self.imu_worker, daemon=True).start()

        self.control_freq = control_freq

        self.angularVelocityScale = 0.25
        self.dof_pos_scale = 1.0
        # self.dof_vel_scale = 0.05
        self.dof_vel_scale = 0.01
        self.action_clip = (-1, 1)
        self.obs_clip = (-5, 5)
        self.zero_yaw = None
        self.action_scale = action_scale

        self.prev_action = np.zeros(15)

        self.mujoco_init_pos = list(self.hwi.init_pos.values()) + [0, 0]
        self.isaac_init_pos = np.array(mujoco_to_isaac(self.mujoco_init_pos))

    def imu_worker(self):
        while True:
            try:
                raw_orientation = self.imu.quaternion  # quat
                raw_ang_vel = np.deg2rad(self.imu.gyro)  # xyz
                euler = R.from_quat(raw_orientation).as_euler("xyz")
            except Exception as e:
                print(e)
                continue

            # Converting to correct axes
            euler = [euler[1], euler[2], euler[0]]
            # zero yaw
            euler[2] = 0

            final_orientation_quat = R.from_euler("xyz", euler).as_quat()

            final_ang_vel = [-raw_ang_vel[1], raw_ang_vel[0], raw_ang_vel[2]]
            final_ang_vel = list(
                (np.array(final_ang_vel) / (1 / self.control_freq))
                * self.angularVelocityScale
            )

            self.imu_queue.put((final_orientation_quat, final_ang_vel))
            time.sleep(1 / (self.control_freq / 2))

    def get_imu_data(self):
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_imu_data

    def get_obs(self, commands):
        # TODO There is something wrong here.
        # Plot the computed observations when replaying with the saved observations to see

        if not self.debug_no_imu:
            orientation_quat, ang_vel = self.get_imu_data()
            if ang_vel is None or orientation_quat is None:
                print("IMU ERROR")
                return None
        else:
            orientation_quat = [1, 0, 0, 0]
            ang_vel = [0, 0, 0]

        dof_pos = self.hwi.get_present_positions()  # rad
        dof_vel = self.hwi.get_present_velocities()  # rev/min

        dof_pos_scaled = list(
            np.array(dof_pos - self.mujoco_init_pos[:13]) * self.dof_pos_scale
        )
        dof_vel_scaled = list(np.array(dof_vel) * self.dof_vel_scale)

        # adding fake antennas
        dof_pos_scaled = np.concatenate([dof_pos_scaled, [0, 0]])
        dof_vel_scaled = np.concatenate([dof_vel_scaled, [0, 0]])

        dof_pos_scaled = mujoco_to_isaac(dof_pos_scaled)
        dof_vel_scaled = mujoco_to_isaac(dof_vel_scaled)

        return np.concatenate(
            [
                orientation_quat,
                ang_vel,
                dof_pos_scaled,
                dof_vel_scaled,
                self.prev_action,
                commands,
            ]
        )

    def start(self):
        self.hwi.turn_on()
        pid = [500, 0, 2000]
        # pid = [100, 0, 50]
        self.hwi.set_pid_all(pid)

        time.sleep(5)

    def run(self):
        saved_obs = pickle.load(open("mujoco_saved_obs.pkl", "rb"))
        i = 0
        robot_computed_obs = []
        try:
            print("Starting")
            while True:
                start = time.time()
                commands = [0.1, 0.0, 0.0]
                obs = self.get_obs(commands)
                if obs is None:
                    break
                robot_computed_obs.append(obs)
                # obs = saved_obs[i]
                obs = np.clip(obs, self.obs_clip[0], self.obs_clip[1])

                action = self.policy.infer(obs)

                action = action * self.action_scale
                action = np.clip(action, self.action_clip[0], self.action_clip[1])

                self.action_filter.push(action)
                action = self.action_filter.get_filtered_action()

                self.prev_action = action.copy()  # here ? # Maybe here
                action = self.isaac_init_pos + action

                robot_action = isaac_to_mujoco(action)

                action_dict = make_action_dict(robot_action, mujoco_joints_order)
                self.hwi.set_position_all(action_dict)

                i += 1
                took = time.time() - start
                # print(
                #     "FPS",
                #     np.around(1 / took, 3),
                #     "-- target",
                #     self.control_freq,
                # )
                time.sleep((max(1 / self.control_freq - took, 0)))
                # if i > len(saved_obs) - 1:
                #     break
        except KeyboardInterrupt:
            pass

        pickle.dump(robot_computed_obs, open("robot_computed_obs.pkl", "wb"))
        time.sleep(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument("-a", "--action_scale", type=float, default=0.1)
    args = parser.parse_args()

    rl_walk = RLWalk(
        args.onnx_model_path, debug_no_imu=False, action_scale=args.action_scale
    )
    rl_walk.start()
    rl_walk.run()
