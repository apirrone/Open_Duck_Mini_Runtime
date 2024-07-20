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
from mini_bdx_runtime.rl_utils import (action_to_pd_targets,
                                       isaac_joints_order, isaac_to_mujoco,
                                       make_action_dict, mujoco_joints_order,
                                       mujoco_to_isaac)


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        serial_port: str = "/dev/ttyUSB0",
        control_freq: float = 30,
        debug_no_imu: bool = False,
    ):
        self.debug_no_imu = debug_no_imu
        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path)
        self.hwi = HWI(serial_port)
        if not self.debug_no_imu:
            self.uart = serial.Serial("/dev/ttyS0", baudrate=115200)
            self.imu = adafruit_bno055.BNO055_UART(self.uart)
            self.imu.mode = adafruit_bno055.NDOF_MODE
            self.last_imu_data = ([0, 0, 0, 0], [0, 0, 0])
            self.imu_queue = Queue()
            Thread(target=self.imu_worker, daemon=True).start()

        self.control_freq = control_freq

        self.angularVelocityScale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_clip = (-1, 1)
        self.obs_clip = (-5, 5)

        # self.obs_size = 55
        self.obs_size = 54  # fake lin vel
        self.action_size = 15

        self.prev_action = np.zeros(self.action_size)

        self.mujoco_init_pos = np.array(
            [
                # right_leg
                0.013627156377842975,
                0.07738878096596595,
                0.5933527914082196,
                -1.630548419252953,
                0.8621333440557593,
                # left leg
                -0.013946457213457239,
                0.07918837709879874,
                0.5325073962634973,
                -1.6225192902713386,
                0.9149246381274986,
                # head
                -0.17453292519943295,
                -0.17453292519943295,
                8.65556854322817e-27,
                0,
                0,
            ]
        )
        self.isaac_init_pos = np.array(mujoco_to_isaac(self.mujoco_init_pos))
        self.pd_action_offset = [
            0.0,
            -0.57,
            0.52,
            0.0,
            0.0,
            -0.57,
            0.0,
            0.0,
            0.48,
            -0.48,
            0.0,
            -0.57,
            0.52,
            0.0,
            0.0,
        ]
        self.pd_action_scale = [
            0.98,
            1.4,
            1.47,
            2.93,
            2.2,
            1.04,
            0.98,
            2.93,
            2.26,
            2.26,
            0.98,
            1.4,
            1.47,
            2.93,
            2.2,
        ]

    def imu_worker(self):
        while True:
            raw_orientation = self.imu.quaternion  # quat
            raw_ang_vel = self.imu.gyro  # xyz

            # convert to correct axes
            quat = [
                raw_orientation[3],
                raw_orientation[0],
                raw_orientation[1],
                raw_orientation[2],
            ]

            try:
                rot_mat = R.from_quat(quat).as_matrix()
            except:
                continue

            tmp = np.eye(4)
            tmp[:3, :3] = rot_mat
            tmp = fv_utils.rotateInSelf(tmp, [0, 0, 90])
            final_orientation_mat = tmp[:3, :3]
            final_orientation_quat = R.from_matrix(final_orientation_mat).as_quat()

            final_ang_vel = [-raw_ang_vel[1], raw_ang_vel[0], raw_ang_vel[2]]

            self.imu_queue.put((final_orientation_quat, final_ang_vel))
            time.sleep(1 / self.control_freq)

    def get_imu_data(self):
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_imu_data

    def get_obs(self, commands):
        # Don't forget to re invert the angles from the hwi
        if not self.debug_no_imu:
            orientation_quat, ang_vel = self.get_imu_data()
        else:
            orientation_quat = [1, 0, 0, 0]
            ang_vel = [0, 0, 0]

        dof_pos = self.hwi.get_present_positions()  # rad
        dof_vel = self.hwi.get_present_velocities()  # rev/min
        dof_vel = (2 * np.pi * dof_vel) / 60  # rad/s

        dof_pos_scaled = dof_pos * self.dof_pos_scale
        dof_vel_scaled = dof_vel * self.dof_vel_scale

        # adding fake antennas

        dof_pos_scaled = np.concatenate([dof_pos_scaled, [0, 0]])
        dof_vel_scaled = np.concatenate([dof_vel_scaled, [0, 0]])

        dof_pos_scaled = mujoco_to_isaac(dof_pos_scaled)
        dof_vel_scaled = mujoco_to_isaac(dof_vel_scaled)

        fake_lin_vel = [0.02, 0, 0]

        return np.concatenate(
            [
                fake_lin_vel,
                # orientation_quat,
                ang_vel,
                dof_pos_scaled,
                dof_vel_scaled,
                self.prev_action,
                commands,
            ]
        )

    def start(self):
        self.hwi.turn_on()

    def run(self):
        # saved_obs = pickle.load(open("saved_obs.pkl", "rb"))
        i = 10
        while True:
            start = time.time()
            commands = [0.0, 0.0, 0.0]
            obs = self.get_obs(commands)  # taks a lot of time
            # obs = saved_obs[i]
            obs = np.clip(obs, self.obs_clip[0], self.obs_clip[1])

            action = self.policy.infer(obs)
            self.prev_action = action.copy()  # here ? # Maybe here
            action = action_to_pd_targets(
                action, self.pd_action_offset, self.pd_action_scale
            )  # order OK
            # action = action + self.isaac_init_pos
            action = np.clip(action, self.action_clip[0], self.action_clip[1])

            robot_action = isaac_to_mujoco(action)
            # print(robot_action)
            action_dict = make_action_dict(robot_action, mujoco_joints_order)
            self.hwi.set_position_all(action_dict)
            i += 1
            took = time.time() - start
            time.sleep((max(1 / self.control_freq - took, 0)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    args = parser.parse_args()

    rl_walk = RLWalk(args.onnx_model_path)
    rl_walk.start()
    rl_walk.run()
