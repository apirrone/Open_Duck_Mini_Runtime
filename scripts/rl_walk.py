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
        control_freq: float = 60,
        debug_no_imu: bool = False,
        action_scale=0.1,
    ):
        self.debug_no_imu = debug_no_imu
        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path)
        self.hwi = HWI(serial_port)
        if not self.debug_no_imu:
            self.uart = serial.Serial("/dev/ttyS0")  # , baudrate=115200)
            self.imu = adafruit_bno055.BNO055_UART(self.uart)
            self.imu.mode = adafruit_bno055.NDOF_MODE
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

        # self.mujoco_init_pos = np.array(
        #     [
        #         # right_leg
        #         -0.014,
        #         0.08,
        #         0.53,
        #         -1.62,
        #         -1.32,
        #         0.91,
        #         # left leg
        #         0.013,
        #         0.077,
        #         0.59,
        #         -1.33,
        #         0.86,
        #         # head
        #         -0.17,
        #         -0.17,
        #         0.0,
        #         0.0,
        #         0.0,
        #     ]
        # )
        self.isaac_init_pos = np.array(mujoco_to_isaac(self.mujoco_init_pos))

        # self.muj_command_value = pickle.load(
        #     open(
        #         "/home/antoine/MISC/mini_BDX/experiments/mujoco/mujoco_command_value.pkl",
        #         "rb",
        #     )
        # )
        # self.robot_command_value = []
        # self.imu_data = []

    def imu_worker(self):
        while True:
            # start = time.time()
            try:
                raw_orientation = self.imu.quaternion  # quat
                raw_ang_vel = np.deg2rad(self.imu.gyro)  # xyz
            except Exception as e:
                print(e)
                # self.imu_queue.put((None, None))
                continue

            # convert to correct axes. (??)
            quat = [
                raw_orientation[3],
                raw_orientation[0],
                raw_orientation[1],
                raw_orientation[2],
            ]

            try:
                rot_mat = R.from_quat(quat).as_matrix()
            except Exception as e:
                print(e)
                continue

            rot_mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ rot_mat

            tmp = np.eye(4)
            tmp[:3, :3] = rot_mat
            tmp = fv_utils.rotateInSelf(tmp, [0, 0, 90])
            tmp_euler = R.from_matrix(tmp[:3, :3]).as_euler("xyz", degrees=False)
            tmp_euler[2] = 0
            tmp[:3, :3] = R.from_euler("xyz", tmp_euler, degrees=False).as_matrix()
            # if self.zero_yaw is None:
            #     self.zero_yaw = R.from_matrix(tmp[:3, :3]).as_euler(
            #         "xyz", degrees=False
            #     )[2]
            # tmp[:3, :3] = (
            #     R.from_euler("xyz", [0, 0, -self.zero_yaw], degrees=False).as_matrix()
            #     @ tmp[:3, :3]
            # )
            final_orientation_mat = tmp[:3, :3]
            final_orientation_quat = R.from_matrix(final_orientation_mat).as_quat()

            final_ang_vel = [-raw_ang_vel[1], raw_ang_vel[0], raw_ang_vel[2]]
            final_ang_vel = list(
                (np.array(final_ang_vel) / (1 / self.control_freq))
                * self.angularVelocityScale
            )

            self.imu_queue.put((final_orientation_quat, final_ang_vel))
            # print("imu worker took", time.time() - start)
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

        # self.imu_data.append([orientation_quat, ang_vel])
        # pickle.dump(self.imu_data, open("imu_data.pkl", "wb"))

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

        fake_lin_vel = [0.02, 0, 0]

        return np.concatenate(
            [
                # fake_lin_vel,
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
                self.prev_action = action.copy()  # here ? # Maybe here
                action = self.isaac_init_pos + action

                robot_action = isaac_to_mujoco(action)

                # robot_action = self.muj_command_value[i][1]
                action_dict = make_action_dict(robot_action, mujoco_joints_order)
                self.hwi.set_position_all(action_dict)
                # robot_action_fake_antennas = list(robot_action) + [0, 0]

                # present_positions_fake_antennas = list(self.hwi.get_present_positions()) + [
                #     0,
                #     0,
                # ]
                # self.robot_command_value.append(
                #     [robot_action_fake_antennas, present_positions_fake_antennas]
                # )

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
