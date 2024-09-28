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
    LowPassActionFilter,
    isaac_to_mujoco,
    make_action_dict,
    mujoco_joints_order,
    mujoco_to_isaac,
    quat_rotate_inverse,
)
import pygame

# Commands
X_RANGE = [0, 0.14]
Y_RANGE = [-0.1, 0.1]
YAW_RANGE = [-0.3, 0.3]


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        serial_port: str = "/dev/ttyUSB0",
        control_freq: float = 30,
        pid=[1100, 0, 0],
        action_scale=0.25,
        cutoff_frequency=5.0,
        commands=False,
        pitch_bias=0.0,
        rma=False,
        adaptation_module_path=None,
        knees_p=None,
    ):
        self.commands = commands
        self.pitch_bias = pitch_bias

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path)

        self.rma = rma
        self.num_obs = 51
        if self.rma:
            self.adaptation_module = OnnxInfer(adaptation_module_path, "obs_history")
            self.obs_history_size = 15
            self.obs_history = np.zeros((self.obs_history_size, self.num_obs)).tolist()
            self.rma_freq = 5  # Hz
            self.last_rma_time = time.time()

        self.hwi = HWI(serial_port)

        # IMU
        self.uart = serial.Serial("/dev/ttyS0")  # , baudrate=115200)
        self.imu = adafruit_bno055.BNO055_UART(self.uart)
        # self.imu.mode = adafruit_bno055.NDOF_MODE
        # self.imu.mode = adafruit_bno055.GYRONLY_MODE
        self.imu.mode = adafruit_bno055.IMUPLUS_MODE
        self.last_imu_data = [0, 0, 0, 0]
        self.imu_queue = Queue(maxsize=1)
        Thread(target=self.imu_worker, daemon=True).start()

        # Control
        self.control_freq = control_freq
        self.pid = pid
        self.knees_p = knees_p
        self.last_control = time.time()

        # Scales
        self.linearVelocityScale = 2.0
        self.angularVelocityScale = 0.25
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = action_scale

        self.prev_action = np.zeros(15)

        self.mujoco_init_pos = list(self.hwi.init_pos.values()) + [0, 0]
        self.isaac_init_pos = np.array(mujoco_to_isaac(self.mujoco_init_pos))

        self.last_commands = [0, 0, 0]
        self.command_freq = 10  # hz
        if self.commands:
            pygame.init()
            self._p1 = pygame.joystick.Joystick(0)
            self._p1.init()
            print(f"Loaded joystick with {self._p1.get_numaxes()} axes.")
            self.cmd_queue = Queue(maxsize=1)
            Thread(target=self.commands_worker, daemon=True).start()
        self.last_command_time = time.time()

        self.action_filter = LowPassActionFilter(self.control_freq, cutoff_frequency)

    def commands_worker(self):
        while True:
            self.cmd_queue.put(self.get_commands())
            time.sleep(1 / self.command_freq)

    def get_commands(self):
        last_commands = self.last_commands
        for event in pygame.event.get():
            lin_vel_y = -1 * self._p1.get_axis(0)
            lin_vel_x = -1 * self._p1.get_axis(1)
            ang_vel = -1 * self._p1.get_axis(3)
            if lin_vel_x >= 0:
                lin_vel_x *= np.abs(X_RANGE[1])
            else:
                lin_vel_x *= np.abs(X_RANGE[0])

            if lin_vel_y >= 0:
                lin_vel_y *= np.abs(Y_RANGE[1])
            else:
                lin_vel_y *= np.abs(Y_RANGE[0])

            if ang_vel >= 0:
                ang_vel *= np.abs(YAW_RANGE[1])
            else:
                ang_vel *= np.abs(YAW_RANGE[0])

            last_commands[0] = lin_vel_x
            last_commands[1] = lin_vel_y
            last_commands[2] = ang_vel

        pygame.event.pump()  # process event queue

        return np.around(last_commands, 3)

    def imu_worker(self):
        while True:
            try:
                raw_orientation = self.imu.quaternion  # quat
                euler = R.from_quat(raw_orientation).as_euler("xyz")
            except Exception as e:
                print(e)
                continue

            # Converting to correct axes
            euler = [euler[1], euler[2], euler[0]]
            euler[1] += np.deg2rad(self.pitch_bias)

            final_orientation_quat = R.from_euler("xyz", euler).as_quat()

            self.imu_queue.put(final_orientation_quat)
            time.sleep(1 / (self.control_freq * 2))

    def get_imu_data(self):
        try:
            self.last_imu_data = self.imu_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_imu_data

    def get_last_command(self):
        try:
            self.last_commands = self.cmd_queue.get(False)  # non blocking
        except Exception:
            pass

        return self.last_commands

    def get_obs(self):
        orientation_quat = self.get_imu_data()
        if orientation_quat is None:
            print("IMU ERROR")
            return None

        if self.commands:
            self.last_commands = self.get_last_command()
            print(self.last_commands)

        dof_pos = self.hwi.get_present_positions()  # rad
        dof_vel = self.hwi.get_present_velocities()  # rad/s

        dof_pos_scaled = list(
            np.array(dof_pos - self.mujoco_init_pos[:13]) * self.dof_pos_scale
        )
        dof_vel_scaled = list(np.array(dof_vel) * self.dof_vel_scale)

        # adding fake antennas
        dof_pos_scaled = np.concatenate([dof_pos_scaled, [0, 0]])
        dof_vel_scaled = np.concatenate([dof_vel_scaled, [0, 0]])

        dof_pos_scaled = mujoco_to_isaac(dof_pos_scaled)
        dof_vel_scaled = mujoco_to_isaac(dof_vel_scaled)

        projected_gravity = quat_rotate_inverse(orientation_quat, [0, 0, -1])

        com = list(
            np.array(self.last_commands).copy()
            * np.array(
                [
                    self.linearVelocityScale,
                    self.linearVelocityScale,
                    self.angularVelocityScale,
                ]
            )
        )

        return np.concatenate(
            [
                projected_gravity,
                com,
                dof_pos_scaled,
                dof_vel_scaled,
                self.prev_action,
            ]
        )

    def start(self):
        self.hwi.turn_on()
        self.hwi.set_pid_all(self.pid)
        self.hwi.set_pid([800, 0, 500], "neck_pitch")
        self.hwi.set_pid([800, 0, 500], "head_pitch")
        self.hwi.set_pid([800, 0, 500], "head_yaw")

        if self.knees_p is not None:
            pid = [self.knees_p, self.pid[1], self.pid[2]]
            self.hwi.set_pid(pid, "left_knee")
            self.hwi.set_pid(pid, "right_knee")

        time.sleep(2)

    def run(self):
        robot_computed_obs = []
        saved_latent = []
        try:
            print("Starting")
            while True:
                t = time.time()
                if not t - self.last_control >= 1 / self.control_freq:
                    time.sleep(0.01)  # to avoid busy waiting
                    continue

                obs = self.get_obs()
                if obs is None:
                    break

                robot_computed_obs.append(obs)

                if self.rma:
                    self.obs_history.append(obs)
                    self.obs_history = self.obs_history[-self.obs_history_size :]

                    if t - self.last_rma_time >= 1 / self.rma_freq:
                        latent = self.adaptation_module.infer(
                            np.array(self.obs_history).flatten()
                        )
                        self.last_rma_time = t
                    saved_latent.append(latent)
                    policy_input = np.concatenate([obs, latent])
                    action = self.policy.infer(policy_input)
                else:
                    action = self.policy.infer(obs)

                self.prev_action = action.copy()

                action = action * self.action_scale + self.isaac_init_pos

                self.action_filter.push(action)
                action = self.action_filter.get_filtered_action()

                robot_action = isaac_to_mujoco(action)

                self.last_control = t

                action_dict = make_action_dict(robot_action, mujoco_joints_order)
                self.hwi.set_position_all(action_dict)

        except KeyboardInterrupt:
            pass

        pickle.dump(robot_computed_obs, open("robot_computed_obs.pkl", "wb"))
        pickle.dump(saved_latent, open("robot_latent.pkl", "wb"))
        time.sleep(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=1100)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=30)
    parser.add_argument("--cutoff_frequency", type=int, default=5)
    parser.add_argument("--rma", action="store_true", default=False)
    parser.add_argument("--adaptation_module_path", type=str, required=False)
    parser.add_argument("--knees_p", type=int, required=False, default=None)
    parser.add_argument("--pitch_bias", type=float, default=0.0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=False,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    rl_walk = RLWalk(
        args.onnx_model_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        cutoff_frequency=args.cutoff_frequency,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        rma=args.rma,
        adaptation_module_path=args.adaptation_module_path,
        knees_p=args.knees_p,
    )
    rl_walk.start()
    rl_walk.run()
