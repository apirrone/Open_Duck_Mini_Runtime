import time
from typing import List

import numpy as np

from mini_bdx_runtime.io_330 import Dxl330IO


class HWI:
    def __init__(self, usb_port="/dev/ttyUSB1", baudrate=1000000):
        self.dxl_io = Dxl330IO(usb_port, baudrate=baudrate)
        self.joints = {
            "right_hip_yaw": 10,
            "right_hip_roll": 11,
            "right_hip_pitch": 12,
            "right_knee": 13,
            "right_ankle": 14,
            "left_hip_yaw": 20,
            "left_hip_roll": 21,
            "left_hip_pitch": 22,
            "left_knee": 23,
            "left_ankle": 24,
            "neck_pitch": 30,
            "head_pitch": 31,
            "head_yaw": 32,
        }
        # self.init_pos = {
        #     "right_hip_yaw": 0.0012322806287681889,
        #     "right_hip_roll": 0.02326413299385176,
        #     "right_hip_pitch": 0.897352997720036,
        #     "right_knee": -1.3590427732988653,
        #     "right_ankle": 0.7617041101973798,
        #     "left_hip_yaw": -0.0012322806287510275,
        #     "left_hip_roll": 0.02326413299396169,
        #     "left_hip_pitch": 0.9488873968876821,
        #     "left_knee": -1.3490097909463939,
        #     "left_ankle": 0.7001367286772635,
        #     "neck_pitch": -0.1835609559422233,
        #     "head_pitch": -0.1834247585248765,
        #     "head_yaw": -9.174169188795582e-16,
        # }
        self.init_pos = {
            "right_hip_yaw": -0.014,  # [rad]
            "right_hip_roll": 0.08,  # [rad]
            "right_hip_pitch": 0.53,  # [rad]
            "right_knee": -1.32,  # [rad]
            "right_ankle": 0.91,  # [rad]
            "left_hip_yaw": 0.013,  # [rad]
            "left_hip_roll": 0.077,  # [rad]
            "left_hip_pitch": 0.59,  # [rad]
            "left_knee": -1.33,  # [rad]
            "left_ankle": 0.86,  # [rad]
            "neck_pitch": -0.17,  # [rad]
            "head_pitch": -0.17,  # [rad]
            "head_yaw": 0.0,  # [rad]
            # "left_antenna": 0.0,  # [rad]
            # "right_antenna": 0.0,  # [rad]
        }

        # current based position
        self.dxl_io.set_operating_mode({id: 0x5 for id in self.joints.values()})

    def set_pid(self, pid, joint_name):
        self.dxl_io.set_pid_gain({self.joints[joint_name]: pid})

    def set_pid_all(self, pid):
        self.dxl_io.set_pid_gain({id: pid for id in self.joints.values()})

    def set_low_torque(self):
        self.dxl_io.set_pid_gain({id: [100, 0, 0] for id in self.joints.values()})

    def set_high_torque(self):
        # https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/#position-pid-gain80-82-84-feedforward-1st2nd-gains88-90
        # 128 P factor
        # 16 D factor
        self.dxl_io.set_pid_gain(
            {id: [10 * 128, 0, int(0.5 * 16)] for id in self.joints.values()}
        )
        for name in ["neck_pitch", "head_pitch", "head_yaw"]:
            self.dxl_io.set_pid_gain({self.joints[name]: [150, 0, 0]})

    def turn_on(self):
        self.set_low_torque()
        self.dxl_io.enable_torque(self.joints.values())
        time.sleep(1)
        self.set_position_all(self.init_pos)
        time.sleep(1)
        # self.set_high_torque()

    def turn_off(self):
        self.dxl_io.disable_torque(self.joints.values())

    def goto_zero(self):
        goal = {joint: 0 for joint in self.joints.values()}
        self.dxl_io.set_goal_position(goal)

    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        ids_positions = {
            self.joints[joint]: np.rad2deg(-position)
            for joint, position in joints_positions.items()
        }

        # print(ids_positions)
        self.dxl_io.set_goal_position(ids_positions)

    def set_position(self, joint_name, position):
        self.dxl_io.set_goal_position({self.joints[joint_name]: np.rad2deg(-position)})

    def get_present_current(self, joint_name):
        return self.dxl_io.get_present_current([self.joints[joint_name]])[0]

    def get_goal_current(self, joint_name):
        return self.dxl_io.get_goal_current([self.joints[joint_name]])[0]

    def get_current_limit(self, joint_name):
        return self.dxl_io.get_current_limit([self.joints[joint_name]])[0]

    def get_present_positions(self):
        present_position = list(
            np.around(
                np.deg2rad((self.dxl_io.get_present_position(self.joints.values()))), 3
            )
        )
        factor = np.ones(len(present_position)) * -1
        return present_position * factor

    def get_present_velocities(self, rad_s=True) -> List[float]:
        """
        Returns the present velocities in rad/s or rev/min
        """
        # rev/min
        present_velocities = np.array(
            self.dxl_io.get_present_velocity(self.joints.values())
        )
        if rad_s:
            present_velocities = (2 * np.pi * present_velocities) / 60  # rad/s

        factor = np.ones(len(present_velocities)) * -1
        return list(present_velocities * factor)

    def get_operating_modes(self):
        return self.dxl_io.get_operating_mode(self.joints.values())
