import logging
import time

import numpy as np
import rustypot
from open_duck_mini_runtime.duck_config import DuckConfig  # top-level, not in hardware/

logger = logging.getLogger(__name__)


class HWI:
    def __init__(self, duck_config: DuckConfig, usb_port: str = "/dev/ttyACM0"):

        self.duck_config = duck_config

        # Order matters here
        self.joints = {
            "left_hip_yaw": 20,
            "left_hip_roll": 21,
            "left_hip_pitch": 22,
            "left_knee": 23,
            "left_ankle": 24,
            "neck_pitch": 30,
            "head_pitch": 31,
            "head_yaw": 32,
            "head_roll": 33,
            # "left_antenna": None,
            # "right_antenna": None,
            "right_hip_yaw": 10,
            "right_hip_roll": 11,
            "right_hip_pitch": 12,
            "right_knee": 13,
            "right_ankle": 14,
        }

        self.zero_pos = {
            "left_hip_yaw": 0,
            "left_hip_roll": 0,
            "left_hip_pitch": 0,
            "left_knee": 0,
            "left_ankle": 0,
            "neck_pitch": 0,
            "head_pitch": 0,
            "head_yaw": 0,
            "head_roll": 0,
            # "left_antenna":0,
            # "right_antenna":0,
            "right_hip_yaw": 0,
            "right_hip_roll": 0,
            "right_hip_pitch": 0,
            "right_knee": 0,
            "right_ankle": 0,
        }

        self.init_pos = {
            "left_hip_yaw": 0.002,
            "left_hip_roll": 0.053,
            "left_hip_pitch": -0.63,
            "left_knee": 1.368,
            "left_ankle": -0.784,
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0,
            "head_roll": 0,
            # "left_antenna": 0,
            # "right_antenna": 0,
            "right_hip_yaw": -0.003,
            "right_hip_roll": -0.065,
            "right_hip_pitch": 0.635,
            "right_knee": 1.379,
            "right_ankle": -0.796,
        }

        self.joints_offsets = self.duck_config.joints_offset

        self.kps = np.ones(len(self.joints)) * 32  # default kp
        self.kds = np.ones(len(self.joints)) * 0  # default kd
        self.low_torque_kps = np.ones(len(self.joints)) * 2

        self.io = rustypot.feetech(usb_port, 1000000)

    def set_kps(self, kps):
        self.kps = kps
        joint_ids = list(self.joints.values())
        for i, motor_id in enumerate(joint_ids):
            self.io.set_kps([motor_id], [self.kps[i]])

    def set_kds(self, kds):
        self.kds = kds
        joint_ids = list(self.joints.values())
        for i, motor_id in enumerate(joint_ids):
            self.io.set_kds([motor_id], [self.kds[i]])

    def set_kp(self, id, kp):
        self.io.set_kps([id], [kp])

    def turn_on(self):
        joint_ids = list(self.joints.values())
        for i, motor_id in enumerate(joint_ids):
            self.io.set_kps([motor_id], [self.low_torque_kps[i]])
        logger.info("turn on: low kps set")
        time.sleep(1)

        self.set_position_all(self.init_pos)
        logger.info("turn on: init pos set")

        time.sleep(1)

        for i, motor_id in enumerate(joint_ids):
            self.io.set_kps([motor_id], [self.kps[i]])
        logger.info("turn on: high kps set")

    def turn_off(self):
        for motor_id in self.joints.values():
            self.io.disable_torque([motor_id])

    def set_position(self, joint_name, pos):
        """
        pos is in radians
        """
        id = self.joints[joint_name]
        pos = pos + self.joints_offsets[joint_name]
        self.io.write_goal_position([id], [pos])

    def set_position_all(self, joints_positions):
        """
        joints_positions is a dictionary with joint names as keys and joint positions as values
        Warning: expects radians
        """
        for joint, position in joints_positions.items():
            motor_id = self.joints[joint]
            pos = position + self.joints_offsets[joint]
            self.io.write_goal_position([motor_id], [pos])

    def scan_servos(self) -> dict:
        """Ping each servo individually and return {joint_name: present} dict."""
        results = {}
        for name, servo_id in self.joints.items():
            try:
                result = self.io.read_present_position([servo_id])
                results[name] = result is not None and len(result) > 0
            except Exception:
                results[name] = False
        return results

    def get_present_positions(self, ignore=[]):
        """
        Returns the present positions in radians
        """
        positions = []
        for joint, motor_id in self.joints.items():
            if joint in ignore:
                continue
            try:
                result = self.io.read_present_position([motor_id])
                if result is None or len(result) == 0:
                    logger.warning("read_present_position empty for %s", joint)
                    return None
                positions.append(result[0] - self.joints_offsets[joint])
            except Exception as e:
                logger.warning("read_present_position failed for %s: %s", joint, e)
                return None
        return np.array(np.around(positions, 3))

    def get_present_velocities(self, rad_s=True, ignore=[]):
        """
        Returns the present velocities in rad/s (default) or rev/min
        """
        velocities = []
        for joint, motor_id in self.joints.items():
            if joint in ignore:
                continue
            try:
                result = self.io.read_present_velocity([motor_id])
                if result is None or len(result) == 0:
                    logger.warning("read_present_velocity empty for %s", joint)
                    return None
                velocities.append(result[0])
            except Exception as e:
                logger.warning("read_present_velocity failed for %s: %s", joint, e)
                return None
        return np.array(np.around(velocities, 3))
