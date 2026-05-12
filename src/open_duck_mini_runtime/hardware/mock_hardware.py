"""
Mock hardware implementations for testing and running the dashboard/TUI on macOS
or without real hardware connected. Provides dynamic, high-fidelity simulated telemetry.
"""

import logging
import math
import time
import numpy as np

logger = logging.getLogger(__name__)


class MockHWI:
    def __init__(self, duck_config, usb_port="/dev/mock"):
        self.duck_config = duck_config
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
            "right_hip_yaw": 10,
            "right_hip_roll": 11,
            "right_hip_pitch": 12,
            "right_knee": 13,
            "right_ankle": 14,
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
            "right_hip_yaw": -0.003,
            "right_hip_roll": -0.065,
            "right_hip_pitch": 0.635,
            "right_knee": 1.379,
            "right_ankle": -0.796,
        }

        self.joints_offsets = getattr(self.duck_config, "joints_offset", {k: 0.0 for k in self.joints})

        self.current_positions = dict(self.init_pos)
        self.current_velocities = {k: 0.0 for k in self.joints}
        self.kps = np.ones(len(self.joints)) * 32
        self.kds = np.ones(len(self.joints)) * 0

        logger.info("MockHWI initialized")

    def set_kps(self, kps):
        self.kps = kps

    def set_kds(self, kds):
        self.kds = kds

    def set_kp(self, id, kp):
        pass

    def turn_on(self):
        logger.info("MockHWI turn_on")
        self.current_positions = dict(self.init_pos)
        self.current_velocities = {k: 0.0 for k in self.joints}

    def turn_off(self):
        logger.info("MockHWI turn_off")

    def set_position(self, joint_name, pos):
        if joint_name in self.current_positions:
            old = self.current_positions[joint_name]
            self.current_velocities[joint_name] = (pos - old) * 50.0
            self.current_positions[joint_name] = pos

    def set_position_all(self, joints_positions):
        for joint, target in joints_positions.items():
            if joint in self.current_positions:
                old = self.current_positions[joint]
                # Smooth approach towards target to simulate physical movement
                new_pos = old + (target - old) * 0.6
                self.current_velocities[joint] = (new_pos - old) * 50.0
                self.current_positions[joint] = new_pos

    def get_present_positions(self, ignore=[]):
        positions = [
            self.current_positions[joint]
            for joint in self.joints.keys()
            if joint not in ignore
        ]
        return np.array(np.around(positions, 3))

    def get_present_velocities(self, rad_s=True, ignore=[]):
        velocities = [
            self.current_velocities[joint]
            for joint in self.joints.keys()
            if joint not in ignore
        ]
        return np.array(np.around(velocities, 3))


class MockImu:
    def __init__(self, sampling_freq, user_pitch_bias=0, calibrate=False, upside_down=True):
        self.sampling_freq = sampling_freq
        self.start_time = time.time()
        logger.info("MockImu initialized")

    def get_data(self):
        t = time.time() - self.start_time
        # Generates smooth, premium telemetry waves to make the dashboard UI visually dynamic
        gx = math.sin(t * 3.0) * 0.05
        gy = math.cos(t * 2.5) * 0.05
        gz = math.sin(t * 1.5) * 0.02
        ax = math.sin(t * 2.0) * 0.1
        ay = math.cos(t * 2.0) * 0.1
        az = 9.81 + math.sin(t * 4.0) * 0.05
        return {
            "gyro": np.array([gx, gy, gz]),
            "accelero": np.array([ax, ay, az]),
            "gravity": np.array([0.0, 0.0, 9.81]),
        }


class MockFeetContacts:
    def __init__(self):
        self.start_time = time.time()
        logger.info("MockFeetContacts initialized")

    def get(self):
        t = time.time() - self.start_time
        # Alternating foot contacts simulating a vibrant walking gait rhythm
        left = math.sin(t * 8.0) > 0
        right = not left
        return [left, right]

    def stop(self):
        pass


class MockEyes:
    def __init__(self, *args, **kwargs):
        logger.info("MockEyes initialized")

    def set_color(self, color):
        pass

    def set_solid(self, solid):
        pass

    def set_standby(self, standby):
        pass

    def stop(self):
        pass


class MockProjector:
    def __init__(self):
        self.on = False
        logger.info("MockProjector initialized")

    def switch(self):
        self.on = not self.on

    def stop(self):
        self.on = False


class MockAntennas:
    def __init__(self):
        logger.info("MockAntennas initialized")

    def set_position_left(self, position):
        pass

    def set_position_right(self, position):
        pass

    def stop(self):
        pass


class MockSounds:
    def __init__(self, *args, **kwargs):
        logger.info("MockSounds initialized")

    def play_random_sound(self):
        pass

    def play_sound(self, index):
        pass
