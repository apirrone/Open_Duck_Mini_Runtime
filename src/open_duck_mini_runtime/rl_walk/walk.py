import logging
import time
import pickle
import threading
import sys

import numpy as np
from open_duck_mini_runtime.rl_walk.onnx_infer import OnnxInfer
from open_duck_mini_runtime.rl_walk.poly_reference_motion import PolyReferenceMotion
from open_duck_mini_runtime.controller.xbox_controller import XBoxController
from open_duck_mini_runtime.rl_walk.rl_utils import make_action_dict, LowPassActionFilter
from open_duck_mini_runtime.duck_config import DuckConfig
from open_duck_mini_runtime.log import setup_logging, TRACE

import os
import signal
from pathlib import Path

logger = logging.getLogger(__name__)

HOME_DIR = os.path.expanduser("~")
# src/assets/ sits three levels above this file (src/open_duck_mini_runtime/rl_walk/walk.py)
ASSETS_ROOT_PATH: str = str(Path(__file__).parent.parent.parent / "assets")


class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        duck_config_path: str = f"{HOME_DIR}/duck_config.json",
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=[30, 0, 0],
        action_scale=0.25,
        commands=False,
        pitch_bias=0,
        save_obs=False,
        replay_obs=None,
        cutoff_frequency=None,
        emulate: bool = False,
    ):

        self.emulate = emulate or sys.platform != "linux"
        self.telemetry = {}
        self._telem_lock = threading.Lock()

        self.duck_config = DuckConfig(config_json_path=duck_config_path)

        self.commands = commands
        self.pitch_bias = pitch_bias

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        self.num_dofs = 14
        self.max_motor_velocity = 5.24  # rad/s

        # Control
        self.control_freq = control_freq
        self.pid = pid

        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        if self.emulate:
            from open_duck_mini_runtime.hardware.mock_hardware import (
                MockHWI, MockImu, MockFeetContacts, MockEyes, MockProjector, MockSounds, MockAntennas
            )
            HWI_cls = MockHWI
            Imu_cls = MockImu
            FeetContacts_cls = MockFeetContacts
            Eyes_cls = MockEyes
            Projector_cls = MockProjector
            Sounds_cls = MockSounds
            Antennas_cls = MockAntennas
            hwi_args = (self.duck_config,)
        else:
            from open_duck_mini_runtime.hardware.hwi import HWI
            from open_duck_mini_runtime.hardware.raw_imu import Imu
            from open_duck_mini_runtime.hardware.feet_contacts import FeetContacts
            from open_duck_mini_runtime.hardware.eyes import Eyes
            from open_duck_mini_runtime.hardware.sounds import Sounds
            from open_duck_mini_runtime.hardware.antennas import Antennas
            from open_duck_mini_runtime.hardware.projector import Projector

            HWI_cls = HWI
            Imu_cls = Imu
            FeetContacts_cls = FeetContacts
            Eyes_cls = Eyes
            Projector_cls = Projector
            Sounds_cls = Sounds
            Antennas_cls = Antennas
            hwi_args = (self.duck_config, serial_port)

        self.hwi = HWI_cls(*hwi_args)

        self.start()

        self.imu = Imu_cls(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )

        self.feet_contacts = FeetContacts_cls()

        # Scales
        self.action_scale = action_scale

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        self.init_pos = list(self.hwi.init_pos.values())

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.paused = self.duck_config.start_paused
        self.motors_enabled = True

        # Fall detection: calibrate "up" from gravity samples collected while paused
        self._up_vector: np.ndarray | None = None
        self._up_calib_acc = np.zeros(3)
        self._up_calib_count = 0
        self._up_calib_target = 10  # 10 samples × 0.1 s pause loop = ~1 s
        self._fall_consecutive = 0
        self._fall_consecutive_required = 3  # frames at 50 Hz before triggering

        self.command_freq = 20  # hz
        if self.commands:
            self.xbox_controller = XBoxController(self.command_freq)

        # Reference motion, but we only really need the length of one phase
        self.PRM = PolyReferenceMotion(
            f"{ASSETS_ROOT_PATH}/polynomial_coefficients.pkl"
        )
        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = (
            self.duck_config.phase_frequency_factor_offset
        )

        # Optional expression features
        if self.duck_config.eyes:
            self.eyes = Eyes_cls(neopixels=self.duck_config.neopixels)
            if self.paused:
                self.eyes.set_standby(True)
                self.eyes.set_solid(False)
                self.eyes.set_color(self._ec(self.duck_config.eye_color_paused))
            else:
                self.eyes.set_standby(False)
                self.eyes.set_solid(False)
                self.eyes.set_color(self._ec(self.duck_config.eye_color_start))
        if self.duck_config.projector:
            self.projector = Projector_cls()
        if self.duck_config.speaker:
            self.sounds = Sounds_cls(volume=1.0, sound_directory=ASSETS_ROOT_PATH)
        if self.duck_config.antennas:
            self.antennas = Antennas_cls()

    @staticmethod
    def _ec(color):
        """Normalize an eye color from config (list or string) to what Eyes.set_color accepts."""
        return tuple(color) if isinstance(color, list) else color

    def get_obs(self):

        imu_data = self.imu.get_data()

        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs:
            logger.warning("dof_pos length %d != %d", len(dof_pos), self.num_dofs)
            return None

        if len(dof_vel) != self.num_dofs:
            logger.warning("dof_vel length %d != %d", len(dof_vel), self.num_dofs)
            return None

        cmds = self.last_commands

        feet_contacts = self.feet_contacts.get()

        obs = np.concatenate(
            [
                imu_data["gyro"],
                imu_data["accelero"],
                cmds,
                dof_pos - self.init_pos,
                dof_vel * 0.05,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                feet_contacts,
                self.imitation_phase,
            ]
        )

        return obs

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

        # lower head kps
        kps[5:9] = [8, 8, 8, 8]

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

    def get_phase_frequency_factor(self, x_velocity):

        max_phase_frequency = 1.2
        min_phase_frequency = 1.0

        # Perform linear interpolation
        freq = min_phase_frequency + (abs(x_velocity) / 0.15) * (
            max_phase_frequency - min_phase_frequency
        )

        return freq

    def _reset_fall_calibration(self):
        self._up_vector = None
        self._up_calib_acc = np.zeros(3)
        self._up_calib_count = 0
        self._fall_consecutive = 0

    def _update_fall_calibration(self):
        """Accumulate gravity samples while paused to establish the upright reference."""
        imu_data = self.imu.get_data()
        g = np.asarray(imu_data.get("gravity", [0, 0, 0]), dtype=float)
        g_norm = np.linalg.norm(g)
        if g_norm < 0.5:
            return
        self._up_calib_acc += g / g_norm
        self._up_calib_count += 1
        if self._up_calib_count >= self._up_calib_target:
            up = self._up_calib_acc / self._up_calib_count
            self._up_vector = up / np.linalg.norm(up)
            logger.info("Fall detection calibrated (up=%s)", np.around(self._up_vector, 3))
            # Reset so we keep refreshing the reference each subsequent pause
            self._up_calib_acc = np.zeros(3)
            self._up_calib_count = 0

    def _fall_detected(self):
        if self._up_vector is None:
            return False
        imu_data = self.imu.get_data()
        g = np.asarray(imu_data.get("gravity", [0, 0, 0]), dtype=float)
        if not np.all(np.isfinite(g)):
            self._fall_consecutive = 0
            return False
        g_norm = np.linalg.norm(g)
        if g_norm < 0.5:
            self._fall_consecutive = 0
            return False
        cos_angle = np.clip(abs(np.dot(g / g_norm, self._up_vector)), 0.0, 1.0)
        tilt_deg = np.degrees(np.arccos(cos_angle))
        if tilt_deg > self.duck_config.fall_threshold_deg:
            self._fall_consecutive += 1
            if self._fall_consecutive >= self._fall_consecutive_required:
                axis_labels = ["X", "Y", "Z"]
                worst = axis_labels[int(np.argmax(np.abs(g / g_norm - self._up_vector)))]
                logger.warning(
                    "tilt=%.1f° (%s-axis dominant) gravity=%s",
                    tilt_deg, worst, np.around(g, 3),
                )
                return True
        else:
            self._fall_consecutive = 0
        return False

    def run(self):
        signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))

        i = 0
        try:
            logger.info("Starting main loop")
            start_t = time.time()
            while True:
                left_trigger = 0
                right_trigger = 0
                t = time.time()

                if self.commands:
                    self.last_commands, self.buttons, left_trigger, right_trigger = (
                        self.xbox_controller.get_last_command()
                    )
                    if self.buttons.dpad_up.triggered:
                        self.phase_frequency_factor_offset += 0.05
                        logger.info(
                            "Phase frequency factor offset: %.3f",
                            self.phase_frequency_factor_offset,
                        )

                    if self.buttons.dpad_down.triggered:
                        self.phase_frequency_factor_offset -= 0.05
                        logger.info(
                            "Phase frequency factor offset: %.3f",
                            self.phase_frequency_factor_offset,
                        )

                    if self.buttons.LB.is_pressed:
                        self.phase_frequency_factor = 1.3
                    else:
                        self.phase_frequency_factor = 1.0

                    if self.buttons.X.triggered:
                        if self.duck_config.projector:
                            self.projector.switch()

                    if self.buttons.B.triggered:
                        if self.duck_config.speaker:
                            self.sounds.play_random_sound()

                    if self.duck_config.antennas:
                        self.antennas.set_position_left(right_trigger)
                        self.antennas.set_position_right(left_trigger)

                    if self.buttons.A.triggered:
                        if not self.motors_enabled:
                            logger.info("Motors are off – press START to re-enable first")
                        else:
                            self.paused = not self.paused
                            if self.paused:
                                logger.info("PAUSE")
                                if self.duck_config.eyes:
                                    self.eyes.set_standby(True)
                                    self.eyes.set_solid(False)
                                    self.eyes.set_color(self._ec(self.duck_config.eye_color_paused))
                            else:
                                self._fall_consecutive = 0
                                logger.info("UNPAUSE")
                                if self.duck_config.eyes:
                                    self.eyes.set_standby(False)
                                    self.eyes.set_solid(False)
                                    self.eyes.set_color(self._ec(self.duck_config.eye_color_start))

                    if self.buttons.START.triggered:
                        if self.motors_enabled:
                            logger.info("START – turning motors OFF")
                            self.hwi.turn_off()
                            self.motors_enabled = False
                            self.paused = True
                            if self.duck_config.eyes:
                                self.eyes.set_standby(False)
                                self.eyes.set_solid(True)
                                self.eyes.set_color(self._ec(self.duck_config.eye_color_off))
                        else:
                            logger.info("START – turning motors ON and reinitialising")
                            self.start()
                            self.motors_enabled = True
                            self.paused = True  # start paused; press A to begin walking
                            start_t = time.time()  # reset action-filter warmup timer
                            if self.duck_config.eyes:
                                self.eyes.set_standby(True)
                                self.eyes.set_solid(False)
                                self.eyes.set_color(self._ec(self.duck_config.eye_color_paused))

                # Fall detection — only while actively walking (motors on, not paused)
                if self.duck_config.fall_detection and self.motors_enabled and not self.paused and self._fall_detected():
                    logger.warning(
                        "FALL DETECTED (tilt > %d°) – turning off motors",
                        self.duck_config.fall_threshold_deg,
                    )
                    self.hwi.turn_off()
                    self.motors_enabled = False
                    self.paused = True
                    if self.duck_config.eyes:
                        self.eyes.set_standby(False)
                        self.eyes.set_solid(True)
                        self.eyes.set_color(self._ec(self.duck_config.eye_color_off))

                if self.paused:
                    if self.motors_enabled and self.duck_config.fall_detection:
                        self._update_fall_calibration()
                    time.sleep(0.1)
                    continue

                obs = self.get_obs()
                if obs is None:
                    continue
                logger.trace("obs: %s", np.around(obs, 3))

                self.imitation_i += 1 * (
                    self.phase_frequency_factor + self.phase_frequency_factor_offset
                )
                self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                self.imitation_phase = np.array(
                    [
                        np.cos(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                        np.sin(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                    ]
                )

                if self.save_obs:
                    self.saved_obs.append(obs)

                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        logger.info("Replay observations exhausted, stopping")
                        break

                action = self.policy.infer(obs)
                logger.trace("action: %s", np.around(action, 3))

                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                # action = np.zeros(10)

                self.motor_targets = self.init_pos + action * self.action_scale

                # self.motor_targets = np.clip(
                #     self.motor_targets,
                #     self.prev_motor_targets
                #     - self.max_motor_velocity * (1 / self.control_freq),  # control dt
                #     self.prev_motor_targets
                #     + self.max_motor_velocity * (1 / self.control_freq),  # control dt
                # )

                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if (
                        time.time() - start_t > 1
                    ):  # give time to the filter to stabilize
                        self.motor_targets = filtered_motor_targets

                self.prev_motor_targets = self.motor_targets.copy()

                head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets

                action_dict = make_action_dict(
                    self.motor_targets, list(self.hwi.joints.keys())
                )

                logger.trace("motor targets: %s", np.around(self.motor_targets, 3))
                self.hwi.set_position_all(action_dict)

                i += 1

                took = time.time() - t
                logger.trace("Loop %d: %.4fs (%.1f Hz)", i, took, 1 / took if took else 0)
                if (1 / self.control_freq - took) < 0:
                    logger.debug(
                        "Control budget exceeded by %.3fs", took - 1 / self.control_freq
                    )
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            if self.duck_config.antennas:
                self.antennas.stop()
            if self.duck_config.eyes:
                self.eyes.stop()
            if self.duck_config.projector:
                self.projector.stop()
            self.feet_contacts.stop()
            logger.info("Turning off motors")
            self.hwi.turn_off()

        if self.save_obs:
            pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/BEST_WALK_ONNX_2.onnx",
    )
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="save the run's observations",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override log level from config: TRACE, DEBUG, INFO, WARNING, ERROR",
    )

    args = parser.parse_args()

    # Resolve log level: CLI flag > duck_config.json > "INFO"
    log_level = "INFO"
    try:
        import json as _json
        _cfg = _json.load(open(args.duck_config_path))
        log_level = _cfg.get("log_level", log_level)
    except Exception:
        pass
    if args.log_level:
        log_level = args.log_level
    setup_logging(log_level)

    pid = [args.p, args.i, args.d]

    logger.debug("Args: %s", args)
    rl_walk = RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
    )
    logger.debug("RLWalk ready")
    rl_walk.run()


if __name__ == "__main__":
    main()
