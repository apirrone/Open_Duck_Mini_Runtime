import json
import logging
from typing import Optional
import os

logger = logging.getLogger(__name__)

HOME_DIR = os.path.expanduser("~")

# some neopixels are GRBW, some are RGBW, some are RGB...
# fill in the order of whatever you picked here
LED_ORDER: str = os.getenv("ODUCK_LED_ORDER", "GRBW").upper()
LED_WHITE_MODE: str = os.getenv("ODUCK_LED_WHITE_MODE", "W").upper()


class DuckConfig:

    def __init__(
        self,
        config_json_path: Optional[str] = f"{HOME_DIR}/duck_config.json",
        ignore_default: bool = False,
    ):
        """
        Looks for duck_config.json in the home directory by default.
        If not found, uses default values.
        """
        self.default = False
        try:
            self.json_config = (
                json.load(open(config_json_path, "r")) if config_json_path else {}
            )
        except FileNotFoundError:
            logger.warning("config json not found at %s, using defaults", config_json_path)
            self.json_config = {}
            self.default = True

        if config_json_path is None:
            logger.warning("no config json path provided, using defaults")
            self.default = True

        if self.default and not ignore_default:
            logger.warning(
                "Running with default values — this probably won't work well. "
                "Please create a duck_config.json file."
            )
            res = input("Do you still want to run? (y/N) ")
            if res.lower() != "y":
                logger.info("Exiting at user request")
                exit(1)

        self.log_level = self.json_config.get("log_level", "INFO")
        self.start_paused = self.json_config.get("start_paused", False)
        self.imu_upside_down = self.json_config.get("imu_upside_down", False)
        self.fall_detection = self.json_config.get("fall_detection", True)
        self.fall_threshold_deg = self.json_config.get("fall_threshold_deg", 45)
        self.phase_frequency_factor_offset = self.json_config.get(
            "phase_frequency_factor_offset", 0.0
        )

        eye_colors = self.json_config.get("eye_colors", {})
        self.eye_color_start = eye_colors.get("start", [255, 255, 255])   # white — walking
        self.eye_color_paused = eye_colors.get("paused", [255, 105, 180]) # hot pink — paused
        self.eye_color_off = eye_colors.get("off", [255, 0, 0])           # red — motors off

        expression_features = self.json_config.get("expression_features", {})

        self.eyes = expression_features.get("eyes", False)
        self.projector = expression_features.get("projector", False)
        self.antennas = expression_features.get("antennas", False)
        self.speaker = expression_features.get("speaker", False)
        self.microphone = expression_features.get("microphone", False)
        self.camera = expression_features.get("camera", False)

        self.led_order = self.json_config.get("led_order", "GRBW")

        # default joints offsets are 0.0
        self.joints_offset = self.json_config.get(
            "joints_offsets",
            {
                "left_hip_yaw": 0.0,
                "left_hip_roll": 0.0,
                "left_hip_pitch": 0.0,
                "left_knee": 0.0,
                "left_ankle": 0.0,
                "neck_pitch": 0.0,
                "head_pitch": 0.0,
                "head_yaw": 0.0,
                "head_roll": 0.00,
                "right_hip_yaw": 0.0,
                "right_hip_roll": 0.0,
                "right_hip_pitch": 0.0,
                "right_knee": 0.0,
                "right_ankle": 0.0,
            },
        )
