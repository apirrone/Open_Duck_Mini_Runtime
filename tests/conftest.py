import pytest
import os
import RPi.GPIO as GPIO
import logging
from pathlib import Path
import json

# Only import DuckConfig at the top, as it's used in actual_config fixture
from mini_bdx_runtime.duck_config import DuckConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_assets_dir():
    """Get the absolute path to the assets directory"""
    current_dir = Path(__file__).parent
    assets_dir = current_dir.parent / "mini_bdx_runtime" / "assets"
    if not assets_dir.exists():
        raise FileNotFoundError(f"Assets directory not found at {assets_dir}")
    return str(assets_dir)

@pytest.fixture(scope="session")
def actual_config():
    """
    Fixture to provide a DuckConfig instance for testing.
    First tries to use the user's duck_config.json, falls back to example_config.json
    """
    user_config = DuckConfig(ignore_default=True)
    if not user_config.default:
        logger.info("Using user's duck_config.json for testing")
        return user_config
    example_config_path = Path(__file__).parent.parent / "example_config.json"
    if not example_config_path.exists():
        raise FileNotFoundError(f"Neither user config nor example_config.json found at {example_config_path}")
    logger.info("Using example_config.json for testing")
    return DuckConfig(config_json_path=str(example_config_path), ignore_default=True)

@pytest.fixture(scope="session", autouse=True)
def gpio_setup_teardown():
    """Global fixture to handle GPIO setup and cleanup"""
    logger.info("Setting up GPIO for testing")
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    yield
    logger.info("Cleaning up GPIO")
    GPIO.cleanup()

@pytest.fixture
def antennas(actual_config):
    """
    Fixture to provide an Antennas instance.
    Skips tests if antennas feature is disabled in config.
    """
    if not actual_config.antennas:
        pytest.skip("Antennas feature is disabled in config")
    logger.info("Initializing Antennas")
    from mini_bdx_runtime.antennas import Antennas
    ant = Antennas()
    yield ant
    logger.info("Cleaning up Antennas")
    ant.stop()

@pytest.fixture(scope="function")
def eyes(actual_config):
    """
    Fixture to provide an Eyes instance.
    Skips tests if eyes feature is disabled in config.
    """
    if not actual_config.eyes:
        logger.info("Skipping eyes tests as feature is disabled")
        pytest.skip("Eyes feature is disabled in config")
    logger.info("Initializing Eyes")
    from mini_bdx_runtime.eyes import Eyes
    eyes_instance = Eyes()
    yield eyes_instance
    logger.info("Cleaning up Eyes")
    eyes_instance.cleanup()

@pytest.fixture(scope="module")
def sounds(actual_config):
    """
    Fixture to provide a Sounds instance with test audio files.
    Skips tests if speaker feature is disabled in config.
    Module scope since sound playback tests don't interfere with each other.
    """
    if not actual_config.speaker:
        pytest.skip("Speaker feature is disabled in config")
    logger.info("Initializing Sounds")
    from mini_bdx_runtime.sounds import Sounds
    assets_dir = get_assets_dir()
    sounds_instance = Sounds(volume=0.1, sound_directory=assets_dir)
    yield sounds_instance
    logger.info("Cleaning up Sounds")

@pytest.fixture(scope="module")
def projector(actual_config):
    """
    Fixture to provide a Projector instance.
    Skips tests if projector feature is disabled in config.
    Module scope since state is always cleaned up between tests.
    """
    if not actual_config.projector:
        pytest.skip("Projector feature is disabled in config")
    logger.info("Initializing Projector")
    from mini_bdx_runtime.projector import Projector
    proj = Projector()
    yield proj
    logger.info("Cleaning up Projector")

@pytest.fixture(scope="module")
def feet_contacts():
    """
    Fixture to provide a FeetContacts instance.
    Module scope since it's used for read-only operations.
    """
    logger.info("Initializing FeetContacts")
    from mini_bdx_runtime.feet_contacts import FeetContacts
    feet = FeetContacts()
    yield feet
    logger.info("Cleaning up FeetContacts")

@pytest.fixture(scope="module")
def hwi():
    """
    Fixture to provide a HWI instance.
    Module scope for hardware interface efficiency.
    """
    logger.info("Initializing HWI")
    from mini_bdx_runtime.rustypot_position_hwi import HWI
    hwi_instance = HWI()
    yield hwi_instance
    logger.info("Cleaning up HWI")

@pytest.fixture(scope="module")
def imu(actual_config):
    """
    Fixture to provide an IMU instance.
    Takes into account the imu_upside_down configuration.
    Module scope for hardware interface efficiency.
    """
    logger.info("Initializing IMU")
    from mini_bdx_runtime.raw_imu import Imu
    imu_instance = Imu(upside_down=actual_config.imu_upside_down)
    yield imu_instance
    logger.info("Cleaning up IMU")
