import pytest
import time
import numpy as np
import logging
from mini_bdx_runtime.antennas import Antennas

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SERVO_DELAY = 0.5  # Increased delay to allow servos to settle
SWEEP_STEPS = 10   # Number of steps in sweep test

def test_antenna_initialization(antennas):
    """Test that antennas are properly initialized"""
    assert antennas is not None
    assert hasattr(antennas, 'pwm1')
    assert hasattr(antennas, 'pwm2')
    logger.info("Antenna initialization test passed")

def test_antenna_movement(antennas):
    """Test basic antenna movement"""
    # Test left antenna
    logger.info("Testing left antenna movement")
    antennas.home()
    time.sleep(SERVO_DELAY)
    
    antennas.set_position_left(0)  # Center position
    time.sleep(SERVO_DELAY)
    
    antennas.set_position_left(0.5)  # Move right
    time.sleep(SERVO_DELAY)
    
    # Test right antenna
    logger.info("Testing right antenna movement")
    antennas.home()
    time.sleep(SERVO_DELAY)
    
    antennas.set_position_right(0)  # Center position
    time.sleep(SERVO_DELAY)
    
    antennas.set_position_right(-0.5)  # Move left
    time.sleep(SERVO_DELAY)

def test_left_antenna_sweep(antennas):
    """Test sweeping motion of left antenna from min to max"""
    logger.info("Testing left antenna sweep")
    positions = np.linspace(-1, 1, SWEEP_STEPS)
    
    # Forward sweep
    for pos in positions:
        antennas.set_position_left(pos)
        logger.info(f"Left antenna position: {pos:.2f}")
        time.sleep(SERVO_DELAY)
    
    # Return to home
    antennas.home()
    time.sleep(SERVO_DELAY)

def test_right_antenna_sweep(antennas):
    """Test sweeping motion of right antenna from min to max"""
    logger.info("Testing right antenna sweep")
    positions = np.linspace(-1, 1, SWEEP_STEPS)
    
    # Forward sweep
    for pos in positions:
        antennas.set_position_right(pos)
        logger.info(f"Right antenna position: {pos:.2f}")
        time.sleep(SERVO_DELAY)
    
    # Return to home
    antennas.home()
    time.sleep(SERVO_DELAY)

def test_synchronized_movement(antennas):
    """Test synchronized movement of both antennas"""
    logger.info("Testing synchronized antenna movement")
    antennas.home()
    time.sleep(SERVO_DELAY)
    
    positions = np.linspace(-1, 1, 5)
    for pos in positions:
        antennas.set_position_left(pos)
        antennas.set_position_right(pos)
        logger.info(f"Synchronized position: {pos:.2f}")
        time.sleep(SERVO_DELAY) 