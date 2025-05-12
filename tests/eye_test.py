import pytest
import time
import RPi.GPIO as GPIO
from mini_bdx_runtime.eyes import Eyes, LEFT_EYE_GPIO, RIGHT_EYE_GPIO

eye_delay = 2.5

@pytest.fixture
def eyes():
    """Fixture to create and cleanup Eyes instance"""
    eyes_instance = Eyes()
    yield eyes_instance
    eyes_instance.cleanup()  # Ensure cleanup after each test

def test_eyes_initialization(eyes):
    """Test that eyes are properly initialized"""
    assert eyes is not None
    
    # Verify initial state (both eyes should be ON)
    states = eyes.get_eye_states()
    assert states['left'] == GPIO.HIGH
    assert states['right'] == GPIO.HIGH
    time.sleep(eye_delay)
    # Verify blink thread is not started yet
    assert not hasattr(eyes, '_blink_thread')

def test_individual_eye_control(eyes):
    """Test individual eye control"""
    # Start with both eyes off
    eyes.set_both_eyes(GPIO.LOW)
    states = eyes.get_eye_states()
    assert states['left'] == GPIO.LOW
    assert states['right'] == GPIO.LOW
    time.sleep(eye_delay)

    # Test left eye
    eyes.set_left_eye(GPIO.HIGH)
    states = eyes.get_eye_states()
    assert states['left'] == GPIO.HIGH
    assert states['right'] == GPIO.LOW
    time.sleep(eye_delay)


    # Test right eye
    eyes.set_right_eye(GPIO.HIGH)
    states = eyes.get_eye_states()
    assert states['left'] == GPIO.HIGH
    assert states['right'] == GPIO.HIGH
    time.sleep(eye_delay)

    # Turn off left eye
    eyes.set_left_eye(GPIO.LOW)
    states = eyes.get_eye_states()
    assert states['left'] == GPIO.LOW
    assert states['right'] == GPIO.HIGH
    time.sleep(eye_delay)

    # Turn off right eye
    eyes.set_right_eye(GPIO.LOW)
    states = eyes.get_eye_states()
    assert states['left'] == GPIO.LOW
    assert states['right'] == GPIO.LOW
    time.sleep(eye_delay)

def test_blink_thread_management(eyes):
    """Test starting and stopping the blink thread"""
    # Start the blink thread
    eyes.start_blink_thread()
    assert hasattr(eyes, '_blink_thread')
    assert eyes._blink_thread.is_alive()
    assert eyes.blink_duration == 0.1
    
    # Stop the blink thread
    eyes.stop_blink_thread()
    assert not eyes._blink_thread.is_alive()


def test_eyes_blink_cycle(eyes):
    """Test a complete blink cycle for both eyes"""
    # Start the blink thread
    eyes.start_blink_thread()
    
    # Initial state should be ON
    states = eyes.get_eye_states()
    assert states['left'] == GPIO.HIGH
    assert states['right'] == GPIO.HIGH
    
    # Wait for at least one blink cycle
    time.sleep(eyes.blink_duration * 2)
    
    # Check multiple times to catch different states
    states_history = []
    for _ in range(10):
        states = eyes.get_eye_states()
        states_history.append((states['left'], states['right']))
        time.sleep(0.1)
    
    # Verify that we saw at least one blink (LOW state)
    assert any(state[0] == GPIO.LOW for state in states_history), "Left eye never blinked"
    assert any(state[1] == GPIO.LOW for state in states_history), "Right eye never blinked"
    
    # Verify that eyes return to ON state
    assert any(state[0] == GPIO.HIGH for state in states_history), "Left eye never turned back on"
    assert any(state[1] == GPIO.HIGH for state in states_history), "Right eye never turned back on"
    
    # Stop the blink thread
    eyes.stop_blink_thread()

def test_eyes_synchronization(eyes):
    """Test that both eyes blink synchronously"""
    # Start the blink thread
    eyes.start_blink_thread()
    
    states_history = []
    # Capture several states to verify synchronization
    for _ in range(10):
        states = eyes.get_eye_states()
        states_history.append((states['left'], states['right']))
        time.sleep(0.1)
    
    # Verify that eyes are always in the same state
    for left_state, right_state in states_history:
        assert left_state == right_state, "Eyes are not synchronized"
    
    # Stop the blink thread
    eyes.stop_blink_thread()

def test_blink_duration(eyes):
    """Test that blink duration is approximately correct"""
    # Start the blink thread
    eyes.start_blink_thread()
    
    # Wait for a blink to start
    while eyes.get_eye_states()['left'] == GPIO.HIGH:
        time.sleep(0.01)
    
    # Measure blink duration
    start_time = time.time()
    while eyes.get_eye_states()['left'] == GPIO.LOW:
        time.sleep(0.01)
    blink_time = time.time() - start_time
    
    # Verify blink duration (with some tolerance)
    assert abs(blink_time - eyes.blink_duration) < 0.05, "Blink duration is not within expected range"
    
    # Stop the blink thread
    eyes.stop_blink_thread()

def test_cleanup(eyes):
    """Test that eyes can be properly cleaned up"""
    # Start the blink thread
    eyes.start_blink_thread()
    
    # Set eyes to known state
    eyes.set_both_eyes(GPIO.HIGH)
    time.sleep(0.1)
    
    # Cleanup
    eyes.cleanup()
    
    # Verify both eyes are off after cleanup
    states = eyes.get_eye_states()
    assert states['left'] == GPIO.LOW
    assert states['right'] == GPIO.LOW
    
    # Verify blink thread has stopped
    assert not eyes._blink_thread.is_alive()

def test_eyes_random_blink_timing(eyes):
    """Test that eyes blink at random intervals"""
    # Start the blink thread
    eyes.start_blink_thread()
    
    # Capture blink timings
    blink_times = []
    start_time = time.time()
    last_state = GPIO.HIGH
    
    # Monitor for a few blinks
    while len(blink_times) < 3 and time.time() - start_time < 10:
        current_state = eyes.get_eye_states()['left']
        if current_state != last_state and current_state == GPIO.LOW:
            blink_times.append(time.time() - start_time)
        last_state = current_state
        time.sleep(0.01)
    
    # Test should complete before timeout
    assert time.time() - start_time < 10, "Failed to detect enough blinks"
    
    # Verify random intervals between blinks
    if len(blink_times) >= 2:
        intervals = [blink_times[i+1] - blink_times[i] for i in range(len(blink_times)-1)]
        # Check that intervals are not all the same (allowing for small timing variations)
        assert not all(abs(intervals[0] - interval) < 0.1 for interval in intervals[1:]), "Blink intervals appear to be constant"
    
    # Stop the blink thread
    eyes.stop_blink_thread() 