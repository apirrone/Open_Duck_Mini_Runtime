"""
Debug script to check all motors in the robot.
Verifies each motor is accessible and allows testing movement.
"""

from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig
import time
import numpy as np
import traceback

def main():
    print("Initializing hardware interface...")
    try:
        # Initialize with default USB port - you might need to modify this
        print("Attempting to connect to motor controller...")
        duck_config = DuckConfig()  # Create default configuration
        
        # Initialize with duck_config
        print("Attempting to connect to motor controller...")
        hwi = HWI(duck_config=duck_config)
        print("Successfully connected to hardware!")
    except Exception as e:
        print(f"Error connecting to hardware: {e}")
        print(f"Error details: {traceback.format_exc()}")
        print("Check that the robot is powered on and USB connection is correct.")
        return

    # Turn on with low torque for safety - ONE BY ONE
    print("\nTurning on motors with low torque (one by one)...")
    unresponsive_motors = []
    
    for joint_name, joint_id in hwi.joints.items():
        try:
            print(f"Setting low torque for motor '{joint_name}' (ID: {joint_id})...")
            hwi.io.set_kps([joint_id], [hwi.low_torque_kps[0]])
            print(f"✓ Low torque set successfully for motor '{joint_name}' (ID: {joint_id}).")
        except Exception as e:
            print(f"✗ Error setting low torque for motor '{joint_name}' (ID: {joint_id}): {e}")
            print(f"Error details: {traceback.format_exc()}")
            unresponsive_motors.append((joint_name, joint_id))
    
    # Check if all motors are responsive
    print("\nChecking if all motors are responsive...")
    
    for joint_name, joint_id in hwi.joints.items():
        # Skip motors that already failed
        if (joint_name, joint_id) in unresponsive_motors:
            print(f"Skipping previously unresponsive motor: '{joint_name}' (ID: {joint_id})")
            continue
            
        print(f"Attempting to read position from motor '{joint_name}' (ID: {joint_id})...")
        try:
            # Try to read the position to check if motor is responsive
            position = hwi.io.read_present_position([joint_id])
            print(f"✓ Motor '{joint_name}' (ID: {joint_id}) is responsive. Position: {position[0]:.3f}")
        except Exception as e:
            print(f"✗ Error accessing motor '{joint_name}' (ID: {joint_id}): {e}")
            print(f"Error details for motor {joint_id}: {traceback.format_exc()}")
            unresponsive_motors.append((joint_name, joint_id))
    
    if unresponsive_motors:
        print("\nWARNING: Some motors are not responsive!")
        print("Unresponsive motors:", unresponsive_motors)
        continue_anyway = input("Do you want to continue anyway? (y/n): ").lower()
        if continue_anyway != 'y':
            print("Exiting...")
            try:
                print("Attempting to turn off responsive motors before exiting...")
                for joint_name, joint_id in hwi.joints.items():
                    if (joint_name, joint_id) not in unresponsive_motors:
                        try:
                            hwi.io.disable_torque([joint_id])
                            print(f"Disabled torque for motor '{joint_name}' (ID: {joint_id})")
                        except:
                            pass
            except:
                pass
            return

    # Test moving each motor individually
    print("\n--- Motor Movement Test ---")
    print("This will move each motor by a small amount to check if it's working correctly.")
    input("Press Enter to begin the movement test...")
    
    for joint_name, joint_id in hwi.joints.items():
        # Skip unresponsive motors
        if (joint_name, joint_id) in unresponsive_motors:
            print(f"Skipping unresponsive motor: '{joint_name}' (ID: {joint_id})")
            continue

        print(f"\nTesting motor: '{joint_name}' (ID: {joint_id})")
        test_this_motor = input(f"Test this motor? (Enter/y for yes, n to skip, q to quit): ").lower()
        
        if test_this_motor == 'q':
            print("Exiting movement test...")
            break
            
        if test_this_motor == 'n':
            print(f"Skipping '{joint_name}' (ID: {joint_id})")
            continue
        
        try:
            # Get current position
            print(f"Reading current position from motor '{joint_name}' (ID: {joint_id})...")
            current_position = hwi.io.read_present_position([joint_id])[0]
            print(f"Current position: {current_position:.3f}")
            
            # Calculate test position (move by 0.1 radians)
            test_position = current_position + 0.1
            
            # Move to test position
            print(f"Moving motor '{joint_name}' (ID: {joint_id}) to test position: {test_position:.3f}...")
            hwi.io.write_goal_position([joint_id], [test_position])
            time.sleep(1)  # Wait for movement
            
            # Read new position
            print(f"Reading new position from motor '{joint_name}' (ID: {joint_id})...")
            new_position = hwi.io.read_present_position([joint_id])[0]
            print(f"New position: {new_position:.3f}")
            
            # Return to original position
            print(f"Returning motor '{joint_name}' (ID: {joint_id}) to original position...")
            hwi.io.write_goal_position([joint_id], [current_position])
            time.sleep(1)  # Wait for movement
            
            # No confirmation question, just assume success
            print(f"✓ Motor '{joint_name}' (ID: {joint_id}) movement test completed.")
            
        except Exception as e:
            print(f"Error testing motor '{joint_name}' (ID: {joint_id}): {e}")
            print(f"Error details: {traceback.format_exc()}")
    
    # Turn off motors
    print("\nTurning off motors one by one...")
    for joint_name, joint_id in hwi.joints.items():
        if (joint_name, joint_id) in unresponsive_motors:
            print(f"Skipping turning off unresponsive motor: '{joint_name}' (ID: {joint_id})")
            continue
            
        try:
            print(f"Disabling torque for motor '{joint_name}' (ID: {joint_id})...")
            hwi.io.disable_torque([joint_id])
            print(f"✓ Motor '{joint_name}' (ID: {joint_id}) turned off successfully.")
        except Exception as e:
            print(f"✗ Error turning off motor '{joint_name}' (ID: {joint_id}): {e}")
            print(f"Error details: {traceback.format_exc()}")
    
    print("\nMotor test completed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Attempting to turn off motors...")
        try:
            print("Initializing HWI to turn off motors...")
            hwi = HWI()
            for joint_name, joint_id in hwi.joints.items():
                try:
                    print(f"Turning off motor '{joint_name}' (ID: {joint_id})...")
                    hwi.io.disable_torque([joint_id])
                    print(f"✓ Motor '{joint_name}' (ID: {joint_id}) turned off successfully.")
                except Exception as e:
                    print(f"✗ Error turning off motor '{joint_name}' (ID: {joint_id}): {e}")
        except Exception as e:
            print(f"Error initializing HWI to turn off motors: {e}")
            print(f"Error details: {traceback.format_exc()}")