"""
Find the offsets to set in self.joints_offsets in hwi_feetech_pwm_control.py
Calibrates all motors simultaneously

All credit to @https://github.com/Clancey 
During the rebase and refactor of the repo, this became standalone in the tools repo
"""

from open_duck_mini_runtime.hwi import HWI
from open_duck_mini_runtime.duck_config import DuckConfig
import time
import numpy as np

def main():
    print("======")
    print("BULK MOTOR OFFSET CALIBRATION")
    print("This script will help calibrate all motor offsets at once")
    print("======")
    
    # Create a dummy config with no offsets
    dummy_config = DuckConfig(config_json_path=None, ignore_default=True)
    
    print("Warning: This script will move the robot to its zero position quickly.")
    print("Make sure it is safe to do so and the robot has room to move.")
    print("======")
    
    input("Press Enter to start. At any time, press Ctrl+C to stop and turn off motors.")
    
    # Initialize hardware interface
    print("Initializing hardware interface...")
    hwi = HWI(dummy_config)
    
    # Set to zero position
    hwi.init_pos = hwi.zero_pos
    hwi.set_kds([0] * len(hwi.joints))
    hwi.turn_on()
    
    print("\nMoving all motors to zero position...")
    hwi.set_position_all(hwi.zero_pos)
    time.sleep(2)  # Give more time to reach position
    
    # Record current positions before disabling torque
    print("\nRecording current positions...")
    current_positions = {}
    for joint_name in hwi.joints.keys():
        joint_id = hwi.joints[joint_name]
        try:
            pos = hwi.io.read_present_position([joint_id])[0]
            current_positions[joint_name] = pos
            print(f"Motor '{joint_name}' (ID: {joint_id}) current position: {pos:.3f}")
        except Exception as e:
            print(f"Error reading position for '{joint_name}' (ID: {joint_id}): {e}")
            current_positions[joint_name] = None
    
    # Disable torque on all motors
    print("\nDisabling torque on all motors...")
    for joint_name, joint_id in hwi.joints.items():
        try:
            hwi.io.disable_torque([joint_id])
            print(f"Disabled torque for motor '{joint_name}' (ID: {joint_id})")
        except Exception as e:
            print(f"Error disabling torque for '{joint_name}' (ID: {joint_id}): {e}")
    
    # Instruct user to position all joints
    print("\n=== NOW POSITION ALL JOINTS ===")
    print("Move ALL motors to their desired zero positions simultaneously.")
    print("Take your time to position each joint carefully.")
    input("Press Enter when ALL joints are in their desired zero positions...")
    
    # Read new positions and calculate offsets
    print("\nCalculating offsets for all motors...")
    new_positions = {}
    offsets = {}
    
    for joint_name, joint_id in hwi.joints.items():
        try:
            new_pos = hwi.io.read_present_position([joint_id])[0]
            new_positions[joint_name] = new_pos
            
            if current_positions[joint_name] is not None:
                offset = new_pos - current_positions[joint_name]
                offsets[joint_name] = offset
                print(f"Motor '{joint_name}' (ID: {joint_id}) offset: {offset:.3f}")
            else:
                print(f"Cannot calculate offset for '{joint_name}' (ID: {joint_id}) - initial position unknown")
                offsets[joint_name] = 0
        except Exception as e:
            print(f"Error reading new position for '{joint_name}' (ID: {joint_id}): {e}")
            offsets[joint_name] = 0
    
    # Apply offsets
    print("\nApplying offsets to all motors...")
    for joint_name, offset in offsets.items():
        hwi.joints_offsets[joint_name] = offset
    
    # Re-enable torque on all motors
    print("\nRe-enabling torque on all motors...")
    for joint_name, joint_id in hwi.joints.items():
        try:
            hwi.io.enable_torque([joint_id])
            print(f"Enabled torque for motor '{joint_name}' (ID: {joint_id})")
        except Exception as e:
            print(f"Error enabling torque for '{joint_name}' (ID: {joint_id}): {e}")
    
    # Move to zero position with offsets applied
    print("\nMoving to zero position with offsets applied...")
    hwi.set_position_all(hwi.zero_pos)
    time.sleep(2)
    
    # Verify results
    confirm = input("\nAre all motors properly aligned? (y/n): ").lower()
    if confirm == 'y':
        print("\n=== CALIBRATION SUCCESSFUL ===")
        print("Copy these offsets to your configuration file:")
        print("\nOffsets:")
        for joint_name, offset in offsets.items():
            print(f'"{joint_name}": {offset:.6f},')
        
        # Offer to update duck_config.json
        update = input("\nWould you like to update ~/duck_config.json with these offsets? (y/N): ").lower()
        if update == 'y':
            import os, json
            config_path = os.path.expanduser("~/duck_config.json")
            try:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                else:
                    config = {}
                config['joints_offsets'] = {k: float(f"{v:.6f}") for k, v in offsets.items()}
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"\nSuccessfully updated {config_path} with new offsets.")
            except Exception as e:
                print(f"\nFailed to update {config_path}: {e}")
        else:
            print("\nOffsets not written to config file.")
    else:
        print("\nCalibration not confirmed. You may need to run the one-by-one calibration instead.")
    
    # Turn off motors
    print("\nTurning off all motors...")
    hwi.turn_off()
    print("Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user. Turning off motors...")
        try:
            hwi = HWI(DuckConfig(config_json_path=None, ignore_default=True))
            hwi.turn_off()
            print("Motors turned off successfully.")
        except Exception as e:
            print(f"Error turning off motors: {e}")
