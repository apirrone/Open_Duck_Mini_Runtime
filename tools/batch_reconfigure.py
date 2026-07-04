from pypot.feetech import FeetechSTS3215IO
import time

io = FeetechSTS3215IO(
    "/dev/ttyACM0",
    baudrate=1000000,
    use_sync_read=False,
)

joints = {
    "left_hip_yaw": 20, "left_hip_roll": 21, "left_hip_pitch": 22,
    "left_knee": 23, "left_ankle": 24, "neck_pitch": 30,
    "head_pitch": 31, "head_yaw": 32, "head_roll": 33,
    "right_hip_yaw": 10, "right_hip_roll": 11, "right_hip_pitch": 12,
    "right_knee": 13, "right_ankle": 14,
}

joint_ids = list(joints.values())

print(f"Connecting to {len(joint_ids)} motors to update configurations...")

# 1. Create the batch dictionaries
# This creates a dictionary like: {20: 0, 21: 0, 22: 0 ...} for all 14 motors
unlock_payload = {motor_id: 0 for motor_id in joint_ids}
zero_payload = {motor_id: 0 for motor_id in joint_ids}
p_payload = {motor_id: 32 for motor_id in joint_ids}

print("Unlocking EEPROM...")
io.set_lock(unlock_payload)

print("Setting Mode, Acceleration, and PID values...")
io.set_mode(zero_payload)
io.set_maximum_acceleration(zero_payload)
io.set_acceleration(zero_payload)
io.set_P_coefficient(p_payload)
io.set_I_coefficient(zero_payload)
io.set_D_coefficient(zero_payload)

print("Configurations sent. Verifying...")
time.sleep(0.5)

# 3. Verify the changes (Pull the P coefficients to check)
# get_P_coefficient accepts a list of IDs and returns a tuple of their values
current_p_values = io.get_P_coefficient(joint_ids)

print("\n=== Verification ===")
for name, motor_id, p_val in zip(joints.keys(), joint_ids, current_p_values):
    status = "OK" if p_val == 32 else "FAILED"
    print(f"{name} (ID {motor_id}): P = {p_val} [{status}]")

print("=== Batch Configuration Complete ===")
