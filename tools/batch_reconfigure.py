from pypot.feetech import FeetechSTS3215IO
import time

io = FeetechSTS3215IO(
    "/dev/ttyACM0",
    baudrate=1000000,
    use_sync_read=False,
)

joints = {
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

joint_ids = list(joints.values())

# Source of truth: what every motor should look like (based on v2 + sync fix)
EXPECTED = {
    "return_delay_time": 0,  # 0 = respond immediately; delays > 0 break SYNC_READ
    "response_status_level": 1,  # 1 = respond only to READ; 2 = respond to all (breaks SYNC_WRITE)
    "mode": 0,  # 0 = position control
    "maximum_acceleration": 0,
    "acceleration": 0,
    "P_coefficient": 32,
    "I_coefficient": 0,
    "D_coefficient": 0,
}


def read_all(io, ids):
    return {
        "return_delay_time": io.get_return_delay_time(ids),
        "response_status_level": io.get_response_status_level(ids),
        "mode": io.get_mode(ids),
        "maximum_acceleration": io.get_maximum_acceleration(ids),
        "acceleration": io.get_acceleration(ids),
        "P_coefficient": io.get_P_coefficient(ids),
        "I_coefficient": io.get_I_coefficient(ids),
        "D_coefficient": io.get_D_coefficient(ids),
    }


print(f"Connecting to {len(joint_ids)} motors...")
current = read_all(io, joint_ids)

# --- Sanity check: one line per motor ---
print("\n=== Sanity Check ===")
print(f"{'Motor':<20}  {'Result':<8}  Mismatches")
print("-" * 70)

any_mismatch = False
for i, (name, motor_id) in enumerate(joints.items()):
    mismatches = [
        f"{reg}={current[reg][i]} (want {exp})"
        for reg, exp in EXPECTED.items()
        if current[reg][i] != exp
    ]
    if mismatches:
        any_mismatch = True
        print(f"{name:<20}  FAIL      {', '.join(mismatches)}")
    else:
        print(f"{name:<20}  OK")

if not any_mismatch:
    print("\nAll motors match expected config — no reconfiguration needed.")
else:
    print("\n*** Mismatches found. Applying fixes... ***")

    print("Unlocking EEPROM...")
    io.set_lock({motor_id: 0 for motor_id in joint_ids})

    print("Applying expected values...")
    io.set_return_delay_time(
        {motor_id: EXPECTED["return_delay_time"] for motor_id in joint_ids}
    )
    io.set_response_status_level(
        {motor_id: EXPECTED["response_status_level"] for motor_id in joint_ids}
    )
    io.set_mode({motor_id: EXPECTED["mode"] for motor_id in joint_ids})
    io.set_maximum_acceleration(
        {motor_id: EXPECTED["maximum_acceleration"] for motor_id in joint_ids}
    )
    io.set_acceleration({motor_id: EXPECTED["acceleration"] for motor_id in joint_ids})
    io.set_P_coefficient(
        {motor_id: EXPECTED["P_coefficient"] for motor_id in joint_ids}
    )
    io.set_I_coefficient(
        {motor_id: EXPECTED["I_coefficient"] for motor_id in joint_ids}
    )
    io.set_D_coefficient(
        {motor_id: EXPECTED["D_coefficient"] for motor_id in joint_ids}
    )

    print("Re-locking EEPROM...")
    io.set_lock({motor_id: 1 for motor_id in joint_ids})

    time.sleep(0.5)

    # --- Verify ---
    verified = read_all(io, joint_ids)

    print("\n=== Post-fix Verification ===")
    print(f"{'Motor':<20}  {'Result':<8}  Mismatches")
    print("-" * 70)

    all_ok = True
    for i, (name, motor_id) in enumerate(joints.items()):
        mismatches = [
            f"{reg}={verified[reg][i]} (want {exp})"
            for reg, exp in EXPECTED.items()
            if verified[reg][i] != exp
        ]
        if mismatches:
            all_ok = False
            print(f"{name:<20}  FAIL      {', '.join(mismatches)}")
        else:
            print(f"{name:<20}  OK")

    if all_ok:
        print("\n=== All motors configured successfully ===")
    else:
        print(
            "\n=== WARNING: Some motors did not apply correctly — check connections ==="
        )
