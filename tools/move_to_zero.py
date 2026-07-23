"""
Slowly move every servo to its factory-defined zero position (0.0 degrees ==
raw center, ~2048 steps), independent of any calibration offsets in
duck_config.json.

Safety:
- Motion is done in small, software-paced steps (STEP_DEG per tick) rather
  than trusting the servo's internal speed register, so the move is always
  slow regardless of servo firmware quirks.
- After each step we read back the actual position and compare it to the
  commanded target. If a joint falls more than MAX_BACKLASH_DEG behind
  target, it is treated as resistance/obstruction (or a bug commanding too
  large a jump): that joint is immediately frozen at its actual position and
  excluded from further motion, while the other joints keep creeping to zero.
"""

import time

from pypot.feetech import FeetechSTS3215IO

PORT = "/dev/ttyACM0"
BAUDRATE = 1_000_000

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

STEP_DEG = 1.0  # max commanded movement per tick, keeps the move slow and bounded
STEP_DELAY_S = 0.08  # pause after each step for the servo to catch up (~12 deg/s)
MAX_BACKLASH_DEG = 3.0  # max allowed gap between commanded and actual position
MAX_ITERATIONS = 1000  # hard cap so a bug can't spin the loop forever


def move_all_to_zero(io, joints):
    motor_ids = list(joints.values())

    print(f"Connecting to {len(motor_ids)} motors...")
    io.enable_torque(motor_ids)
    print("Torque enabled.")

    positions = dict(zip(joints.keys(), io.get_present_position(motor_ids)))
    print("Starting positions (deg):")
    for name, pos in positions.items():
        print(f"  {name:<15} {pos:+7.2f}")

    remaining = set(joints.keys())
    resisted = set()

    iteration = 0
    while remaining and iteration < MAX_ITERATIONS:
        iteration += 1

        names = list(remaining)
        targets = {}
        for name in names:
            cur = positions[name]
            if abs(cur) <= STEP_DEG:
                targets[name] = 0.0
            elif cur > 0:
                targets[name] = cur - STEP_DEG
            else:
                targets[name] = cur + STEP_DEG

        ids = [joints[name] for name in names]
        io.set_goal_position({joints[name]: targets[name] for name in names})
        time.sleep(STEP_DELAY_S)

        actual = io.get_present_position(ids)
        for name, actual_pos in zip(names, actual):
            target = targets[name]
            error = abs(actual_pos - target)

            if error > MAX_BACKLASH_DEG:
                print(
                    f"STOP {name}: resisted move (target {target:+.2f}, "
                    f"actual {actual_pos:+.2f}, error {error:.2f} > "
                    f"max backlash {MAX_BACKLASH_DEG}). Holding here."
                )
                io.set_goal_position({joints[name]: actual_pos})
                remaining.discard(name)
                resisted.add(name)
            else:
                positions[name] = actual_pos
                if target == 0.0:
                    remaining.discard(name)

    if iteration >= MAX_ITERATIONS and remaining:
        print(f"WARNING: hit MAX_ITERATIONS with joints still moving: {sorted(remaining)}")

    if resisted:
        print(f"\nDone. {len(resisted)} joint(s) halted early due to resistance: {sorted(resisted)}")
    else:
        print("\nAll motors reached zero position.")


def discover_joints(io, joints):
    """Ping each configured joint and return only the ones that respond."""
    present = {name: mid for name, mid in joints.items() if io.ping(mid)}
    missing = [name for name in joints if name not in present]
    if missing:
        print(f"Not responding, skipping: {sorted(missing)}")
    return present


def main():
    io = FeetechSTS3215IO(PORT, baudrate=BAUDRATE, use_sync_read=False)
    print("Scanning for connected motors...")
    present_joints = discover_joints(io, joints)
    if not present_joints:
        print("No motors responded. Check the connection and try again.")
        return

    input(
        f"This will slowly move {len(present_joints)}/{len(joints)} connected "
        "motor(s) to their factory zero position. Make sure the robot has "
        "room to move. Press Enter to start (Ctrl+C at any time to freeze "
        "in place)..."
    )
    move_all_to_zero(io, present_joints)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Motors are left holding their last commanded position.")
