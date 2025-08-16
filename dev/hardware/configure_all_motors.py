from pypot.feetech import FeetechSTS3215IO
import time

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

joints_inv = {v: k for k, v in joints.items()}

ids = list(joints.values())
io = FeetechSTS3215IO("/dev/ttyACM0")
for current_id in ids:
    print("Configuring", joints_inv[current_id])
    io.set_lock({current_id: 0})
    # io.set_mode({current_id: 0})
    io.set_maximum_acceleration({current_id: 0})
    io.set_acceleration({current_id: 0})
    # io.set_maximum_velocity({current_id: 0})
    # io.set_goal_speed({current_id: 0})
    io.set_P_coefficient({current_id: 32})
    io.set_I_coefficient({current_id: 0})
    io.set_D_coefficient({current_id: 0})

    io.set_lock({current_id: 1})
    time.sleep(1)
    input("Press any key to set this dof to 0 position ... Or press Ctrl+C to cancel")
    io.set_goal_position({current_id: 0})
