from pypot.feetech import FeetechSTS3215IO

io = FeetechSTS3215IO(
    "/dev/ttyACM0",
    baudrate=1000000,
    use_sync_read=True,
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
    # "left_antenna": None,
    # "right_antenna": None,
    "right_hip_yaw": 10,
    "right_hip_roll": 11,
    "right_hip_pitch": 12,
    "right_knee": 13,
    "right_ankle": 14,
}


voltages = io.get_present_voltage(list(joints.values()))
for i, name in enumerate(joints.keys()):
    print(name, round(voltages[i] * 0.1, 2), "V")
