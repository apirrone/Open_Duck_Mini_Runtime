print("WARNING Don't use this script !!!")
# exit()

from pypot.feetech import FeetechSTS3215IO
import time

ids = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33]
io = FeetechSTS3215IO("/dev/ttyACM0")
for current_id in ids:
    io.set_lock({current_id: 0})
    # io.set_mode({current_id: 0})
    io.set_maximum_acceleration({current_id: 0})
    io.set_acceleration({current_id: 0})
    io.set_maximum_velocity({current_id: 0})
    io.set_goal_speed({current_id: 0})
    io.set_P_coefficient({current_id: 32})
    io.set_I_coefficient({current_id: 0})
    io.set_D_coefficient({current_id: 0})

    io.set_goal_position({current_id: 0})
    io.set_lock({current_id: 1})
    time.sleep(1)
