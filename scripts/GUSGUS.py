from mini_bdx_runtime.io_330 import Dxl330IO
import time
import numpy as np

dxl_io = Dxl330IO("/dev/ttyUSB0", baudrate=2000000, use_sync_read=True)
dxl_io.enable_torque([31, 32])

while True:
    print(dxl_io.get_present_position([31, 32]))
    # dxl_io.set_goal_position({31: 0, 32: 0})
    val = 30 * np.sin(2 * np.pi * 4 * time.time())
    dxl_io.set_goal_position({31: val, 32: -val})
    # time.sleep(0.1)
