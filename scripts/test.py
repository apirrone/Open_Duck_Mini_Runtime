from mini_bdx_runtime.hwi import HWI

hwi = HWI("/dev/ttyUSB0")

while True:
    print(hwi.get_present_velocities()[4])
