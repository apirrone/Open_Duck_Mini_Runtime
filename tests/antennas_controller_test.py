from mini_bdx_runtime.xbox_controller import XBoxController
from mini_bdx_runtime.antennas import Antennas
import time

controller = XBoxController(60)

antennas = Antennas()


while True:

    _, _, left_trigger, right_trigger = controller.get_last_command()

    antennas.set_position_left(right_trigger)
    antennas.set_position_right(left_trigger)

    # print(left_trigger, right_trigger)
    time.sleep(1 / 50)
