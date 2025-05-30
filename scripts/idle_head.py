import time
import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig

from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector

duck_config = DuckConfig()

if duck_config.speaker:
    sounds = Sounds(volume=1.0, sound_directory="../mini_bdx_runtime/assets/")
if duck_config.antennas:
    antennas = Antennas()
if duck_config.eyes:
    eyes = Eyes()
if duck_config.projector:
    projector = Projector()

hwi = HWI(duck_config)

kps = [8] * 14
kds = [0] * 14


hwi.set_kps(kps)
hwi.set_kds(kds)
hwi.turn_on()


# limits = {
#     "neck_pitch": [-20, 60],
#     "head_pitch": [-60, 45],
#     "head_yaw": [-60, 60],
#     "head_roll": [-20, 20],
# }


t0 = 0.0
while True:
    t = time.time() - t0

    head_yaw_pos_rad = np.deg2rad(20) * np.sin(2*np.pi*0.2*t)

    hwi.set_position("head_yaw", head_yaw_pos_rad)
    # hwi.set_position("head_roll", head_roll_pos_rad)
    # hwi.set_position("head_pitch", head_pitch_pos_rad)

    time.sleep(0.01)
