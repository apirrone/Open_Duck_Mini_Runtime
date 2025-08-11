from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig
import time

duck_config = DuckConfig()

hwi = HWI(duck_config)
hwi.turn_on()
time.sleep(1)
