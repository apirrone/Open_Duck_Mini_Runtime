from open_duck_mini_runtime.hwi import HWI
from open_duck_mini_runtime.duck_config import DuckConfig
import time

duck_config = DuckConfig()

hwi = HWI(duck_config)
hwi.turn_off()
time.sleep(1)
