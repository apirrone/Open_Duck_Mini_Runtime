from open_duck_mini_runtime.hwi import HWI
from open_duck_mini_runtime.duck_config import DuckConfig
import time


def main():
    duck_config = DuckConfig()

    hwi = HWI(duck_config)
    hwi.turn_on()
    time.sleep(1)


if __name__ == "__main__":
    main()
