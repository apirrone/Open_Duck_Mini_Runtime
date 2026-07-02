"""
Controller diagnostic tool.

Run with:  uv run src/open_duck_mini_runtime/controller_info.py

Prints real-time axis and button state for the connected joystick so you can
identify which index maps to each physical button/axis and build a custom
button_map / axis_map for duck_config.json.

"""

import pygame
import time
import sys


def main():
    pygame.init()
    pygame.joystick.init()

    n = pygame.joystick.get_count()
    if n == 0:
        print("No joystick detected. Plug in your controller and try again.")
        sys.exit(1)

    if n > 1:
        print(f"Found {n} joysticks:")
        for i in range(n):
            j = pygame.joystick.Joystick(i)
            print(f"  [{i}] {j.get_name()}")
        idx = int(input("Select joystick index: "))
    else:
        idx = 0

    js = pygame.joystick.Joystick(idx)
    js.init()

    print(f"\nConnected: '{js.get_name()}'")
    print(f"  Axes:    {js.get_numaxes()}")
    print(f"  Buttons: {js.get_numbuttons()}")
    print(f"  Hats:    {js.get_numhats()}")
    print("\nPress buttons / move sticks. Ctrl-C to quit.\n")

    prev_buttons = [False] * js.get_numbuttons()
    prev_axes = [0.0] * js.get_numaxes()

    try:
        while True:
            pygame.event.pump()

            for i in range(js.get_numbuttons()):
                state = bool(js.get_button(i))
                if state != prev_buttons[i]:
                    print(f"Button {i:2d}  {'PRESSED' if state else 'released'}")
                    prev_buttons[i] = state

            for i in range(js.get_numaxes()):
                val = js.get_axis(i)
                if abs(val - prev_axes[i]) > 0.05:
                    print(f"Axis   {i:2d}  {val:+.3f}")
                    prev_axes[i] = val

            for i in range(js.get_numhats()):
                hat = js.get_hat(i)
                print(f"Hat    {i:2d}  {hat}") if any(hat) else None

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
