import time
import board
import neopixel
import threading
import curses
import math

# --- NeoPixel setup ---
pixel_pin = board.D10
num_pixels = 10  # Ok so for some unknown reason this works with 10 but NOT with 3? TODO figure this out I guess
ORDER = neopixel.RGBW  # <-- flip to GRBW if colors look wrong

pixels = neopixel.NeoPixel(
    pixel_pin, num_pixels, brightness=0.3, auto_write=False, pixel_order=ORDER
)

# --- Colors (R, G, B, W) ---
colors = {
    "1": (255, 0, 0, 0),  # Red
    "2": (0, 255, 0, 0),  # Green
    "3": (0, 0, 255, 0),  # Blue
    "4": (0, 0, 0, 255),  # White
    "5": (255, 255, 0, 0),  # Yellow
    "6": (0, 255, 255, 0),  # Cyan
    "7": (255, 0, 255, 0),  # Magenta
    "8": (128, 128, 128, 0),  # Gray
    "9": (0, 0, 0, 0),  # Off
}

# --- Globals ---
rainbow_active = False
brightness = 0.3


# --- Gradient rainbow (wave across strip) ---
def smooth_rainbow(wait=0.02):
    global rainbow_active
    hue = 0
    while rainbow_active:
        for i in range(num_pixels):
            # Offset hue per pixel for gradient effect
            offset = (hue + (i * 360 / num_pixels)) % 360
            r, g, b = hsv_to_rgb(offset / 360.0, 1, 1)
            pixels[i] = (int(r * 255), int(g * 255), int(b * 255), 0)
        pixels.show()
        hue = (hue + 1) % 360
        time.sleep(wait)


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


# --- Main with curses ---
def main(stdscr):
    global rainbow_active, brightness
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(True)
    stdscr.addstr(
        0, 0, "Press 1–9 for colors, 0 for rainbow, ←/→ for brightness, q to quit."
    )

    while True:
        key = stdscr.getch()
        if key == -1:
            time.sleep(0.05)
            continue

        if key in range(49, 58):  # Keys '1'–'9'
            ch = chr(key)
            rainbow_active = False
            pixels.fill(colors[ch])
            pixels.show()
            stdscr.addstr(2, 0, f"Set color {ch}: {colors[ch]}       ")

        elif key == ord("0"):  # Rainbow toggle
            if not rainbow_active:
                rainbow_active = True
                threading.Thread(target=smooth_rainbow, daemon=True).start()
                stdscr.addstr(2, 0, "🌈 Gradient rainbow started       ")
            else:
                rainbow_active = False
                stdscr.addstr(2, 0, "Rainbow stopped                ")

        elif key == curses.KEY_RIGHT:  # Brightness up
            brightness = min(1.0, brightness + 0.1)
            pixels.brightness = brightness
            pixels.show()
            stdscr.addstr(3, 0, f"Brightness: {brightness:.1f}       ")

        elif key == curses.KEY_LEFT:  # Brightness down
            brightness = max(0.0, brightness - 0.1)
            pixels.brightness = brightness
            pixels.show()
            stdscr.addstr(3, 0, f"Brightness: {brightness:.1f}       ")

        elif key in (ord("q"), ord("Q")):
            rainbow_active = False
            pixels.fill((0, 0, 0, 0))
            pixels.show()
            stdscr.addstr(2, 0, "Goodbye!                       ")
            time.sleep(0.5)
            break


curses.wrapper(main)
