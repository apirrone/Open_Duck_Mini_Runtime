import pygame
import time
import os
import random
from threading import Thread, Lock

from open_duck_mini_runtime.led_controller import get_controller


class Sounds:
    def __init__(self, volume=1.0, sound_directory="", default_color: str = "white", sound_color_map: dict | None = None):
        pygame.mixer.init()
        pygame.mixer.music.set_volume(volume)
        self.sounds = {}
        self.ok = True
        self._ctrl = get_controller()
        self._default_color = default_color
        self._sound_color_map = sound_color_map or {}
        self._color_token = 0
        self._lock = Lock()
        try:
            for file in os.listdir(sound_directory):
                if file.endswith(".wav"):
                    sound_path = os.path.join(sound_directory, file)
                    try:
                        self.sounds[file] = pygame.mixer.Sound(sound_path)
                        print(f"Loaded: {file}")
                    except pygame.error as e:
                        print(f"Failed to load {file}: {e}")
        except FileNotFoundError:
            print(f"Directory {sound_directory} not found.")
            self.ok = False
        if len(self.sounds) == 0:
            print("No sound files found in the directory.")
            self.ok = False

        # Initialize color map defaults (cycle through a limited palette)
        if self.ok and not self._sound_color_map:
            palette = ["red", "green", "blue", "white"]
            for idx, name in enumerate(sorted(self.sounds.keys())):
                self._sound_color_map[name] = palette[idx % len(palette)]

        # Ensure LEDs are default color initially (without changing on/off state)
        try:
            self._ctrl.set_all_color(self._default_color)
        except Exception:
            pass

    def play(self, sound_name):
        if not self.ok:
            print("Sounds not initialized properly.")
            return
        if sound_name in self.sounds:
            chan = self.sounds[sound_name].play()
            print(f"Playing: {sound_name}")

            # Change LEDs to mapped color (supports string for all or dict per pixel)
            mapping_val = self._sound_color_map.get(sound_name, self._default_color)
            try:
                if isinstance(mapping_val, dict):
                    left = mapping_val.get("left", self._default_color)
                    right = mapping_val.get("right", self._default_color)
                    proj = mapping_val.get("projector", self._default_color)
                    self._ctrl.set_left_eye_color(left)
                    self._ctrl.set_right_eye_color(right)
                    self._ctrl.set_projector_color(proj)
                else:
                    self._ctrl.set_all_color(mapping_val)
            except Exception as e:
                print(f"LED color set failed: {e}")

            # Schedule reset to default when the sound finishes
            with self._lock:
                self._color_token += 1
                token = self._color_token

            def _reset_when_done(channel, expected_token):
                try:
                    # If no channel returned, fallback to sleep by length
                    if channel is None:
                        duration = self.sounds[sound_name].get_length()
                        time.sleep(duration)
                    else:
                        # Wait until the channel is no longer busy
                        while channel.get_busy():
                            time.sleep(0.02)
                    # Only reset if no newer sound changed the color
                    with self._lock:
                        if expected_token == self._color_token:
                            self._ctrl.set_all_color(self._default_color)
                except Exception:
                    pass

            Thread(target=_reset_when_done, args=(chan, token), daemon=True).start()
        else:
            print(f"Sound '{sound_name}' not found!")

    def play_random_sound(self):
        if not self.ok:
            print("Sounds not initialized properly.")
            return
        sound_name = random.choice(list(self.sounds.keys()))
        self.play(sound_name)

    def play_happy(self):
        self.play("happy1.wav")

    # API helpers
    def set_default_color(self, color: str):
        self._default_color = color
        try:
            self._ctrl.set_all_color(self._default_color)
        except Exception:
            pass

    def set_sound_color(self, sound_name: str, color: str):
        self._sound_color_map[sound_name] = color

    def set_sound_color_map(self, mapping: dict):
        self._sound_color_map.update(mapping)



# Example usage
if __name__ == "__main__":
    sound_player = Sounds(1.0, "../assets/")
    time.sleep(1)
    while True:
        sound_player.play_random_sound()
        time.sleep(5)
