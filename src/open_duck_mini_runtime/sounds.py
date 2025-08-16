import pygame
import time
import os
import random


class Sounds:
    def __init__(self, volume=1.0, sound_directory="./"):
        pygame.mixer.init()
        pygame.mixer.music.set_volume(volume)
        self.sounds = {}
        self.ok = True
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

    def play(self, sound_name):
        if not self.ok:
            print("Sounds not initialized properly.")
            return
        if sound_name in self.sounds:
            self.sounds[sound_name].play()
            print(f"Playing: {sound_name}")
        else:
            print(f"Sound '{sound_name}' not found!")

    def play_random_sound(self):
        if not self.ok:
            print("Sounds not initialized properly.")
            return
        sound_name = random.choice(list(self.sounds.keys()))
        self.sounds[sound_name].play()

    def play_happy(self):
        self.sounds["happy1.wav"].play()



# Example usage
if __name__ == "__main__":
    sound_player = Sounds(1.0, "../assets/")
    time.sleep(1)
    while True:
        # sound_player.play_random_sound()
        sound_player.play_happy()
        time.sleep(3)
