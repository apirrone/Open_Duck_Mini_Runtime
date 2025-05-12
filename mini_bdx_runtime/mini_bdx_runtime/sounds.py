import pygame
import time
import os
import random
import logging


class Sounds:
    def __init__(self, volume=1.0, sound_directory="./", mixer=None):
        """
        Initialize the Sounds system.
        Args:
            volume (float): Volume level between 0.0 and 1.0
            sound_directory (str): Directory containing .wav files
            mixer: Optional pygame.mixer module for dependency injection (testing)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initializing Sounds class")
        self.mixer = mixer if mixer is not None else pygame.mixer
        # Clamp volume between 0 and 1
        clamped_volume = max(0.0, min(1.0, volume))
        try:
            self.mixer.init()
            self.mixer.music.set_volume(clamped_volume)
        except Exception as e:
            self.logger.error(f"Failed to initialize mixer: {e}")
            self.ok = False
            return
        self.sounds = {}
        self.ok = True
        try:
            for file in os.listdir(sound_directory):
                if file.endswith(".wav"):
                    sound_path = os.path.join(sound_directory, file)
                    try:
                        sound = self.mixer.Sound(sound_path)
                        sound.set_volume(clamped_volume)
                        self.sounds[file] = sound
                    except Exception as e:
                        self.logger.error(f"Failed to load {file}: {e}")
        except FileNotFoundError:
            self.logger.error(f"Directory {sound_directory} not found.")
            self.ok = False
        if len(self.sounds) == 0:
            self.logger.warning("No sound files found in the directory.")
            self.ok = False

    def is_playing(self):
        """Check if any sound is currently playing"""
        return self.mixer.get_busy()

    def play(self, sound_name, wait_if_playing=True):
        """
        Play a sound file
        Args:
            sound_name: Name of the sound file to play
            wait_if_playing: If True, wait for current sound to finish before playing new sound
        """
        if not self.ok:
            self.logger.error("Sounds not initialized properly.")
            return False

        if sound_name not in self.sounds:
            self.logger.error(f"Sound '{sound_name}' not found!")
            return False

        if wait_if_playing:
            # Wait for current sound to finish
            self.logger.debug(f"Waiting for current sound to finish before playing {sound_name}")
            while self.is_playing():
                time.sleep(0.1)

        self.sounds[sound_name].play()
        self.logger.info(f"Playing: {sound_name}")
        return True

    def play_random_sound(self, wait_if_playing=True):
        """Play a random sound from loaded sounds"""
        if not self.ok:
            self.logger.error("Sounds not initialized properly.")
            return False
        sound_name = random.choice(list(self.sounds.keys()))
        return self.play(sound_name, wait_if_playing)

    def play_happy(self, wait_if_playing=True):
        """Play happy sound"""
        return self.play("happy1.wav", wait_if_playing)

    def wait_for_sound(self):
        """Wait for current sound to finish playing"""
        self.logger.debug("Waiting for sound to finish playing")
        while self.is_playing():
            time.sleep(0.1)


# Example usage
if __name__ == "__main__":
    sound_player = Sounds(1.0, "../assets/")
    time.sleep(1)
    while True:
        # sound_player.play_random_sound()
        sound_player.play_happy()
        sound_player.wait_for_sound()  # Wait for sound to finish
        time.sleep(3)
