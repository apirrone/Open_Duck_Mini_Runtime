import argparse
import random
import time
from pathlib import Path

import pygame


def find_assets_dir() -> Path:
    # assets/ is a sibling of the package directory
    pkg_dir = Path(__file__).resolve().parent
    src_dir = pkg_dir.parent
    assets = src_dir / "assets"
    if not assets.exists():
        raise FileNotFoundError(f"Assets directory not found: {assets}")
    return assets


def list_wavs(assets_dir: Path) -> list[Path]:
    return sorted(assets_dir.glob("*.wav"))


def play_sound(path: Path, volume: float = 1.0) -> None:
    snd = pygame.mixer.Sound(str(path))
    snd.set_volume(max(0.0, min(1.0, volume)))
    ch = snd.play()
    while ch.get_busy():
        time.sleep(0.05)


class Sounds:
    """High-level sound manager used by :class:`~open_duck_mini_runtime.walk.RLWalk`."""

    def __init__(self, volume: float = 1.0, sound_directory: str = None):
        self.volume = volume
        if sound_directory is not None:
            assets = Path(sound_directory)
        else:
            assets = find_assets_dir()
        self.wav_files: list[Path] = list_wavs(assets)
        if not pygame.mixer.get_init():
            pygame.mixer.init()

    def play_random_sound(self) -> None:
        if not self.wav_files:
            return
        path = random.choice(self.wav_files)
        play_sound(path, volume=self.volume)

    def play_sound(self, index: int) -> None:
        if 0 <= index < len(self.wav_files):
            play_sound(self.wav_files[index], volume=self.volume)


def main():
    parser = argparse.ArgumentParser(description="Step through sounds in assets/")
    parser.add_argument("--auto", action="store_true", help="Automatically advance without waiting for Enter")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between sounds when --auto is set")
    parser.add_argument("--volume", type=float, default=1.0, help="Playback volume (0.0 - 1.0)")
    args = parser.parse_args()

    assets = find_assets_dir()
    files = list_wavs(assets)
    if not files:
        print(f"No .wav files found in {assets}")
        return

    pygame.mixer.init()
    print(f"pygame {pygame.version.ver}")
    print(f"Found {len(files)} wav files in {assets}")

    try:
        for i, wav in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Playing: {wav.name}")
            play_sound(wav, volume=args.volume)
            if args.auto:
                time.sleep(max(0.0, args.delay))
            else:
                input("Press Enter for next...")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        pygame.quit()


if __name__ == "__main__":
    main()
