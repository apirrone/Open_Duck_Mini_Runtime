import argparse
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