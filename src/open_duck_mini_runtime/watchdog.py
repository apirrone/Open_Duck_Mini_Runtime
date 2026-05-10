"""
Watchdog wrapper for `uv run walk`.

Launches the walk process as a child and disables motors when it exits for
*any* reason — including SIGKILL, OOM kills, and crashes — because SIGKILL
cannot be caught inside the walk process itself.

Usage:
    uv run walk-watchdog                      # default config
    uv run walk-watchdog -- --log-level DEBUG # pass args to walk
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root


def _turn_off():
    print("[watchdog] Disabling motors...", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, str(_REPO_ROOT / "tools" / "turn_off.py")],
            timeout=10,
        )
        if result.returncode == 0:
            print("[watchdog] Motors disabled.", flush=True)
        else:
            print(f"[watchdog] turn_off exited {result.returncode}", flush=True)
    except Exception as e:
        print(f"[watchdog] turn_off failed: {e}", flush=True)


def main():
    argv = sys.argv[1:]
    walk_extra = argv[argv.index("--") + 1 :] if "--" in argv else argv

    cmd = ["uv", "run", "walk"] + walk_extra
    print(f"[watchdog] Starting: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT))
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    exit_code = proc.returncode
    print(f"[watchdog] Walk exited (code {exit_code}).", flush=True)
    _turn_off()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
