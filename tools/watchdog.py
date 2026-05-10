"""
Watchdog wrapper for `uv run walk`.

Launches the walk process as a child and ensures motors are disabled when it
exits for *any* reason — including SIGKILL, OOM kills, or crashes — because
SIGKILL cannot be caught inside the walk process itself.

Usage:
    python3 tools/watchdog.py [-- <extra walk args>]
    uv run python3 tools/watchdog.py -- --log-level DEBUG

The '--' separator is optional; all args after it are forwarded to `walk`.
"""

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_turn_off():
    print("[watchdog] Running turn_off...", flush=True)
    try:
        subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "turn_off.py")],
            timeout=10,
        )
        print("[watchdog] Motors disabled.", flush=True)
    except Exception as e:
        print(f"[watchdog] turn_off failed: {e}", flush=True)


def main():
    # Split off any args meant for walk (everything after optional '--')
    argv = sys.argv[1:]
    if "--" in argv:
        walk_extra = argv[argv.index("--") + 1 :]
    else:
        walk_extra = argv

    cmd = ["uv", "run", "walk"] + walk_extra
    print(f"[watchdog] Starting: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))

    try:
        proc.wait()
    except KeyboardInterrupt:
        # Ctrl-C propagates to the child too; wait for it to finish its own cleanup
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    exit_code = proc.returncode
    print(f"[watchdog] Walk exited with code {exit_code}.", flush=True)

    # Always try to disable motors, regardless of why the process died
    run_turn_off()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
