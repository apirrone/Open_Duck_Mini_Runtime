"""
Boot-time hardware validation for Open Duck Mini Runtime.

Runs on every boot via a systemd service. Checks:
  1. Unit tests (pytest, no hardware required)
  2. Motor USB serial port exists
  3. I2C bus exists (for BNO055 IMU)
  4. ONNX model file present
  5. duck_config.json present and parseable

Exit code 0 = all checks passed; 1 = one or more failures.
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path.home()

PASS = "[ PASS ]"
FAIL = "[ FAIL ]"
WARN = "[ WARN ]"


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    msg = f"{tag} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main() -> int:
    print("=" * 56)
    print("  Open Duck Mini — boot hardware validation")
    print("=" * 56)

    results = []

    # ── 1. Unit tests ─────────────────────────────────────────
    try:
        # Find project root (two levels up from this file's installed location)
        project_root = Path(__file__).parent.parent.parent
        tests_dir = project_root / "tests"
        if not tests_dir.exists():
            # Installed package — run against installed site-packages tests
            results.append(check("Unit tests", False, "tests/ directory not found"))
        else:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(tests_dir),
                 "-m", "not hardware", "-q", "--tb=short", "--no-header"],
                capture_output=True, text=True, timeout=120,
            )
            passed = result.returncode == 0
            # Extract summary line (last non-empty line)
            summary = next(
                (l for l in reversed(result.stdout.splitlines()) if l.strip()), ""
            )
            results.append(check("Unit tests", passed, summary))
    except subprocess.TimeoutExpired:
        results.append(check("Unit tests", False, "timed out after 120 s"))
    except Exception as e:
        results.append(check("Unit tests", False, str(e)))

    # ── 2. Motor USB serial port ───────────────────────────────
    serial_port = os.environ.get("DUCK_SERIAL_PORT", "/dev/ttyACM0")
    results.append(check(
        f"Motor serial  ({serial_port})",
        Path(serial_port).exists(),
        "not found — is the motor controller plugged in?" if not Path(serial_port).exists() else "",
    ))

    # ── 3. I2C bus ─────────────────────────────────────────────
    i2c_buses = glob.glob("/dev/i2c-*")
    results.append(check(
        "I2C bus",
        bool(i2c_buses),
        f"found: {', '.join(sorted(i2c_buses))}" if i2c_buses else "no /dev/i2c-* devices — I2C not enabled?",
    ))

    # ── 4. ONNX model ──────────────────────────────────────────
    onnx_path = Path(os.environ.get("DUCK_ONNX_PATH", HOME / "BEST_WALK_ONNX_2.onnx"))
    results.append(check(
        f"ONNX model    ({onnx_path.name})",
        onnx_path.exists(),
        "upload the model to ~/BEST_WALK_ONNX_2.onnx" if not onnx_path.exists() else "",
    ))

    # ── 5. duck_config.json ────────────────────────────────────
    config_path = HOME / "duck_config.json"
    if not config_path.exists():
        results.append(check(
            "duck_config.json",
            False,
            "not found — copy example_config.json from the repo to ~/duck_config.json",
        ))
    else:
        try:
            json.loads(config_path.read_text())
            results.append(check("duck_config.json", True))
        except json.JSONDecodeError as e:
            results.append(check("duck_config.json", False, f"invalid JSON: {e}"))

    # ── Summary ────────────────────────────────────────────────
    print("=" * 56)
    passed = sum(results)
    total = len(results)
    print(f"  {passed}/{total} checks passed")
    print("=" * 56)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
