# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Runtime for the Open Duck Mini — a small RL-controlled bipedal robot running on a Raspberry Pi Zero 2W (or Pi 5). Controls 14 Feetech STS3215 servos via USB serial using an ONNX neural network policy at 50 Hz.

## Commands

```bash
# Install (uses uv, not pip directly)
uv sync

# On Raspberry Pi 5 only (different GPIO library)
uv pip uninstall RPi.GPIO && uv pip install lgpio

# Run
uv run walk                                        # gamepad control
uv run walk --onnx_model_path /path/to/model.onnx  # custom model
uv run walk-keyboard                               # SSH keyboard control
uv run walk-watchdog                               # auto-restarts walk on crash

# Tests (hardware tests excluded by default — see pyproject.toml addopts)
uv run pytest                                      # unit tests only
uv run pytest tests/test_motor_controller.py       # requires connected robot

# Calibration tools
python3 tools/find_soft_offsets.py   # find joint offsets interactively
python3 tools/controller_info.py     # identify gamepad axis/button indices
python3 tools/check_voltage.py       # verify servo bus voltage
```

## Architecture

Three-layer design running at 50 Hz:

**Hardware layer** (`hardware/`): `HWI` manages 14 DOFs over `/dev/ttyACM0` at 1 Mbaud using the `rustypot` library (Feetech STS3215 protocol). `Imu` wraps a BNO055 I2C sensor sampled in a background thread at 50 Hz and exposes cached gyro/accel/gravity values. Expression peripherals (eyes, antennas, projector, speaker, camera) are all optional and gated by `expression_features` in config.

**Controller layer** (`controller/`): `XBoxController` polls Xbox/DualSense/generic USB gamepad or keyboard at 20 Hz in a background thread. Supports custom axis/button remapping for generic USB controllers via config.

**RL walk layer** (`rl_walk/`): `RLWalk` in `walk.py` is the main orchestrator. `OnnxInfer` wraps the ONNX policy (initialized with `awd=True`). `PolyReferenceMotion` provides a gait phase signal. `LowPassActionFilter` in `rl_utils.py` optionally smooths actions.

### Control loop (50 Hz, `walk.py RLWalk.run()`)

1. Read cached IMU data (gyro, accel, gravity)
2. Read motor positions/velocities via HWI
3. Build 54-element observation: `[gyro(3), accel(3), commands(7), joint_pos(14), joint_vel(14), action_history(14×3), targets(14), foot_contacts(2), gait_phase(2)]`
4. Run ONNX inference → 14-element action
5. Motor target = `init_pos + action * action_scale` (default 0.25)
6. Send targets; check fall detection (3 consecutive frames > `fall_threshold_deg`)

### Joint order

Defined once in `hwi.py` and must stay consistent everywhere:
```
[left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
 neck_pitch, head_pitch, head_yaw, head_roll,
 right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle]
```
`rl_utils.py` contains coordinate transforms for Mujoco ↔ Isaac Gym compatibility.

## Configuration

Loaded from `~/duck_config.json` (not in the repo). If absent, `DuckConfig` uses defaults and prompts for confirmation. See `example_config.json` for all fields. Key non-obvious fields:

- `imu_upside_down`: flips IMU orientation if mounted inverted
- `phase_frequency_factor_offset`: added to base 1.0 factor (1.3 = sprint)
- `generic_usb_controller`: override axis/button indices when `controller_type = "generic_usb"`
- `joints_offsets`: per-joint servo correction in radians (output of `find_soft_offsets.py`)

## Notes

- `uv run walk` requires a valid ONNX model — there is no bundled default. Pass `--onnx_model_path` or set it in config.
- Hardware tests in `tests/test_motor_controller.py` are excluded from the default pytest run via `addopts` in `pyproject.toml`, not by marker alone.
- Assets (ONNX models, reference motions) are resolved relative to the installed package at `src/assets/`.
- The `TRACE` log level (below DEBUG) is defined in `log.py` for per-frame diagnostics.
