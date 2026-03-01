#!/usr/bin/env python3
"""
walk_keyboard.py – WASD keyboard-controlled walking over SSH
=============================================================
Run this on the robot via SSH to walk without a gamepad.

Usage
-----
    uv run src/open_duck_mini_runtime/walk_keyboard.py
    uv run src/open_duck_mini_runtime/walk_keyboard.py --onnx_model_path ~/BEST_WALK_ONNX_2.onnx

Key bindings
------------
  W / S         Forward / Backward
  A / D         Turn left / Turn right
  Q / E         Strafe left / Strafe right
  L (hold)      Speed boost  (maps to LB button)
  Space         Toggle pause / unpause
  X             Toggle projector
  B             Play random sound
  P             Toggle head-control mode
  Ctrl-C / ESC  Exit

No DISPLAY or gamepad required – reads raw stdin over the SSH connection.
"""

import os
import sys
import argparse

# Ensure SDL does not try to open a display or audio device.
# Must be set before any pygame import occurs inside modules.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

HOME_DIR = os.path.expanduser("~")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk the Open Duck Mini using keyboard (WASD) over SSH."
    )
    parser.add_argument(
        "--onnx_model_path",
        type=str,
        default=f"{HOME_DIR}/BEST_WALK_ONNX_2.onnx",
    )
    parser.add_argument(
        "--duck_config_path",
        type=str,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", "--kp", type=int, default=30)
    parser.add_argument("-i", "--ki", type=int, default=0)
    parser.add_argument("-d", "--kd", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--save_obs",
        type=str,
        default=None,
        help="Save the run's observations to the given path.",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        default=None,
        help="Replay observations from a previous run.",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)

    args = parser.parse_args()

    # Import here so that SDL environment variables are already set
    from open_duck_mini_runtime.keyboard_controller import KeyboardController
    from open_duck_mini_runtime.walk import RLWalk

    KeyboardController.print_controls()

    print("Initialising RLWalk (hardware connect)…")
    rl_walk = RLWalk(
        onnx_model_path=args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=[args.kp, args.ki, args.kd],
        control_freq=args.control_freq,
        commands=True,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
        controller_type_override="keyboard",
    )
    print("Hardware connected – walking with keyboard control.\n")

    try:
        rl_walk.run()
    finally:
        # Ensure the keyboard controller restores terminal state on exit
        if hasattr(rl_walk, "controller") and hasattr(rl_walk.controller, "stop"):
            rl_walk.controller.stop()
        print("\nKeyboard walk finished.")


if __name__ == "__main__":
    main()
