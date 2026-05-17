import argparse
import os
import subprocess
import sys
import time

import pygame


HOME_DIR = os.path.expanduser("~")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def wait_for_controller(poll_interval: float = 1.0):
    pygame.init()
    pygame.joystick.init()

    print("Waiting for bluetooth controller to connect...")
    while True:
        pygame.event.pump()
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print(f"Controller connected: {joystick.get_name()}")
            return

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("--serial_port", type=str, default="/dev/ttyACM0")
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="save the run's observations",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)
    args = parser.parse_args()

    os.chdir(SCRIPT_DIR)

    turn_on_script = os.path.join(SCRIPT_DIR, "turn_on.py")
    walk_script = os.path.join(SCRIPT_DIR, "v2_rl_walk_mujoco.py")

    # Call the original turn_on script
    subprocess.run(
        [
            sys.executable,
            turn_on_script,
        ],
        check=True,
        cwd=SCRIPT_DIR,
    )

    wait_for_controller()

    # Build the walk command
    walk_cmd = [
        sys.executable,
        walk_script,
        "--onnx_model_path",
        args.onnx_model_path,
        "--duck_config_path",
        args.duck_config_path,
        "-a",
        str(args.action_scale),
        "-p",
        str(args.p),
        "-i",
        str(args.i),
        "-d",
        str(args.d),
        "-c",
        str(args.control_freq),
        "--pitch_bias",
        str(args.pitch_bias),
    ]

    if args.save_obs is not False:
        walk_cmd.extend(["--save_obs", args.save_obs])

    if args.replay_obs is not None:
        walk_cmd.extend(["--replay_obs", args.replay_obs])

    if args.cutoff_frequency is not None:
        walk_cmd.extend(["--cutoff_frequency", str(args.cutoff_frequency)])

    # Hand off execution to the walking script
    os.execv(sys.executable, walk_cmd)


if __name__ == "__main__":
    main()
