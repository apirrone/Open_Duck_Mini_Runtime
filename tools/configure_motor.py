"""
This may end up being deprecated in favor of using a better configuration tool.

Right now, since all our motors have id "1" when they pop out of the packaging,
it's annoying to have to set them up one by one.

I'm working on another version of this which will (...hopefully) work with a fully
assembled droid, and it'll just call out either via the speaker or via the console
which motor to set up next, and then you will move motor forward and back

so it'll be like "move the left hip forward and back, then press enter to continue"
and it'll check for that movement and then assign the correct ID that way, then apply
the correct configuration for that motor.

todo: maybe make this work on more than just linux dev. would be nice to
have this run on first boot for the pi and have it configure itself & store
"""

import sys
import time
import argparse

sys.stdout.reconfigure(line_buffering=True)

from pypot.feetech import FeetechSTS3215IO

DEFAULT_ID = 1  # A brand new motor should have id 1

parser = argparse.ArgumentParser()
parser.add_argument(
    "--port",
    help="The port the motor is connected to. Default is /dev/ttyACM0. Use `ls /dev/cu.* | grep usb` on macOS to find the port.",
    default="/dev/ttyACM0",
)
parser.add_argument(
    "--id", help="The id to assign to the motor.", type=int, required=True
)
args = parser.parse_args()

print(f"Opening port {args.port} ...")
io = FeetechSTS3215IO(args.port)
print(f"Port opened.")

current_id = DEFAULT_ID


def scan():
    print(f"Scanning IDs 0-254 (this may take up to 13 seconds) ...")
    for i in range(255):
        print(f"  scanning id {i:3d} / 254 ...", end="\r")
        try:
            io.get_present_position([i])
            print(f"  Found motor at id {i}        ")
            return i
        except Exception:
            pass
    print()
    return None


print(f"Looking for motor at default id ({DEFAULT_ID}) ...")
try:
    io.get_present_position([DEFAULT_ID])
    print(f"Found motor at default id ({DEFAULT_ID}).")
except Exception:
    print(f"No motor at default id ({DEFAULT_ID}).")
    current_id = scan()
    if current_id is None:
        print("Could not find any motor. Check the port and power. Exiting.")
        sys.exit(1)

print()
print(f"--- Current config (id={current_id}) ---")
print(f"  P={io.get_P_coefficient([current_id])}")
print(f"  I={io.get_I_coefficient([current_id])}")
print(f"  D={io.get_D_coefficient([current_id])}")
print(f"  acceleration={io.get_acceleration([current_id])}")
print(f"  max_acceleration={io.get_maximum_acceleration([current_id])}")
print(f"  mode={io.get_mode([current_id])}")

print()
print(f"Configuring motor: id {current_id} -> {args.id} ...")
io.set_lock({current_id: 0})
io.set_mode({current_id: 0})
io.set_maximum_acceleration({current_id: 0})
io.set_acceleration({current_id: 0})
io.set_P_coefficient({current_id: 32})
io.set_I_coefficient({current_id: 0})
io.set_D_coefficient({current_id: 0})
io.change_id({current_id: args.id})

current_id = args.id

time.sleep(1)
print(f"Centering motor (goal position -> 0) ...")
io.set_goal_position({current_id: 0})
time.sleep(1)

print()
print(f"--- Done. Motor id: {current_id} ---")
print(f"  P={io.get_P_coefficient([current_id])}")
print(f"  I={io.get_I_coefficient([current_id])}")
print(f"  D={io.get_D_coefficient([current_id])}")
print(f"  acceleration={io.get_acceleration([current_id])}")
print(f"  max_acceleration={io.get_maximum_acceleration([current_id])}")
print(f"  mode={io.get_mode([current_id])}")
