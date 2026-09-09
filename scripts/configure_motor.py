from pypot.feetech import FeetechSTS3215IO
import argparse
import time
from mini_bdx_runtime.utils import get_port
import tqdm

DEFAULT_ID = 1  # A brand new motor should have id 1

# Parse arguments. This allows an override if you supply --port.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--port",
    help=(
        "The port the motor is connected to. If not specified, "
        "the script will try to auto-detect the port using provided USB IDs or "
        "fall back to /dev/ttyACM0."
    ),
    default=None,
)
parser.add_argument("--id", help="The id to set to the motor.", type=str, required=True)
args = parser.parse_args()

port = args.port
# If no port is provided, try to auto-detect using the USB id info.
if port is None:
    port = get_port()
if port is None:
    exit()


io = FeetechSTS3215IO(port)

current_id = DEFAULT_ID


def scan():
    id = None
    for i in tqdm.tqdm(range(255), desc="Scanning ...", unit="id"):
        try:
            io.get_present_position([i])
            id = i
            print(f"Found motor with id {id}")
            break
        except Exception:
            pass
    return id


try:
    io.get_present_position([DEFAULT_ID])
except Exception:
    print(
        f"Could not find motor with default id ({DEFAULT_ID}). Scanning for motor ..."
    )
    res = scan()
    print("ID :", res)
    if res is not None:
        current_id = res
    else:
        print("Could not find motor. Exiting ...")
        exit()


kp = io.get_P_coefficient([current_id])
ki = io.get_I_coefficient([current_id])
kd = io.get_D_coefficient([current_id])
max_acceleration = io.get_maximum_acceleration([current_id])
acceleration = io.get_acceleration([current_id])
mode = io.get_mode([current_id])

io.set_lock({current_id: 0})
io.set_mode({current_id: 0})
io.set_maximum_acceleration({current_id: 0})
io.set_acceleration({current_id: 0})
io.set_P_coefficient({current_id: 32})
io.set_I_coefficient({current_id: 0})
io.set_D_coefficient({current_id: 0})
io.change_id({current_id: int(args.id)})

current_id = int(args.id)

time.sleep(1)
print("==")
res = input("WARNING, the motor will move to its zero position. Continue ? ([Y]/n)")
if res.lower() != "y" and res.lower() != "":
    print("Exiting ...")
    exit()

print("==")
io.set_goal_position({current_id: 0})

time.sleep(1)

print("===")
print("Done configuring motor.")
print(f"Motor id: {current_id}")
print(f"P coefficient : {io.get_P_coefficient([current_id])}")
print(f"I coefficient : {io.get_I_coefficient([current_id])}")
print(f"D coefficient : {io.get_D_coefficient([current_id])}")
print(f"acceleration: {io.get_acceleration([current_id])}")
print(f"max_acceleration: {io.get_maximum_acceleration([current_id])}")
print(f"mode: {io.get_mode([current_id])}")
print("===")
