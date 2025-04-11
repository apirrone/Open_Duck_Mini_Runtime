from pypot.feetech import FeetechSTS3215IO
import argparse
import time
import serial.tools.list_ports

DEFAULT_ID = 1  # A brand new motor should have id 1

# These are placeholders. Please replace these with your actual device's USB IDs.
TARGET_VENDOR_ID = 0x1A86  # e.g., your device's vendor id (in hex)
TARGET_PRODUCT_ID = 0x55D3  # e.g., your device's product id (in hex)

# Update the find_port function to accept a list of (vendor_id, product_id) tuples.
def find_port(device_ids):
    """
    Scans available serial ports and returns the port name for the first device
    matching any of the given (vendor_id, product_id) pairs.
    """
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        for vendor_id, product_id in device_ids:
            if port.vid == vendor_id and port.pid == product_id:
                print(f"Found device on port: {port.device} (VID: {hex(vendor_id)}, PID: {hex(product_id)})")
                return port.device
    return None

# Update the script to use a list of device IDs.
DEVICE_IDS = [
    (0x1A86, 0x55D3),  # Example device 1
    # Add more (vendor_id, product_id) pairs here as needed
]

# Parse arguments. This allows an override if you supply --port.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--port",
    help=("The port the motor is connected to. If not specified, "
          "the script will try to auto-detect the port using provided USB IDs or "
          "fall back to /dev/ttyACM0."),
    default=None,
)
parser.add_argument("--id", help="The id to set to the motor.", type=str, required=True)
args = parser.parse_args()

# If no port is provided, try to auto-detect using the USB id info.
if args.port is None:
    auto_port = find_port(DEVICE_IDS)
    if (auto_port is None):
        fallback_port = "/dev/ttyACM0"
        print(f"Device not auto-detected. Attempting connection using fallback port {fallback_port}.")
        args.port = fallback_port
    else:
        args.port = auto_port

# Attempt to connect and catch any connection errors.
try:
    io = FeetechSTS3215IO(args.port)
except Exception as exc:
    message = f"Error connecting to the motor using port {args.port}: {exc}. Please check your connection.\n"
    # If the fallback port was used, advise the user with additional usage information.
    fallback_port = "/dev/ttyACM0"
    if args.port == fallback_port:
        message += f" If your device is not connected via {fallback_port}, please specify the correct port using the '--port' argument (e.g., --port /dev/ttyUSB0)."
    else:
        message += " If you believe your device is connected on a different port, try specifying it using the '--port <PORT>' argument."
    print(message)
    exit(1)

current_id = DEFAULT_ID


def scan():
    id = None
    for i in range(255):

        print(f"scanning for id {i} ...")
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
    if res is not None:
        current_id = res
    else:
        print("Could not find motor. Exiting ...")
        exit()


# print("current id: ", current_id)

kp = io.get_P_coefficient([current_id])
ki = io.get_I_coefficient([current_id])
kd = io.get_D_coefficient([current_id])
max_acceleration = io.get_maximum_acceleration([current_id])
acceleration = io.get_acceleration([current_id])
mode = io.get_mode([current_id])

# print(f"PID : {kp}, {ki}, {kd}")
# print(f"max_acceleration: {max_acceleration}")
# print(f"acceleration: {acceleration}")
# print(f"mode: {mode}")

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
