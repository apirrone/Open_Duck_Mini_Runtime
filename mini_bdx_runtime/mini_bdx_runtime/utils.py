import serial.tools.list_ports
from pypot.feetech import FeetechSTS3215IO

DEVICE_IDS = [
    (0x1A86, 0x55D3),  # Example device 1
    # Add more (vendor_id, product_id) pairs here as needed
]

FALLBACK_PORT = "/dev/ttyACM0"


def get_port():
    port = None
    auto_port = auto_detect_port()
    if auto_port is None:
        print(
            f"Device not auto-detected. Attempting connection using fallback port {FALLBACK_PORT}."
        )
        port = FALLBACK_PORT
    else:
        port = auto_port

    # Attempt to connect and catch any connection errors.
    try:
        FeetechSTS3215IO(port)
        return port
    except Exception as exc:
        message = f"Error connecting to the motor using port {port}: {exc}. Please check your connection.\n"
        # If the fallback port was used, advise the user with additional usage information.
        if port == FALLBACK_PORT:
            message += f" If your device is not connected via {FALLBACK_PORT}, please specify the correct port using the '--port' argument (e.g., --port /dev/ttyUSB0)."
        else:
            message += " If you believe your device is connected on a different port, try specifying it using the '--port <PORT>' argument."
        print(message)
    return None


def auto_detect_port():
    """
    Scans available serial ports and returns the port name for the first device
    matching any of the given (vendor_id, product_id) pairs.
    """
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        for vendor_id, product_id in DEVICE_IDS:
            if port.vid == vendor_id and port.pid == product_id:
                print(
                    f"Found device on port: {port.device} (VID: {hex(vendor_id)}, PID: {hex(product_id)})"
                )
                return port.device
    return None
