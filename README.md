# Open Duck Mini Runtime

Runtime software for the [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) – a small, open-source robotic duck that walks using a reinforcement-learning policy.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Raspberry Pi Setup](#raspberry-pi-setup)
  - [Install Raspberry Pi OS](#install-raspberry-pi-os)
  - [Setup SSH](#setup-ssh)
  - [System Updates and Dependencies](#system-updates-and-dependencies)
  - [Enable I2C](#enable-i2c)
  - [Set the USB Serial Latency Timer](#set-the-usb-serial-latency-timer)
  - [Motor Control Board udev Rules](#motor-control-board-udev-rules)
- [Install the Runtime](#install-the-runtime)
- [Configuration](#configuration)
  - [duck_config.json Reference](#duck_configjson-reference)
  - [Controller Types](#controller-types)
  - [Generic USB Controller Mapping](#generic-usb-controller-mapping)
- [Hardware Configuration](#hardware-configuration)
  - [Speaker Wiring](#speaker-wiring)
- [Testing and Calibration](#testing-and-calibration)
  - [Test the IMU](#test-the-imu)
  - [Find Joint Offsets](#find-joint-offsets)
- [Running the Duck](#running-the-duck)
  - [Gamepad Walking](#gamepad-walking)
  - [SSH Keyboard Walking](#ssh-keyboard-walking)
- [Controls Reference](#controls-reference)
  - [Xbox / DualSense Controls](#xbox--dualsense-controls)
  - [Keyboard Controls (SSH)](#keyboard-controls-ssh)
- [Running Tests](#running-tests)

---

## Quick Start

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and enter the repo
git clone https://github.com/apirrone/Open_Duck_Mini_Runtime
cd Open_Duck_Mini_Runtime

# 3. Install all dependencies into a managed virtual environment
uv sync

# 4. Walk!
uv run walk
```

---

## Raspberry Pi Setup

These instructions target a **Raspberry Pi Zero 2W** running Raspberry Pi OS Lite (64-bit).

### Install Raspberry Pi OS

1. Download [Raspberry Pi OS Lite (64-bit)](https://www.raspberrypi.com/software/operating-systems/).
2. Flash it with the [Raspberry Pi Imager](https://www.raspberrypi.com/documentation/computers/getting-started.html).
3. In the Imager's advanced options, pre-configure your username, Wi-Fi, and SSH key.

> **Tip:** Configure Wi-Fi to connect to your phone's hotspot for easy field access.

### Setup SSH

If SSH was not enabled during imaging, connect a screen and keyboard, then:

1. Connect to a Wi-Fi network.
2. Enable SSH: [Raspberry Pi SSH guide](https://www.raspberrypi.com/documentation/computers/configuration.html#setting-up-wifi).

### System Updates and Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Optional: camera support
sudo apt install -y python3-picamzero
```

### Enable I2C

```bash
sudo raspi-config
# Interface Options → I2C → Enable
```

### Set the USB Serial Latency Timer

```bash
sudo nano /etc/udev/rules.d/99-usb-serial.rules
```

Add:
```
SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
```

### Motor Control Board udev Rules

*(TODO)*

---

## Install the Runtime

```bash
git clone https://github.com/apirrone/Open_Duck_Mini_Runtime
cd Open_Duck_Mini_Runtime
uv sync
```

**Raspberry Pi 5 only** — replace the GPIO library after sync:
```bash
uv pip uninstall RPi.GPIO
uv pip install lgpio
```

---

## Configuration

### duck_config.json Reference

Copy the example config to your home directory:
```bash
cp example_config.json ~/duck_config.json
```

| Field | Type | Default | Description |
|---|---|---|---|
| `start_paused` | bool | `false` | Start the walk loop in a paused state |
| `imu_upside_down` | bool | `false` | Flip IMU orientation (robot upside-down mount) |
| `phase_frequency_factor_offset` | float | `0.0` | Offset added to gait phase frequency |
| `controller_type` | string | `"xbox"` | Which controller to use. See [Controller Types](#controller-types) |
| `expression_features.enable_eyes` | bool | `false` | Enable NeoPixel eye LEDs |
| `expression_features.enable_projector` | bool | `false` | Enable NeoPixel projector LED |
| `expression_features.enable_sounds` | bool | `false` | Enable audio playback |
| `expression_features.enable_antennas` | bool | `false` | Enable servo-driven antennas |
| `joints_offset` | object | `{}` | Per-joint offset corrections (radians) |

### Controller Types

Set `"controller_type"` in `~/duck_config.json` to one of:

| Value | Description |
|---|---|
| `"xbox"` | Xbox One / Xbox Series controller via Bluetooth |
| `"dualsense"` | PlayStation 5 DualSense controller via Bluetooth or USB |
| `"generic_usb"` | Any SDL2-compatible USB gamepad with a configurable axis/button map |
| `"keyboard"` | WASD keyboard input via raw stdin — ideal for SSH sessions |

### Generic USB Controller Mapping

When using `"controller_type": "generic_usb"`, add an optional `generic_usb_controller` block to your config (defaults shown):

```json
{
  "controller_type": "generic_usb",
  "generic_usb_controller": {
    "joystick_index": 0,
    "axis_map": {
      "left_x": 0,
      "left_y": 1,
      "right_x": 2,
      "right_y": 3,
      "left_trigger": 4,
      "right_trigger": 5
    },
    "button_map": {
      "A": 0,
      "B": 1,
      "X": 2,
      "Y": 3,
      "LB": 4,
      "RB": 5
    }
  }
}
```

---

## Hardware Configuration

### Speaker Wiring

Follow the [Adafruit MAX98357 I2S Class-D Mono Amp](https://learn.adafruit.com/adafruit-max98357-i2s-class-d-mono-amp?view=all) tutorial for wiring.

> **Note:** Do **not** activate `/dev/zero` when prompted by the tutorial.

---

## Testing and Calibration

### Test the IMU

```bash
# Quick sanity check
python3 src/open_duck_mini_runtime/raw_imu.py

# Visualise IMU data (server on robot, client on your machine)
python3 dev/hardware/imu_server.py                   # on the robot
python3 dev/hardware/imu_client.py --ip <robot_ip>   # on your machine
```

Use `ifconfig` on the robot to find its IP address.

### Find Joint Offsets

This script guides you through finding the correct resting-position offsets for each servo. Add the reported values to `~/duck_config.json` under `joints_offset`.

```bash
python3 tools/find_soft_offsets.py
```

> **Note:** This step will eventually be replaced by flashing offsets into each motor's EEPROM.

---

## Running the Duck

### Gamepad Walking

Use an Xbox One, Xbox Series, or DualSense controller paired over Bluetooth (or DualSense via USB).

```bash
uv run walk                           # uses ~/duck_config.json
uv run walk --help                    # show all options
uv run walk --onnx_model_path /path/to/model.onnx
```

**Xbox One Controller Bluetooth Pairing**

1. Long-press the sync button on the controller to enter pairing mode.
2. On the Pi:
   ```bash
   bluetoothctl
   scan on
   # wait for your controller MAC to appear, then:
   pair    <MAC>
   trust   <MAC>
   connect <MAC>
   ```

### SSH Keyboard Walking

Walk the duck entirely over SSH — no Bluetooth controller required.

```bash
uv run walk-keyboard                  # uses ~/duck_config.json
uv run walk-keyboard --help           # show all options
```

The terminal switches to raw mode while running. Press **Space** to pause, **Ctrl-C** to quit.

---

## Controls Reference

### Xbox / DualSense Controls

| Input | Action |
|---|---|
| **Left stick** | Forward / Back / Turn |
| **Right stick X** | Strafe left / right |
| **LB (hold)** | Sprint (increase walk frequency) |
| **A** | Pause / Unpause |
| **X** | Toggle projector |
| **B** | Play a random sound |
| **Y** | Toggle head control *(experimental)* |
| **Left trigger** | Left antenna |
| **Right trigger** | Right antenna |

### Keyboard Controls (SSH)

| Key | Action |
|---|---|
| **W / S** | Forward / Backward |
| **A / D** | Turn left / Turn right |
| **Q / E** | Strafe left / Strafe right |
| **L (hold)** | Sprint (increase walk frequency) |
| **Space** | Pause / Unpause |
| **X** | Toggle projector |
| **B** | Play a random sound |
| **P** | Toggle head control *(experimental)* |

---

## Running Tests

### Unit Tests (no hardware required)

```bash
uv run pytest
```

85 tests covering duck_config parsing, button state machine, keyboard controller commands, controller dispatch, RL utilities, and DualSense API parity. No connected robot needed.

### Hardware Integration Tests

These tests require the duck to be plugged in and powered on (`/dev/ttyACM0`).

```bash
uv run pytest -m hardware
```

Runs `tests/test_servo_presence.py`, which polls each of the 14 servos individually and reports any that do not respond.

To explicitly exclude hardware tests during normal development:

```bash
uv run pytest -m "not hardware"
```
