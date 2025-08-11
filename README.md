# Open Duck Mini Runtime

This repository contains the runtime software for the Open Duck Mini, a small, open-source robotic duck. This guide will walk you through setting up the hardware and software to get your duck waddling.

## Table of Contents
- [Open Duck Mini Runtime](#open-duck-mini-runtime)
  - [Table of Contents](#table-of-contents)
  - [Raspberry Pi Setup](#raspberry-pi-setup)
    - [Install Raspberry Pi OS](#install-raspberry-pi-os)
    - [Setup SSH (If not setup during the installation)](#setup-ssh-if-not-setup-during-the-installation)
    - [System Updates and Dependencies](#system-updates-and-dependencies)
    - [Enable I2C](#enable-i2c)
    - [Set the USB Serial Latency Timer](#set-the-usb-serial-latency-timer)
    - [Motor Control Board udev Rules](#motor-control-board-udev-rules)
  - [Install the Runtime](#install-the-runtime)
    - [Make a Virtual Environment and Activate it](#make-a-virtual-environment-and-activate-it)
    - [Install the Repository](#install-the-repository)
    - [Xbox One Controller Setup](#xbox-one-controller-setup)
  - [Hardware Configuration](#hardware-configuration)
    - [Speaker Wiring and Configuration](#speaker-wiring-and-configuration)
  - [Testing and Calibration](#testing-and-calibration)
    - [Test the IMU](#test-the-imu)
    - [Test Motors](#test-motors)
    - [Make your duck\_config.json](#make-your-duck_configjson)
    - [Find the Joints Offsets](#find-the-joints-offsets)
  - [Run the walk !](#run-the-walk-)
  - [Controls](#controls)

## Raspberry Pi Setup

These instructions are for setting up a Raspberry Pi Zero 2W.

### Install Raspberry Pi OS

1.  Download [Raspberry Pi OS Lite (64-bit)](https://www.raspberrypi.com/software/operating-systems/).
2.  Follow the official instructions to install the OS on an SD card: [Getting Started Guide](https://www.raspberrypi.com/documentation/computers/getting-started.html).
3.  Using the Raspberry Pi Imager, you can pre-configure your user, Wi-Fi, and SSH settings.

    ![imager_setup](https://github.com/user-attachments/assets/7a4987b2-de83-41dd-ab7f-585259685f16)
    > **Tip:** Configure the Raspberry Pi to connect to your phone's hotspot for easy access anywhere.

### Setup SSH (If not setup during the installation)

If you didn't enable SSH during the OS installation, you'll need a screen and keyboard for the initial boot.

1.  Connect to a Wi-Fi network.
2.  Enable SSH using this guide: [Raspberry Pi Configuration](https://www.raspberrypi.com/documentation/computers/configuration.html#setting-up-wifi).

Once SSH is enabled, you can connect to your Raspberry Pi remotely.

### System Updates and Dependencies

Update your system and install the required packages:

```bash
sudo apt update
sudo apt upgrade
sudo apt install git python3-pip python3-virtualenvwrapper
# Optional for camera support
sudo apt install python3-picamzero
```

Add the following lines to the end of your `.bashrc` file to configure the virtual environment wrapper:

```bash
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/Devel
source /usr/share/virtualenvwrapper/virtualenvwrapper.sh
```

### Enable I2C

Use the Raspberry Pi configuration tool to enable I2C:
`sudo raspi-config` -> `Interface Options` -> `I2C`

*(TODO: Set to 400KHz?)*

### Set the USB Serial Latency Timer

Create a udev rule to set the latency timer for the USB-to-serial adapter:
```bash
sudo nano /etc/udev/rules.d/99-usb-serial.rules
```
Add the following line to the file:
```
SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
```

### Motor Control Board udev Rules

*(TODO)*

## Install the Runtime

### Make a Virtual Environment and Activate it

```bash
mkvirtualenv -p python3 open-duck-mini-runtime
workon open-duck-mini-runtime
```

### Install the Repository

Clone the repository and install it in editable mode:
```bash
git clone https://github.com/apirrone/Open_Duck_Mini_Runtime
cd Open_Duck_Mini_Runtime
pip install -e .
```

**For Raspberry Pi 5:** You may need to replace the GPIO library.
```bash
pip uninstall -y RPi.GPIO
pip install lgpio
```


### Xbox One Controller Setup

1.  Turn on your Xbox One controller and put it in pairing mode by long-pressing the sync button.
2.  On your Raspberry Pi, run the following commands:
    ```bash
    bluetoothctl
    scan on
    ```
3.  Wait for the controller to appear, then pair, trust, and connect to it:
    ```bash
    pair <controller_mac_address>
    trust <controller_mac_address>
    connect <controller_mac_address>
    ```
    The controller's LED should stop blinking.

4.  Test the connection:
    ```bash
    python3 mini_bdx_runtime/xbox_controller.py
    ```

## Hardware Configuration

### Speaker Wiring and Configuration
Follow this Adafruit tutorial for wiring the speaker: [Adafruit MAX98357 I2S Class-D Mono Amp](https://learn.adafruit.com/adafruit-max98357-i2s-class-d-mono-amp?view=all).

> **Note:** For now, do not activate `/dev/zero` when prompted in the tutorial.

## Testing and Calibration

### Test the IMU

Run a basic test to ensure the IMU is working:
```bash
python3 mini_bdx_runtime/raw_imu.py
```

To visualize the IMU data, run the server on the robot and the client on your computer:
```bash
# On the robot
python3 dev/hardware/imu_server.py

# On your computer
python3 dev/hardware/imu_client.py --ip <robot_ip>
```
> Use `ifconfig` on the robot to find its IP address.

### Test Motors
Verify that all motors are connected and configured correctly:
```bash
python3 dev/hardware/configure_all_motors.py
```

### Make your duck_config.json

Copy the example configuration file to your home directory and rename it:
```bash
cp example_config.json ~/duck_config.json
```
This file allows you to configure features like expressions, IMU orientation, and joint offsets.

### Find the Joints Offsets

This script helps you find the correct joint offsets for your robot. The offsets should be added to your `duck_config.json` file.
```bash
python3 tools/find_soft_offsets.py
```
> **Note:** This step will be replaced in the future by flashing offsets directly to each motor's EEPROM.

## Run the walk !

1.  Download the [latest policy checkpoint](https://github.com/apirrone/Open_Duck_Mini/blob/v2/BEST_WALK_ONNX_2.onnx).
2.  Copy the checkpoint file to your duck.
3.  Run the walk script:
    ```bash
    python3 run_rl_walk.py --onnx_model_path <path_to>/BEST_WALK_ONNX_2.onnx
    ```

## Controls

-   **A**: Pause/Unpause
-   **X**: Turn on/off the projector
-   **B**: Play a random sound
-   **Y**: Turn on/off head control (experimental, use with caution)
-   **Left/Right Triggers**: Control the left and right antennas
-   **LB (Hold)**: Increase walking frequency (sprint mode)