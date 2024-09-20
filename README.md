# Open Duck Mini Runtime

TODO : Write a description

## Raspberry Pi zero 2W setup

### Install Raspberry Pi OS

Download Raspberry Pi OS Lite (64-bit) from here : https://www.raspberrypi.com/software/operating-systems/

Follow the instructions here to install the OS on the SD card : https://www.raspberrypi.com/documentation/computers/getting-started.html

### Setup SSH

When first booting on the rasp, you will need to connect a screen and a keyboard. The first thing you should do is connect to a wifi network and enable SSH.

To do so, you can follow this guide : https://www.raspberrypi.com/documentation/computers/configuration.html#setting-up-wifi

Then, you can connect to your rasp using SSH without having to plug a screen and a keyboard.

### Enable UART

First disable the shell over serial. Run `sudo raspi-config`, navigate to `Interface Options`, then `Serial Port`, and select `No`, then `Yes`.

Then, edit the file `/boot/firmware/config.txt` and add the following line at the end of the file :

```
enable_uart=1
```

### Set the usbserial latency timer

```bash
cd  /etc/udev/rules.d/
sudo touch 99-usb-serial.rules
sudo nano 99-usb-serial.rules
# copy the following line in the file
SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
```


## Install the runtime

Clone this repository on your rasp, cd into the repo, then :

```bash
pip install -e .
```

## Test the IMU

```bash
cd scripts/
python imu_test.py
```
