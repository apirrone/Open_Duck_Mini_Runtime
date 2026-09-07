# Pre‑Built Duck Image (BDX V2)

A ready‑to‑flash Raspberry Pi image to get your Duck up and walking in minutes. This image is a **fully configured Linux distribution** based on Raspberry Pi Os Lite 64 Bit Bookworm, that includes all required packages, scripts, and conveniences. It will be kept in sync with this repository over time.

> This is currently only confirmed working on the Raspberry Pi Zero from the BOM
> **Important:** This approach assumes you followed the **original build tutorial** for the unmodified Duck in this repo. Hardware and wiring must match the reference build for everything to work as intended.
> **Very Important:** You will still have to configure the motor offsets and calibrate the IMU, also see the section below at „Must-Run Setup Scripts“, this will not work automatically!
> If you wish to create your own release based on your self-configured Linux copy, please follow the tutorial in the duck_release_creation_tutorial.md file

---

## At a Glance

- **File:** `duck.iso` (compressed archive provided alongside this repo).
- **Target media:** 32GB microSD cards (tested successfully on various 32GB cards).
- **Convenience:** Foot‑switch controlled autostart for **Walk** or **Head‑Puppet** mode.
- **Default access:** Two pre-configured WiFi-Setups, as described below.

---

## 1. Flashing the Image to your SD card

### On any operating system
> We use the official tool for flashing provided by RaspberryPI called 'Raspberry Pi Imager'
> You will not unpack the ZIP file containing the .ISO, but directly open the .ZIP in the imager. As we use ISO compression when building the image, trying to burn the bare .ISO file will result in an error.

- The imager will allow you to flash the duck.iso image to your SD card
- If you want, you can pre-set your home WiFi access point configuration, so you can directly connect to your Duck using SSH
- If you want, you can set your own SSH user name and password during flashing. Please be aware that for simplicity sake, we use the standard user "bdxv2" in this readme.md file and you need to switch that out if you need a custom user name.

> In some instances, the flashed SD card will initially expose only the size used by the image (≈7GB). You’ll grow it to the full card capacity on first boot.
> Do do this, SSH into your Duck and enter the command listed below at "Expand the file system"

---

## 2. First Boot: Wi‑Fi & SSH

If you did not enter your home WiFi access point configuration during the flashing process, you would usually need a keyboard with USB adapter and Micro HDMI cable.
But: No worries, we've got you covered. These are two additional ways to connect to your Duck via WiFi:

### 2.1 - Preferred - Let the Duck make its own network access point

- If the Duck fails to connect to any WiFi, it will open up its own WiFi called "Openduck"
- You can connect your Laptop or Desktop directly to this network and then access the Duck via SSH
- Note: This feature is always enabled, so if you take your Duck out on the road or show, this allows you to connect to it anytime
- To disable this feature, enter the command: sudo systemctl disable openduck-wifi-fallback.service

> If you wish to permanently use this feature, please ensure to change the password for the hotspot, so you don't get "Duck-napped"
> Please be aware that start-up time will get about 15-20 seconds longer with this feature active!
> Please be aware that this feature will sometimes auto-activate only starting from the second reboot after flashing, simply cycle power once if the access point does not come up!

``` sudo nmcli connection modify Openduck wifi-sec.psk "NEWstrongPASS123"
sudo nmcli connection down Openduck || true
sudo nmcli connection up Openduck
```

### 2.2 - Backup mode - Using your Phones tethering mode

- The Duck has a pre-set WiFi Network, that it will always connect to, if it sees it
- Create a temporary hotspot on your phone named `Duckspot` with password `12345678`, connect your laptop to the same hotspot, and power the Duck
- Connect both your laptop/desktop to the WiFi
- **SSID:** `Duckspot`  
- **Password:** `12345678`

> **Change these after setup** to prevent unauthorized access.

### SSH Access

No matter how you connect to your Duck, use this to log-in using SSH. Be aware, if you changed the name during the flashing process in RPi imager, these might have also changed.

- **Host:** `bdxv2.local` (mDNS) or the Pi’s IP address.
- **User:** `bdxv2`
- **Password:** `ilovemyduck`

```bash
ssh bdxv2@bdxv2.local
# If .local doesn't resolve on your system, use the device's IP instead.
```

**Immediately change the password after your first login:**

```bash
passwd
```

---

## 3. Expand the Filesystem (use the full SD card)

After first boot, if it didn't automatically happen, expand the root filesystem so the OS can use the entire 32GB:

```bash
sudo raspi-config nonint do_expand_rootfs
sudo reboot
```

This is especially relevant if you flashed the image on **Windows**.

---

## 4. Pair your Xbox Controller (Bluetooth)

The foot‑based start relies on an active Bluetooth connection to your own gamepad. Pair it once:

```bash
sudo bluetoothctl
# inside bluetoothctl:
power on
agent on
default-agent
scan on                # wait until you see your controller MAC/Name
pair XX:XX:XX:XX:XX:XX # replace with your controller MAC
trust XX:XX:XX:XX:XX:XX
connect XX:XX:XX:XX:XX:XX
quit
```

Turn the controller on before booting the Duck to ensure the selected mode starts automatically.

---

## 5. Foot‑Based Start (Autostart Service)

This image includes a convenience service that chooses the startup mode based on the Duck’s **foot switches** at boot:

- **No foot pressed:** do nothing (idle).
- **Left foot pressed (looking at the robot from the front):** **Head‑Puppet** mode starts.
- **Right foot pressed (looking at the robot from the front):** **Walk** mode starts.

> **Requirement:** An **Xbox** gamepad must be powered on and connected via Bluetooth, otherwise the script will fail (error clicking noise in the speaker). Pair your own controller first (see below).

### Service Management

```bash
# Check current status
sudo systemctl status open-duck-walk.service

# Start / Stop now
sudo systemctl start open-duck-walk.service
sudo systemctl stop open-duck-walk.service

# Enable / Disable at boot
sudo systemctl enable open-duck-walk.service
sudo systemctl disable open-duck-walk.service
```

---

## 6. How to run any Open Duck project scripts on the Raspberry Pi

Since the whole project is isolated inside of a virtual environment on the Raspberry Pi OS, sometimes executing scripts can be very confusing. Running projects on a Raspberry Pi inside a Python virtual environment (venv) prevents conflicts with the system Python and keeps the OS stable, since all project dependencies stay isolated. It ensures clean, reproducible setups where each project can have its own package versions without interfering with others. A venv also avoids permission issues caused by system-wide installs and makes deployment, updates, and debugging far more predictable. Overall, it’s the safest and most maintainable way to run Python-based Raspberry Pi projects.

> This release will try to automatically mount the virtual environment on start of the Raspberry Pi, so you don’t have to keep in mind to enter the venv
> If it fails, in order to run any scripts from the project at all, make sure to enter the virtual environment yourself first by typing "workon open-duck-mini-runtime“ in the terminal


---

## 6.2 Must‑Run Setup Scripts

Every Duck is slightly different. Before serious use, **run the configuration scripts** included in this repository/image. Check the main readme.md of the Open Duck Mini Runtime repository to learn how to do this.

- **IMU configuration / calibration**
- **Motor configuration**
- **Xbox Bluetooth pairing** (see above)
- **Creation of duck_config.json in home folder**

> Motor and sensor offsets and device IDs differ across builds; re‑using someone else’s values will cause poor tracking or unstable motion.
> Configuration is stored in the duck_config.json file - you can find an example in the home folder
> Please be aware the Duck will NOT START before you take care of these setup steps

---

## 7. Troubleshooting

- **Can’t SSH to `bdxv2.local`?**  
  - Use the device’s IP address instead (e.g., from your router or `arp -a` on your LAN).  
  - Install Bonjour / mDNS support if your OS doesn’t resolve `.local` hostnames.
- **Nothing starts at boot:**  
  - Ensure **neither** foot switch is stuck pressed if you expect idle.  
  - Ensure the correct foot is pressed if you expect Head‑Puppet or Walk mode.  
  - Verify the **Xbox controller is paired and powered on**.
- **Only ~7GB available on a 32GB card:**  
  - Run the **Expand the Filesystem** step above.
- **Motion feels off:**  
  - Re‑run the **IMU** and **Motor** configuration scripts.
- **Security reminders:**  
  - Change the default Wi‑Fi and SSH credentials as part of your initial setup, so you don't get Duck-napped!

---

## Fix: If you are unable to manually launch the head-puppet or walk scripts after picking a custom username (instead of bdxv2) during flashing

When you run manually from `~/Open_Duck_Mini_Runtime/scripts`, you need the **same PYTHONPATH** the automatic launcher provides:

```bash
workon open-duck-mini-runtime
export PYTHONPATH="$HOME/Open_Duck_Mini_Runtime/mini_bdx_runtime:$HOME/Open_Duck_Mini_Runtime/src:$HOME/Open_Duck_Mini_Runtime:$PYTHONPATH"
cd "$HOME/Open_Duck_Mini_Runtime/scripts"

# Walk:
python v2_rl_walk_mujoco.py --onnx_model_path BEST_WALK_ONNX_2.onnx

# Head:
python head_puppet.py
```

(That export makes it work for **any** username, since it’s based on `$HOME`.)

---

## Notes & Expectations

- This image is maintained to track this repository, but **hardware parity** with the reference build is assumed.
- If you customize wiring, sensors, or motors, adjust the configuration scripts accordingly.
- For production or public demos, secure your Duck: change all credentials, disable unneeded services, and restrict network access.

---

### Credits

Thanks to everyone contributing to the Open Duck Mini build and documentation. Have fun—and don’t let anyone steal your Duck control! 🦆
