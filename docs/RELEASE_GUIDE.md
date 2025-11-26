# This will help you create your own .iso release of the Open Duck Runtime environment, if you don't want to use an official release

BEWARE! You only need to read this if you are looking to create your own distributable .iso file of your Open Duck, or if you did not use one of the release images but still want to add the additional features as described below.

Firstly, you need to write the unmodified Linux OS to the SD card, then set-up the SD card of your RaspberryPI as described in the tutorial, like cloning the Open Duck Runtime repository and getting everything to work in your duck.
As a next step, you can modify the system as described below in order to add the quality of life features (automatic duck hotspot, start scripts) that are contained in the release images.

# You will be adding

1. an **optional config file** for overrides
2. a tiny **GPIO chooser** (decides which robot script to run)
3. a **unified launcher** that auto-detects the correct username/paths
4. a **systemd service** that runs the launcher at boot
5. the **Wi-Fi fallback** (hotspot) scripts + service
8. a **auto venv mounter** and **user information** that is shown after SSH login

---

# 0) Prerequisites (one-time)

**A. Make sure NetworkManager is used** (Bookworm does by default):

```bash
systemctl is-active NetworkManager
```

If this returns “active”, you’re good. If not, you'll need to adapt for hostapd/dnsmasq.

**B. Set Wi-Fi country (required for AP mode):**

```bash
sudo raspi-config nonint do_wifi_country DE
```
As an example if you are based in Germany.

**C. Create/verify the hotspot profile (SSID “Openduck”):**

```bash
sudo nmcli connection add type wifi ifname wlan0 con-name Openduck ssid Openduck
sudo nmcli connection modify Openduck 802-11-wireless.mode ap 802-11-wireless.band bg
sudo nmcli connection modify Openduck ipv4.method shared ipv6.method ignore
sudo nmcli connection modify Openduck wifi-sec.key-mgmt wpa-psk wifi-sec.psk "openduck1234"
sudo nmcli connection modify Openduck connection.autoconnect no
```

(You can change the password later with one command—see the end.)

---

# 1) Optional config (lets you override defaults later)

This file is optional. If you skip it, the scripts will **auto-detect** the right user and paths.

**Command to create the file:**

```bash
sudo nano /etc/default/openduck
```

**Paste this content, then save & close:**

```
# Optional overrides. Leave empty to auto-detect.
OPENDUCK_USER=
PROJECT_DIR_NAME=Open_Duck_Mini_Runtime
VENV_NAME=open-duck-mini-runtime
PIN_WALK=22
PIN_HEAD=27
ONNX_MODEL_PATH=BEST_WALK_ONNX_2.onnx
```

---

# 2) GPIO chooser (decides which mode to run)

**Make folder and create the file:**

```bash
sudo mkdir -p /usr/local/lib/openduck
sudo nano /usr/local/lib/openduck/check_gpio.py
```

**Paste this content, then save & close:**

```python
import os, sys, time
import RPi.GPIO as GPIO

PIN_WALK = int(os.getenv('PIN_WALK', '22'))   # LOW => walk
PIN_HEAD = int(os.getenv('PIN_HEAD', '27'))   # LOW => head puppet

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(PIN_WALK, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(PIN_HEAD, GPIO.IN, pull_up_down=GPIO.PUD_UP)

time.sleep(0.1)  # stabilize

if GPIO.input(PIN_WALK) == 0:
    GPIO.cleanup(); sys.exit(1)  # WALK
elif GPIO.input(PIN_HEAD) == 0:
    GPIO.cleanup(); sys.exit(2)  # HEAD
else:
    GPIO.cleanup(); sys.exit(0)  # DO NOTHING
```

---

# 3) Unified launcher (auto-detects username & paths)

This wrapper activates the right virtualenv, checks GPIOs, and runs **either**:

* `v2_rl_walk_mujoco.py --onnx_model_path …` **or**
* `head_puppet.py`
  Or exits (does nothing) if no switch is closed.

**Create the file:**

```bash
sudo nano /usr/local/bin/start-walk.sh
```

**Paste this content, then save & close:**

```bash
#!/bin/bash
set -euo pipefail

log(){ echo "[openduck-launcher] $*"; logger -t openduck-launcher "$*"; }
[ -f /etc/default/openduck ] && . /etc/default/openduck

PROJECT_DIR_NAME="${PROJECT_DIR_NAME:-Open_Duck_Mini_Runtime}"
VENV_NAME="${VENV_NAME:-open-duck-mini-runtime}"
PIN_WALK="${PIN_WALK:-22}"
PIN_HEAD="${PIN_HEAD:-27}"
ONNX_MODEL_PATH="${ONNX_MODEL_PATH:-BEST_WALK_ONNX_2.onnx}"

detect_run_user() {
  if [[ -n "${OPENDUCK_USER:-}" ]] && id "$OPENDUCK_USER" &>/dev/null; then echo "$OPENDUCK_USER"; return; fi
  for d in /home/*/.virtualenvs/"$VENV_NAME"; do [[ -d "$d" ]] && basename "$(dirname "$(dirname "$d")")" && return; done
  for d in /home/*/"$PROJECT_DIR_NAME"; do [[ -d "$d" ]] && basename "$(dirname "$d")" && return; done
  getent passwd 1000 | cut -d: -f1
}

RUN_AS="$(detect_run_user)"
if ! id "$RUN_AS" &>/dev/null; then log "ERROR: no valid run user"; exit 1; fi
HOME_DIR="$(getent passwd "$RUN_AS" | cut -d: -f6)"
PROJECT_DIR="$HOME_DIR/$PROJECT_DIR_NAME"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
VENV_DIR="$HOME_DIR/.virtualenvs/$VENV_NAME"
PY="$VENV_DIR/bin/python"

log "RUN_AS=$RUN_AS"
log "PROJECT_DIR=$PROJECT_DIR"
log "VENV_DIR=$VENV_DIR"

[[ -x "$PY" ]] || { log "ERROR: Python not found at $PY"; exit 1; }
[[ -d "$PROJECT_DIR" ]] || { log "ERROR: Project dir missing: $PROJECT_DIR"; exit 1; }
[[ -f "$SCRIPTS_DIR/head_puppet.py" ]] || { log "ERROR: head_puppet.py missing"; exit 1; }
[[ -f "$SCRIPTS_DIR/v2_rl_walk_mujoco.py" ]] || { log "ERROR: v2_rl_walk_mujoco.py missing"; exit 1; }

# Model path (absolute), but we'll run from scripts/ for relative assets like polynomial_coefficients.pkl
if [[ "${ONNX_MODEL_PATH}" = /* ]]; then MODEL_PATH="${ONNX_MODEL_PATH}"; else MODEL_PATH="${SCRIPTS_DIR}/${ONNX_MODEL_PATH}"; fi

# ---- PYTHONPATH: order matters. Carrier first, then src, then project root. ----
CARRIER_DIR="$PROJECT_DIR/mini_bdx_runtime"
export OPENDUCK_PYTHONPATH="$CARRIER_DIR:$PROJECT_DIR/src:$PROJECT_DIR"
log "PYTHONPATH will include (in order): $OPENDUCK_PYTHONPATH"

# ---- SDL/pygame runtime dir (silence XDG complaint) ----
RUN_UID="$(id -u "$RUN_AS")"
RUNDIR="/run/user/$RUN_UID"
if [[ ! -d "$RUNDIR" ]]; then
  RUNDIR="/tmp/xdg-$RUN_AS"
  mkdir -p "$RUNDIR"
  chown "$RUN_AS:$RUN_AS" "$RUNDIR"
  chmod 700 "$RUNDIR"
fi
export OPENDUCK_XDG_RUNTIME_DIR="$RUNDIR"

# Helper: run as user, with venv, PYTHONPATH, XDG dir, and chosen CWD
run_user_in_dir() {
  local workdir="$1"; shift
  runuser -l "$RUN_AS" -c "bash -lc 'export XDG_RUNTIME_DIR=\"$OPENDUCK_XDG_RUNTIME_DIR\"; export PYTHONPATH=\"$OPENDUCK_PYTHONPATH:\${PYTHONPATH:-}\"; source \"$VENV_DIR/bin/activate\"; cd \"$workdir\"; $*'"
}

# Decide which mode to launch (22=walk, 27=head)
export PIN_WALK PIN_HEAD
set +e
run_user_in_dir "$PROJECT_DIR" "\"$PY\" /usr/local/lib/openduck/check_gpio.py"
MODE=$?
set -e
log "GPIO chooser exit=$MODE (1=walk, 2=head, 0=none)"

case "$MODE" in
  1)
    log "GPIO$PIN_WALK LOW -> starting WALK (CWD=scripts)"
    run_user_in_dir "$SCRIPTS_DIR" "\"$PY\" v2_rl_walk_mujoco.py --onnx_model_path \"$MODEL_PATH\""
    ;;
  2)
    log "GPIO$PIN_HEAD LOW -> starting HEAD (CWD=scripts)"
    run_user_in_dir "$SCRIPTS_DIR" "\"$PY\" head_puppet.py"
    ;;
  0)
    log "No switches closed. Exiting."
    exit 0
    ;;
  *)
    log "Unexpected MODE=$MODE"; exit 1
    ;;
esac
```

**Make it executable:**

```bash
sudo chmod +x /usr/local/bin/start-walk.sh
```

---

# 4) Systemd service for the launcher

**Create the service file:**

```bash
sudo nano /etc/systemd/system/open-duck-walk.service
```

**Paste this content, then save & close:**

```
[Unit]
Description=Open Duck mode launcher (GPIO-based)
After=network.target NetworkManager.service openduck-wifi-fallback.service
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=-/etc/default/openduck
ExecStart=/usr/local/bin/start-walk.sh
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

---

# 5) Wi-Fi “Openduck” fallback (hotspot if still offline)

## 5a) Boot-time fallback script

**Create the file:**

```bash
sudo nano /usr/local/bin/openduck-wifi-fallback.sh
```

**Paste this content, then save & close:**

```bash
#!/bin/bash
set -euo pipefail
log(){ logger -t openduck-fallback "$*"; echo "$*"; }

timeout_sec=15
log "Waiting up to ${timeout_sec}s for NetworkManager to connect..."
if nm-online -q -t "$timeout_sec"; then
  log "Connected. Ensure hotspot is down (if active)."
  nmcli -t -f NAME con show --active | grep -qx 'Openduck' && nmcli con down Openduck || true
  exit 0
fi

log "Still offline after wait; checking wifi state..."
nmcli radio || true
nmcli device status || true

nmcli dev wifi rescan || true
visible_count=$(nmcli -t -f SSID dev wifi list | awk 'length>0' | wc -l || echo 0)
log "Visible SSIDs (info): ${visible_count}"

if ! nm-online -q -t 1; then
  log "Offline ⇒ bringing up 'Openduck' AP…"
  nmcli con up Openduck || { log "Failed to start Openduck AP"; exit 0; }
  log "Hotspot started."
else
  log "Came online during checks; not starting AP."
fi
exit 0
```

**Make it executable:**

```bash
sudo chmod +x /usr/local/bin/openduck-wifi-fallback.sh
```

## 5b) Auto-tear-down/re-enable hook (runs on NM events)

**Create the file:**

```bash
sudo nano /etc/NetworkManager/dispatcher.d/99-openduck-hotspot
```

**Paste this content, then save & close:**

```bash
#!/bin/bash
# $1 = iface, $2 = event
IFACE="$1"; EVENT="$2"
[ "$IFACE" = "wlan0" ] || exit 0

logger -t openduck "dispatcher: $IFACE $EVENT"

# If we connect to any non-Openduck Wi-Fi, ensure hotspot is down
if [ "$EVENT" = "up" ]; then
  active_on_wlan0=$(nmcli -t -f NAME,DEVICE con show --active | awk -F: '$2=="wlan0"{print $1}')
  if [ -n "$active_on_wlan0" ] && [ "$active_on_wlan0" != "Openduck" ]; then
    nmcli -t -f NAME con show --active | grep -qx 'Openduck' && nmcli con down Openduck || true
  fi
fi

# If we go offline and no networks are usable, bring AP up
if [ "$EVENT" = "down" ] || [ "$EVENT" = "connectivity-change" ] || [ "$EVENT" = "dhcp4-change" ]; then
  if ! nm-online -q -t 1; then
    nmcli dev wifi rescan || true
    visible_count=$(nmcli -t -f SSID dev wifi list | awk 'length>0' | wc -l || echo 0)
    if [ "$visible_count" -eq 0 ]; then
      nmcli -t -f NAME con show --active | grep -qx 'Openduck' || nmcli con up Openduck
    end if
  fi
fi
exit 0
```

**Make it executable:**

```bash
sudo chmod +x /etc/NetworkManager/dispatcher.d/99-openduck-hotspot
```

## 5c) Service to run the fallback at boot

**Create the service file:**

```bash
sudo nano /etc/systemd/system/openduck-wifi-fallback.service
```

**Paste this content, then save & close:**

```
[Unit]
Description=Openduck Wi-Fi fallback (start hotspot if offline)
After=NetworkManager.service
Wants=network-online.target
Before=open-duck-walk.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/openduck-wifi-fallback.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

---

# 6) Enable and start everything

```bash
sudo systemctl daemon-reload
sudo systemctl enable openduck-wifi-fallback.service
sudo systemctl enable open-duck-walk.service
sudo systemctl restart openduck-wifi-fallback.service
sudo systemctl restart open-duck-walk.service
```

---

# 7) How to test

* **See launcher logs live:**

  ```bash
  sudo journalctl -f | grep openduck
  ```
* **Run launcher once (without reboot) to verify user/paths:**

  ```bash
  sudo /usr/local/bin/start-walk.sh
  ```
* **See hotspot come up if offline:**

  ```bash
  nmcli -t -f NAME,TYPE,DEVICE con show --active
  ```

---

# 8) Everyday admin

* **Disable auto-hotspot at boot:**

  ```bash
  sudo systemctl disable openduck-wifi-fallback.service
  sudo systemctl stop openduck-wifi-fallback.service
  ```
* **Re-enable auto-hotspot:**

  ```bash
  sudo systemctl enable openduck-wifi-fallback.service
  sudo systemctl start openduck-wifi-fallback.service
  ```
* **Change hotspot password:**

  ```bash
  sudo nmcli connection modify Openduck wifi-sec.psk "NEWstrongPASS123"
  sudo nmcli connection down Openduck || true
  sudo nmcli connection up Openduck
  ```
* **Check services:**

  ```bash
  systemctl is-enabled open-duck-walk.service
  systemctl is-enabled openduck-wifi-fallback.service
  sudo systemctl status open-duck-walk.service
  sudo systemctl status openduck-wifi-fallback.service
  ```

# 9) Automatically mount the virtual environment and show greeting message

This adds a friendly **SSH login banner** (with a simple mute switch) and **auto-activates the venv** so users don’t get confused. Everything is username-agnostic.

---

## 9a) (Optional) Defaults file

**Create (or update) the defaults file:**
bash
```bash
sudo nano /etc/default/openduck
```

**Paste this content, then save & close:**
```
# Used by the login hook
PROJECT_DIR_NAME=Open_Duck_Mini_Runtime
VENV_NAME=open-duck-mini-runtime
```

---

## 9b) Helper CLI to mute the banner / toggle auto-venv

**Create the helper:**
bash
```bash
sudo nano /usr/local/bin/openduck-login
```

**Paste this content, then save & close:**
```bash
#!/bin/bash
set -euo pipefail
CONFIG_DIR="${HOME}/.config/openduck"
MUTE_WELCOME="${CONFIG_DIR}/mute_welcome"
NO_AUTO_VENV="${CONFIG_DIR}/no_auto_venv"
mkdir -p "${CONFIG_DIR}"

cmd="${1:-status}"; arg="${2:-}"

case "$cmd" in
  welcome)
    case "$arg" in
      off|mute)   : >"${MUTE_WELCOME}";  echo "Welcome message OFF for ${USER}";;
      on|unmute)  rm -f "${MUTE_WELCOME}"; echo "Welcome message ON for ${USER}";;
      *) echo "Usage: openduck-login welcome on|off"; exit 1;;
    esac
    ;;
  venv)
    case "$arg" in
      off|disable) : >"${NO_AUTO_VENV}";  echo "Auto-venv OFF for ${USER}";;
      on|enable)   rm -f "${NO_AUTO_VENV}"; echo "Auto-venv ON for ${USER}";;
      *) echo "Usage: openduck-login venv on|off"; exit 1;;
    esac
    ;;
  status|*)
    echo "Welcome:   $([ -f "${MUTE_WELCOME}" ] && echo OFF || echo ON)"
    echo "Auto-venv: $([ -f "${NO_AUTO_VENV}" ] && echo OFF || echo ON)"
    echo "Usage:  openduck-login welcome on|off   # show/hide welcome"
    echo "        openduck-login venv on|off      # enable/disable auto-venv"
    ;;
esac
```

**Make it executable:**
bash
```bash
sudo chmod +x /usr/local/bin/openduck-login
```

---

## 9c) Login hook (banner + auto-activate venv + safe PYTHONPATH)

**Create the profile script:**
bash
```bash
sudo nano /etc/profile.d/openduck-init.sh
```

**Paste this content, then save & close:**
```bash
# OpenDuck login helper: banner + auto-venv + PYTHONPATH for SSH/interactive shells.

# Load defaults if present
[ -f /etc/default/openduck ] && . /etc/default/openduck

# Defaults (if not provided)
PROJECT_DIR_NAME="${PROJECT_DIR_NAME:-Open_Duck_Mini_Runtime}"
VENV_NAME="${VENV_NAME:-open-duck-mini-runtime}"

# Only for interactive shells or SSH sessions
case "$-" in *i*) interactive=1;; esac
if [ -z "${interactive:-}" ] && [ -z "${SSH_CONNECTION:-}" ]; then
  return 0
fi

# Keep virtualenvwrapper (if installed) using system Python, not the venv
[ -x /usr/bin/python3 ] && export VIRTUALENVWRAPPER_PYTHON="/usr/bin/python3"

# Per-user config flags
CONFIG_DIR="${HOME}/.config/openduck"
MUTE_WELCOME="${CONFIG_DIR}/mute_welcome"
NO_AUTO_VENV="${CONFIG_DIR}/no_auto_venv"
[ -d "$CONFIG_DIR" ] || mkdir -p "$CONFIG_DIR" >/dev/null 2>&1 || true

# ---- Friendly welcome (mute with: openduck-login welcome off) ----
if [ ! -f "$MUTE_WELCOME" ]; then
  bold="$(tput bold 2>/dev/null || true)"; normal="$(tput sgr0 2>/dev/null || true)"
  printf "\n${bold}Welcome to your Open Duck Mini${normal}\n"
  cat <<'MSG'
Please complete these steps before using the duck:

- IMU configuration / calibration
- Motor configuration
- Xbox Bluetooth pairing
- Creation of duck_config.json in your home folder

Read: ~/Open_Duck_Mini_Runtime/README.md (main repository readme).

Tip: mute this message once you're done:
  openduck-login welcome off
Unmute anytime:
  openduck-login welcome on
--------------------------------------------------------------------------------
MSG
fi

# ---- Auto-activate venv (disable with: openduck-login venv off) ----
# Skip for root, and if already inside a venv, or if user disabled it.
if [ "$(id -u)" -ne 0 ] && [ -z "${VIRTUAL_ENV:-}" ] && [ ! -f "$NO_AUTO_VENV" ]; then
  VENV_ACT="${HOME}/.virtualenvs/${VENV_NAME}/bin/activate"
  if [ -f "$VENV_ACT" ]; then
    # Activate venv
    . "$VENV_ACT" 2>/dev/null || true

    # Helpful PYTHONPATH so imports work from anywhere
    PROJ="${HOME}/${PROJECT_DIR_NAME}"
    if [ -d "$PROJ" ]; then
      export PYTHONPATH="${PROJ}/mini_bdx_runtime:${PROJ}/src:${PROJ}:${PYTHONPATH:-}"
    fi

    echo "[OpenDuck] Activated venv '${VENV_NAME}'. Disable with: openduck-login venv off"
  fi
fi
```

**Set permissions:**
bash
```bash
sudo chmod 644 /etc/profile.d/openduck-init.sh
```

---

## 9d) How to use & test

* **Open a new SSH session** — you should see the banner and the venv prompt.
* **Check current toggles:**
bash
```bash
openduck-login status
```
* **Mute/unmute the banner:**
bash
```bash
openduck-login welcome off
openduck-login welcome on
```
* **Disable/enable auto-venv:**
bash
```bash
openduck-login venv off
openduck-login venv on
```

> Per-user settings are stored under `~/.config/openduck/` so each user can choose independently.

---

Great! So you got your Duck up and running. Before you melt down your SD card into a re-distributable image file, be sure to "clean it up" as described below.
When you clone a Pi image for other people, you want to remove anything that “fingerprints” your device or exposes your accounts. Here’s a **practical, safe cleanup** you can run **before** you make the image

---

# What to remove (and why)

**Identity (must-do)**

* **SSH host keys** → otherwise every clone has the same SSH identity.
  Files: `/etc/ssh/ssh_host_*`
* **Machine ID** → unique OS ID used by systemd/logging.
  Files: `/etc/machine-id` (truncate), `/var/lib/dbus/machine-id` (remove)

**Your user traces**

* **Shell history** for all users: `~/.bash_history`, `~/.lesshst`, `~/.wget-hsts`, `~/.python_history`
* **SSH client keys & servers you connected to**: `~/.ssh/*` (especially `authorized_keys`, `known_hosts`, any `id_*.pub/id_*`)
* **Caches**: `~/.cache`, pip cache, etc.

**Networking**

* **Saved Wi-Fi profiles & PSKs** (keep only the AP “Openduck” if you want):
  `/etc/NetworkManager/system-connections/*.nmconnection`
* **DHCP leases / NM state**: `/var/lib/NetworkManager/*`

**Logs**

* Journald logs, wtmp/btmp/lastlog, miscellaneous logs under `/var/log`

**Optional space tidy**

* APT cache, thumbnails, temp files

---

# One-time prep (first-boot regeneration)

Delete SSH host keys **and** make sure a fresh machine-id will be created on first boot:

```bash
# 1) Remove host keys (they will be re-generated on first boot)
sudo rm -f /etc/ssh/ssh_host_*

# 2) Reset machine-id so clones don’t share it
sudo truncate -s 0 /etc/machine-id
sudo rm -f /var/lib/dbus/machine-id
# (systemd will auto-generate a new /etc/machine-id on the next boot)
```

---

# Adding a script that re-generates the SSH keys after first start

Sadly the Linux distribution we use (Bookworm) is not automatically re-generating SSH keys. That means, after you create your .img, SSH will not automatically work anymore.
You can mitigate this issue by adding the little script below.

### 1) Make sure OpenSSH server is installed and enabled

```bash
sudo apt-get update
sudo apt-get install -y openssh-server
sudo systemctl enable ssh
```

### 2) Create the “ensure SSH keys” script

This script only creates host keys **if they’re missing** (which will be true on first boot after you wipe them before imaging).

**Create file:**

```bash
sudo nano /usr/local/sbin/openduck-ensure-ssh.sh
```

**Paste this content, save, exit:**

```sh
#!/bin/sh
set -e

# Only (re)generate host keys if they are missing
if ! ls /etc/ssh/ssh_host_* >/dev/null 2>&1; then
  /usr/bin/ssh-keygen -A
fi
```

**Make it executable:**

```bash
sudo chmod +x /usr/local/sbin/openduck-ensure-ssh.sh
```

### 3) Create a systemd unit that runs **before ssh.service**

It runs once at boot (very fast), **only** when keys are missing.

**Create file:**

```bash
sudo nano /etc/systemd/system/openduck-ensure-ssh.service
```

**Paste this content, save, exit:**

```ini
[Unit]
Description=Ensure SSH host keys exist (first boot safe)
# Run this only if the Ed25519 key doesn't exist yet (signals "fresh clone")
ConditionPathExists=!/etc/ssh/ssh_host_ed25519_key
# Make sure this happens before ssh tries to start
Before=ssh.service
Wants=ssh.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/openduck-ensure-ssh.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

**Enable the unit:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable openduck-ensure-ssh.service
```

> Tip: `ssh.service` is already enabled in step 1.
> Now, if host keys are missing on boot, they’re generated **before** ssh starts.



---

# Clean up users, Wi-Fi, logs (copy–paste block)

> ⚠️ Read once before running. This keeps the **Openduck** AP profile, and removes **other saved Wi-Fi**. Adjust if needed.

```bash
# --- USERS: wipe histories & SSH client creds for all home users + root ---
for d in /root /home/*; do
  [ -d "$d" ] || continue
  sudo rm -f  "$d/.bash_history" "$d/.lesshst" "$d/.wget-hsts" "$d/.python_history"
  sudo rm -rf "$d/.cache"
  # Remove any SSH client keys / known_hosts / authorized_keys
  [ -d "$d/.ssh" ] && sudo rm -f "$d/.ssh/"*
done

# --- NETWORKMANAGER: remove saved Wi-Fi profiles except the AP "Openduck" ---
# (Keeps wired profiles. Only deletes TYPE=wifi other than Openduck.)
nmcli -t -f NAME,TYPE con show | awk -F: '$2=="wifi"{print $1}' \
  | grep -v '^Openduck$' \
  | while read -r name; do sudo nmcli con delete "$name"; done

# Optional: reset NM state/leases
sudo rm -f /var/lib/NetworkManager/*lease* /var/lib/NetworkManager/NetworkManager.state

# --- LOGS: rotate + vacuum journals, clear classic login logs, truncate misc logs ---
sudo journalctl --rotate
sudo journalctl --vacuum-time=1s
sudo rm -f /var/log/wtmp /var/log/btmp
sudo truncate -s 0 /var/log/lastlog
sudo find /var/log -type f -name "*.log" -exec truncate -s 0 {} \; 2>/dev/null

# --- SPACE: apt & thumbnail caches (optional) ---
sudo apt clean
sudo rm -rf /var/cache/apt/archives/*
sudo rm -rf /var/tmp/* /tmp/* 2>/dev/null || true
```

---

# Remove your personal files

```bash
# No git login data should remain from your account, if you logged in on the Duck
rm -f ~/.gitconfig 

# Cache and temporary files that could have been created
rm -f ~/.bash_history ~/.lesshst ~/.wget-hsts ~/alsa.log ~/.sudo_as_admin_successful
rm -rf ~/.cache

# It's good practice to remove your generated duck configuration files, so the next user has to make their own
# This is both your duck_config.json, as well as your IMU calibration data and motor offsets
# e.g.

rm duck_config.json


```


---

# Verify what remains

```bash
# No host keys:
ls -l /etc/ssh/ssh_host_*        # should show nothing

# Machine-id empty:
sudo hexdump -C /etc/machine-id  # should be 0 bytes

# Only Openduck wifi:
nmcli -t -f NAME,TYPE con show   # wifi entries should be just 'Openduck' (if you kept it)

# No personal SSH:
for d in /root /home/*; do echo ">> $d"; ls -la "$d/.ssh" 2>/dev/null || true; done
```

---

# Lastly...

* **Power off cleanly** before imaging to avoid journal/fs replay on first boot:

  ```bash
  sudo sync && sudo poweroff
  ```

---

Setting up the image as described above will make it work as an official release would, with any username and SSH information set in the RPi Imager tool.
You can now insert your SD card into a SD card reader on your Desktop PC/MAC/Linux system and use the RPi imager tool to turn the SD card into an image file.
