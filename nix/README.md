# NixOS Image System — Open Duck Mini Runtime

Reproducible NixOS SD-card images for Raspberry Pi Zero 2W, Pi 4, and Pi 5.
The runtime (Python virtual environment, system services, hardware config) is
baked into the image — no manual `apt install` or `uv sync` after flashing.

---

## Supported targets

| Build target        | Hardware        | Hostname          |
|---------------------|-----------------|-------------------|
| `duck-pi-zero2w-sd` | Pi Zero 2W      | `duck-pi-zero2w`  |
| `duck-pi4-sd`       | Raspberry Pi 4  | `duck-pi4`        |
| `duck-pi5-sd`       | Raspberry Pi 5  | `duck-pi5`        |

---

## Prerequisites

- **Nix** with `flakes` and `nix-command` enabled
  Add to `~/.config/nix/nix.conf`:
  ```
  experimental-features = nix-command flakes
  ```
- (Recommended) **nixos-raspberrypi cachix** for faster builds — pre-built
  cross-compilation results are fetched automatically via the flake's `nixConfig`.

---

## Build an SD-card image

```bash
# From the Open_Duck_Mini_Runtime/ directory:
nix build .#duck-pi4-sd          # Raspberry Pi 4
nix build .#duck-pi5-sd          # Raspberry Pi 5
nix build .#duck-pi-zero2w-sd    # Pi Zero 2W
```

The compressed image lands at `result/sd-image/*.img.zst`.

---

## Flash to SD card

```bash
# Replace /dev/sdX with your SD card device (check with lsblk)
zstdcat result/sd-image/*.img.zst | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync
```

---

## First boot

1. Insert SD card and boot the Pi.
2. Connect via Wi-Fi or Ethernet.
3. SSH in:
   ```bash
   ssh operator@duck-pi4      # or use IP address
   # default password: operator
   # (wyant user also available, password: wyant)
   ```
4. Verify hardware validation passed (runs automatically every boot):
   ```bash
   journalctl -u duck-validation --no-pager
   ```

---

## One-time setup after first boot

Upload your ONNX model and configuration:

```bash
# From your development machine:
scp ~/BEST_WALK_ONNX_2.onnx operator@duck-pi4:~/
scp ~/duck_config.json        operator@duck-pi4:~/
```

If you don't have a `duck_config.json` yet, copy the template:
```bash
# On the Pi:
cp /run/current-system/sw/share/... ~/duck_config.json   # or scp from repo
```

---

## Running the duck

### Local terminal TUI (over SSH)

```bash
walk                   # Textual dashboard — shows IMU, motors, controller
walk-keyboard          # Raw terminal keyboard control (no TUI)
```

**TUI keybindings:**

| Key     | Action               |
|---------|----------------------|
| `P`     | Pause / Resume       |
| `M`     | Motors On / Off      |
| `Q`     | Quit                 |

### Browser TUI (textual-serve on port 7000)

The `duck-walk-serve` systemd service starts automatically on boot.
Open a browser and navigate to:

```
http://duck-pi4:7000
```

Or start it manually:
```bash
walk-serve             # serves on port 7000, Ctrl-C to stop
```

The browser TUI shows the same live dashboard as the terminal TUI and
supports the same `P`/`M`/`Q` keybindings.

---

## Boot validation

A systemd service (`duck-validation`) runs on every boot and checks:

1. **Unit tests** — pytest suite (no hardware required, ~5 s)
2. **Motor serial port** — `/dev/ttyACM0` present
3. **I2C bus** — `/dev/i2c-*` devices present
4. **ONNX model** — `~/BEST_WALK_ONNX_2.onnx` exists
5. **duck_config.json** — present and valid JSON

View results:
```bash
journalctl -u duck-validation --no-pager -n 50
```

---

## Rebuilding after code changes

After editing Python source, nix files, or `pyproject.toml`:

```bash
# If you changed Python dependencies:
uv lock

# Rebuild image:
nix build .#duck-pi4-sd

# Or push a live update to a running Pi (no reflash needed):
nixos-rebuild switch \
  --flake .#duck-pi4 \
  --target-host operator@duck-pi4 \
  --use-remote-sudo
```

---

## Hardware notes

| Feature              | Pi Zero 2W | Pi 4 | Pi 5 |
|----------------------|:----------:|:----:|:----:|
| RPi.GPIO             | ✓          | ✓    | ✗    |
| lgpio (Pi 5 GPIO)    | ✗          | ✗    | ✓ *  |
| I2C (BNO055 IMU)     | ✓          | ✓    | ✓    |
| Bluetooth (gamepad)  | ✓          | ✓    | ✓    |

\* Pi 5 uses `lgpio` instead of `RPi.GPIO`. If you need foot-contact GPIO
sensors on Pi 5, run after first boot:
```bash
uv pip uninstall RPi.GPIO && uv pip install lgpio
```

---

## Directory layout

```
nix/
├── README.md          # This file
├── flake.nix          # Flake inputs, outputs, build targets
├── module.nix         # Shared NixOS module (hardware, users, SSH, services)
├── package.nix        # Python virtualenv via uv2nix
├── pi-zero2w.nix      # Pi Zero 2W board config
├── pi4.nix            # Pi 4 board config
└── pi5.nix            # Pi 5 board config
```
