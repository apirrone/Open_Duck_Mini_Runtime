# Shared NixOS module for Open Duck Mini Runtime.
# Imported by all Pi board configs (pi-zero2w, pi4, pi5).
# Handles: hardware interfaces, user groups, SSH, system packages,
# and installation of the Python runtime virtualenv.
{
  lib,
  pkgs,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
  ...
}: let
  runtimeEnv = import ./package.nix {
    inherit pkgs uv2nix pyproject-nix pyproject-build-systems;
  };
in {
  # ── Hardware interfaces ────────────────────────────────────────────────────

  # Enable I2C, SPI, and I2S via device tree parameters.
  # I2C:  BNO055 IMU on /dev/i2c-1
  # SPI:  NeoPixel LED strip
  # I2S:  Adafruit MAX98357A mono amp (dtoverlay=max98357a required)
  hardware.raspberry-pi.config.all.base-dt-params = {
    i2c_arm = {enable = true; value = "on";};
    spi     = {enable = true; value = "on";};
    i2s     = {enable = true; value = "on";};
  };

  # I2S amp overlay for Adafruit MAX98357A.
  # See: https://learn.adafruit.com/adafruit-max98357-i2s-class-d-mono-amp/raspberry-pi-usage
  # If this attribute path is invalid for your nixos-raspberrypi version, replace with:
  #   boot.kernelParams = [ "dtparam=i2s=on" "dtoverlay=max98357a" ];
  hardware.raspberry-pi.config.all.dt-overlays = {
    max98357a = {enable = true;};
  };

  # Load kernel modules so /dev/i2c-* and /dev/spidev* get character devices
  boot.kernelModules = ["i2c-dev" "spi-dev"];

  # Reduce console noise on serial terminals
  boot.consoleLogLevel = 3;

  # udev: reduce USB serial latency for motor control board (FTDI adapter)
  services.udev.extraRules = ''
    SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
  '';

  # ── Network ───────────────────────────────────────────────────────────────

  networking.firewall.enable = lib.mkDefault false;
  networking.networkmanager.enable = lib.mkDefault true;
  networking.useNetworkd = lib.mkDefault false;
  networking.wireless.enable = lib.mkDefault false;

  # ── SSH ───────────────────────────────────────────────────────────────────

  services.openssh = {
    enable = true;
    settings = {
      PasswordAuthentication = lib.mkForce true;
      PermitRootLogin = lib.mkForce "yes";
    };
  };

  # ── Nix settings ──────────────────────────────────────────────────────────

  nix.settings.experimental-features = lib.mkForce ["nix-command" "flakes"];

  # ── Users ─────────────────────────────────────────────────────────────────
  # Hardware access groups:
  #   dialout  - serial port (/dev/ttyACM0 motor controller)
  #   i2c      - I2C bus (BNO055 IMU)
  #   gpio     - GPIO pins (foot contact sensors, NeoPixels)
  #   spi      - SPI bus
  #   video    - camera / frame buffer access
  #   tty      - terminal access

  users.users.operator = {
    isNormalUser = true;
    group = "users";
    extraGroups = ["wheel" "dialout" "tty" "networkmanager" "i2c" "gpio" "spi" "video"];
    initialPassword = "operator";
  };

  users.users.wyant = {
    isNormalUser = true;
    group = "users";
    extraGroups = ["wheel" "dialout" "tty" "networkmanager" "i2c" "gpio" "spi" "video"];
    initialPassword = "wyant";
  };

  # ── Python runtime ─────────────────────────────────────────────────────────
  # The virtualenv is installed system-wide. Entry points `walk`,
  # `walk-keyboard`, `test-hardware`, and `test-hw` are available on PATH
  # from runtimeEnv/bin/.
  #
  # Run `test-hw` for the interactive Textual TUI (select and test each
  # hardware subsystem independently with live feedback):
  #   test-hw
  #
  # Or use the non-interactive CLI to run all tests in sequence:
  #   test-hardware                  # full suite, no servo motion
  #   test-hardware --no-move        # explicit safe mode (same as default)
  #   test-hardware --skip-sounds    # skip if no speaker attached
  #   test-hardware --skip-servos    # skip if motor controller not connected
  #
  # Pi 5 note: uv.lock pins RPi.GPIO which doesn't support Pi 5 kernel.
  # If using foot contact sensors on Pi 5, install lgpio separately:
  #   uv pip uninstall RPi.GPIO && uv pip install lgpio

  environment.systemPackages = with pkgs; [
    runtimeEnv

    # Development and debugging tools
    git
    uv
    vim
    tmux
    screen
    htop
    python3
    i2c-tools
    alsa-utils  # aplay / amixer — useful for debugging I2S audio
    minicom
    usbutils
    pciutils
    iproute2
    curl
    wget
  ];

  # ── Systemd services ──────────────────────────────────────────────────────

  # Run hardware validation (pytest unit tests + hardware probe) every boot.
  # Results visible via: journalctl -u duck-validation
  systemd.services.duck-validation = {
    description = "Duck hardware validation (unit tests + hardware probe)";
    wantedBy = ["multi-user.target"];
    after = ["local-fs.target" "systemd-udevd.service"];
    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
      ExecStart = "${runtimeEnv}/bin/python -m open_duck_mini_runtime.validate";
      StandardOutput = "journal+console";
      User = "operator";
      WorkingDirectory = "/home/operator";
      Environment = "HOME=/home/operator";
    };
  };

  # Serve the Textual TUI in a browser on port 7000.
  # Access at: http://<hostname>:7000
  # Logs: journalctl -u duck-walk-serve
  systemd.services.duck-walk-serve = {
    description = "Duck runtime Textual web TUI (port 7000)";
    wantedBy = ["multi-user.target"];
    after = ["network.target" "duck-validation.service"];
    serviceConfig = {
      Type = "simple";
      ExecStart = "${runtimeEnv}/bin/walk-serve";
      Restart = "always";
      RestartSec = "5s";
      User = "operator";
      WorkingDirectory = "/home/operator";
      Environment = "HOME=/home/operator";
    };
  };

  # ── File system ───────────────────────────────────────────────────────────

  fileSystems."/" = {
    device = "/dev/disk/by-label/NIXOS_SD";
    fsType = "ext4";
  };

  system.stateVersion = lib.mkDefault "25.05";
}
