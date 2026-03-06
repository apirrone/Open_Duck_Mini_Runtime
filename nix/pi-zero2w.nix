# NixOS configuration for Raspberry Pi Zero 2W.
# The Zero 2W uses the BCM2710A1 chip — the same silicon as the Pi 3B+.
# nixos-raspberrypi exposes this as the "raspberry-pi-02" hardware module.
{
  lib,
  inputs,
  nixos-raspberrypi,
  ...
}: {
  imports =
    [
      ./module.nix
    ]
    ++ (with nixos-raspberrypi.nixosModules; [
      raspberry-pi-02.base
      sd-image
    ]);

  networking.hostName = lib.mkForce "duck-pi-zero2w";

  boot.loader.raspberry-pi.bootloader = "kernel";

  boot.initrd.includeDefaultModules = false;
  boot.initrd.availableKernelModules = lib.mkForce [
    "usbhid"
    "usb_storage"
    "sdhci_iproc"
    "sdhci"
    "mmc_block"
  ];

  # Pi Zero 2W uses RPi.GPIO (not lgpio).
  # withLgpio defaults to false in module.nix — no override needed.

  system.stateVersion = lib.mkForce "25.05";
  system.configurationRevision = inputs.self.rev or inputs.self.dirtyRev or null;
}
