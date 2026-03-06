# NixOS configuration for Raspberry Pi 4.
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
      raspberry-pi-4.base
      raspberry-pi-4.bluetooth
      sd-image
    ]);

  networking.hostName = lib.mkForce "duck-pi4";

  boot.loader.raspberry-pi.bootloader = "kernel";

  boot.initrd.includeDefaultModules = false;
  boot.initrd.availableKernelModules = lib.mkForce [
    "xhci_pci"
    "xhci_hcd"
    "usbhid"
    "usb_storage"
    "sdhci_iproc"
    "sdhci"
    "mmc_block"
  ];

  # Pi 4 uses RPi.GPIO (not lgpio).
  # withLgpio defaults to false in module.nix — no override needed.

  system.stateVersion = lib.mkForce "25.05";
  system.configurationRevision = inputs.self.rev or inputs.self.dirtyRev or null;
}
