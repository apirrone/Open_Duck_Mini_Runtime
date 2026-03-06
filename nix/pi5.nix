# NixOS configuration for Raspberry Pi 5.
#
# Pi 5 GPIO note: RPi.GPIO (pinned in uv.lock) does not support Pi 5.
# To use GPIO features (foot contact sensors) on Pi 5, after first boot run:
#   uv pip uninstall RPi.GPIO && uv pip install lgpio
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
      raspberry-pi-5.base
      raspberry-pi-5.page-size-16k
      sd-image
    ]);

  networking.hostName = lib.mkForce "duck-pi5";

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
    "nvme"
  ];

  system.stateVersion = lib.mkForce "25.05";
  system.configurationRevision = inputs.self.rev or inputs.self.dirtyRev or null;
}
