{
  nixConfig = {
    extra-substituters = "https://nixos-raspberrypi.cachix.org";
    extra-trusted-public-keys = "nixos-raspberrypi.cachix.org-1:4iMO9LXa8BqhU+Rpg6LQKiGa2lsNh/j2oiYLNOQ5sPI=";
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

    nixos-raspberrypi = {
      # Do NOT follow our nixpkgs here — nixos-raspberrypi is tested against
      # its own pinned nixpkgs. Forcing our nixos-25.05 onto it causes a
      # boot.loader.raspberryPi rename.nix conflict (same pattern as GSE).
      url = "github:nvmd/nixos-raspberrypi/main";
    };

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    nixos-raspberrypi,
    uv2nix,
    pyproject-nix,
    pyproject-build-systems,
    ...
  } @ inputs: let
    inherit (nixpkgs.lib) mapAttrs mapAttrs';

    piVariants = {
      duck-pi-zero2w = {
        modules = [./nix/pi-zero2w.nix];
      };
      duck-pi4 = {
        modules = [./nix/pi4.nix];
      };
      duck-pi5 = {
        modules = [./nix/pi5.nix];
      };
    };

    piConfigurations = mapAttrs (_: variant:
      nixos-raspberrypi.lib.nixosSystem {
        specialArgs = {inherit inputs self nixos-raspberrypi uv2nix pyproject-nix pyproject-build-systems;};
        modules = variant.modules;
      })
    piVariants;

    piSdPackages =
      mapAttrs' (name: cfg:
        nixpkgs.lib.nameValuePair "${name}-sd" cfg.config.system.build.sdImage)
      piConfigurations;
  in {
    nixosConfigurations = piConfigurations;

    # Expose the runtime as a NixOS module for use in other flakes
    nixosModules.open-duck-mini-runtime = import ./nix/module.nix;

    packages = {
      x86_64-linux =
        piSdPackages
        // {
          default = piSdPackages.duck-pi4-sd;
          nixos-rebuild = nixpkgs.legacyPackages.x86_64-linux.nixos-rebuild;
        };
      aarch64-linux =
        piSdPackages
        // {
          default = piSdPackages.duck-pi4-sd;
          nixos-rebuild = nixpkgs.legacyPackages.aarch64-linux.nixos-rebuild;
        };
    };

    # `nix run .#nixos-rebuild` — used by switch.sh
    apps = {
      x86_64-linux.nixos-rebuild = {
        type = "app";
        program = "${nixpkgs.legacyPackages.x86_64-linux.nixos-rebuild}/bin/nixos-rebuild";
      };
      aarch64-linux.nixos-rebuild = {
        type = "app";
        program = "${nixpkgs.legacyPackages.aarch64-linux.nixos-rebuild}/bin/nixos-rebuild";
      };
    };
  };
}
