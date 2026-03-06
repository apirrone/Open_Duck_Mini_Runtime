# Python package derivation for open-duck-mini-runtime using uv2nix.
# Builds a virtualenv with all pinned dependencies from uv.lock.
#
# Note on Pi 5 GPIO: uv.lock pins RPi.GPIO which doesn't support Pi 5.
# After flashing a Pi 5 image, run:
#   uv pip uninstall RPi.GPIO && uv pip install lgpio
# Or add lgpio to the project's dependencies and regenerate uv.lock.
{
  pkgs,
  uv2nix,
  pyproject-nix,
  pyproject-build-systems,
}: let
  lib = pkgs.lib;

  workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ../.;};

  overlay = workspace.mkPyprojectOverlay {
    # Prefer pre-built wheels from PyPI.
    # onnxruntime, pygame, and opencv all ship aarch64-linux wheels.
    sourcePreference = "wheel";
  };

  python = pkgs.python311;

  pythonBase = pkgs.callPackage pyproject-nix.build.packages {inherit python;};

  # Several Pi-specific packages and the git-sourced pypot omit `setuptools`
  # from build-system.requires even though they use setuptools.build_meta.
  # Inject it for each offender so source builds don't fail.
  buildSystemOverrides = final: prev: let
    addSetuptools = pkg:
      pkg.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [final.setuptools];
      });
  in {
    pypot = addSetuptools prev.pypot; # git source, no wheel
    rpi-gpio = addSetuptools prev.rpi-gpio; # sdist only, C extension
    rpi-ws281x = addSetuptools prev.rpi-ws281x; # sdist only, C extension
    wget = addSetuptools prev.wget; # sdist only, old package (2015)

    # adafruit-blinka's wheel includes pre-built binaries for Amlogic chips
    # (meson_g12_common, a311d) that link libgpiod.so.2.  The bcm283x (RPi)
    # binaries in the same wheel are statically linked and work fine.
    # Tell autopatchelf to skip the unsatisfied libgpiod dep for those
    # irrelevant platform blobs.
    "adafruit-blinka" = prev."adafruit-blinka".overrideAttrs (old: {
      autoPatchelfIgnoreMissingDeps =
        (old.autoPatchelfIgnoreMissingDeps or []) ++ ["libgpiod.so.2"];
    });
  };

  # Full Python package set: build backends + all packages from uv.lock + fixes
  pythonSet = pythonBase.overrideScope (
    lib.composeManyExtensions [
      pyproject-build-systems.overlays.wheel
      overlay
      buildSystemOverrides
    ]
  );
in
  pythonSet.mkVirtualEnv "open-duck-mini-runtime-env" workspace.deps.default
