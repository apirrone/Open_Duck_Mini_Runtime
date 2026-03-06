#!/bin/sh

set -eu

usage() {
    echo "Usage: switch [USER] PROFILE HOST"
    echo "       switch PROFILE HOST (uses SSH default user)"
    echo ""
    echo "Profiles:"
    echo "  duck-pi4        Raspberry Pi 4"
    echo "  duck-pi5        Raspberry Pi 5"
    echo "  duck-pi-zero2w  Raspberry Pi Zero 2W"
    echo ""
    echo "Examples:"
    echo "  ./switch.sh wyant duck-pi4 192.168.50.151"
    echo "  ./switch.sh duck-pi4 duck-pi4.local"
    echo ""
    echo "Warning: Your ~/.ssh/config must be set up with the correct key for the target host!"
}

if [ "${1-}" = "--help" ]; then
    usage
    exit 0
fi

case $# in
    3)
        user=$1
        profile=$2
        host=$3
        ;;
    2)
        user=""
        profile=$1
        host=$2
        ;;
    *)
        usage
        exit 1
        ;;
esac

target_host=${user:+$user@}$host

nix run .#nixos-rebuild -- switch \
    --flake ".#$profile" \
    --target-host "$target_host" \
    --use-remote-sudo \
    --fast
