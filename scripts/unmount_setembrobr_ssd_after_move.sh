#!/usr/bin/env bash

set -Eeuo pipefail

readonly MOVE_UNIT="setembrobr-ssd-move.service"
readonly MOUNT_POINT="/mnt/setembrobr-archive"

if [[ $EUID -ne 0 ]]; then
  echo "This script must run as root." >&2
  exit 1
fi

user_systemctl() {
  runuser -u aluisioamorim -- \
    env \
      XDG_RUNTIME_DIR=/run/user/1000 \
      DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
      systemctl --user "$@"
}

while user_systemctl is-active --quiet "$MOVE_UNIT"; do
  sleep 30
done

move_status=$(user_systemctl show "$MOVE_UNIT" --property=ExecMainStatus --value)
if [[ $move_status != "0" ]]; then
  printf '%s MOVE_SERVICE_FAILED exit_code=%s ssd_left_mounted=true\n' \
    "$(date --iso-8601=seconds)" \
    "$move_status"
  exit 1
fi

sync
if mountpoint -q "$MOUNT_POINT"; then
  umount "$MOUNT_POINT"
fi
printf '%s SSD_UNMOUNTED_AFTER_VERIFIED_MOVE\n' "$(date --iso-8601=seconds)"
