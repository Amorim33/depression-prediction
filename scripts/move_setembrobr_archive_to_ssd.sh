#!/usr/bin/env bash

set -Eeuo pipefail

readonly DISK_BY_ID="/dev/disk/by-id/usb-SanDisk_Portable_SSD_323534304551343035333837-0:0"
readonly PARTITION_BY_ID="${DISK_BY_ID}-part1"
readonly EXPECTED_DISK="/dev/sda"
readonly EXPECTED_DISK_BYTES="1000204886016"
readonly EXPECTED_VOLUME_UUID="6A57-8F8E"
readonly EXPECTED_VOLUME_LABEL="SETEMBROBR"
readonly SOURCE="/home/aluisioamorim/codex-runs/setembrobr-v6-restricted-archive"
readonly MOUNT_POINT="/mnt/setembrobr-archive"
readonly DESTINATION="${MOUNT_POINT}/setembrobr-v6-restricted-archive"
readonly EXPECTED_FILE_COUNT="1036"
readonly EXPECTED_ARCHIVE_SUMMARY_SHA256="01e8bd39a388afb4ed90d9f826139d81dffe5ada8035146a9afe62b403f08be7"
readonly EXPECTED_MANIFEST_SHA256="d616e7a622d5bad606eea5a070b3eed46b897d18c3fb4df02c14e41669eeb449"
readonly EXPECTED_SHA256SUMS_SHA256="448ff2d060d830a12279726456302767865bfba50a34e7abd96693e3b00e19dd"

mounted_here=0

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

cleanup() {
  local status=$?
  sync || true
  if (( mounted_here == 1 )) && mountpoint -q "$MOUNT_POINT"; then
    umount "$MOUNT_POINT" || true
  fi
  if (( status == 0 )); then
    log "MOVE_COMPLETE source_removed=true ssd_unmount_deferred=true"
  else
    log "MOVE_FAILED exit_code=${status} source_retained=$([[ -d "$SOURCE" ]] && echo true || echo false)"
  fi
  exit "$status"
}
trap cleanup EXIT

resolved_disk=$(readlink -f "$DISK_BY_ID")
[[ $resolved_disk == "$EXPECTED_DISK" ]]
[[ $(lsblk -dn -o TRAN "$resolved_disk" | xargs) == "usb" ]]
[[ $(lsblk -bdn -o SIZE "$resolved_disk" | xargs) == "$EXPECTED_DISK_BYTES" ]]
[[ $(lsblk -dn -o SERIAL "$resolved_disk" | xargs) == "323534304551343035333837" ]]
[[ $(blkid -s TYPE -o value "$PARTITION_BY_ID") == "exfat" ]]
[[ $(blkid -s LABEL -o value "$PARTITION_BY_ID") == "$EXPECTED_VOLUME_LABEL" ]]
[[ $(blkid -s UUID -o value "$PARTITION_BY_ID") == "$EXPECTED_VOLUME_UUID" ]]

[[ $SOURCE == "/home/aluisioamorim/codex-runs/setembrobr-v6-restricted-archive" ]]
[[ $DESTINATION == "/mnt/setembrobr-archive/setembrobr-v6-restricted-archive" ]]
[[ -d $SOURCE ]]
[[ -f $SOURCE/archive-summary.json ]]
[[ -f $SOURCE/.archive-state/manifest.jsonl ]]
[[ -f $SOURCE/SHA256SUMS ]]

echo "${EXPECTED_ARCHIVE_SUMMARY_SHA256}  ${SOURCE}/archive-summary.json" | sha256sum -c --quiet -
echo "${EXPECTED_MANIFEST_SHA256}  ${SOURCE}/.archive-state/manifest.jsonl" | sha256sum -c --quiet -
echo "${EXPECTED_SHA256SUMS_SHA256}  ${SOURCE}/SHA256SUMS" | sha256sum -c --quiet -

mkdir -p "$MOUNT_POINT"
if mountpoint -q "$MOUNT_POINT"; then
  [[ $(findmnt -nro SOURCE "$MOUNT_POINT") == "/dev/sda1" ]]
  [[ $(findmnt -nro FSTYPE "$MOUNT_POINT") == "exfat" ]]
else
  if [[ $EUID -ne 0 ]]; then
    echo "The SSD must already be mounted when this script runs without root privileges." >&2
    exit 1
  fi
  mount -t exfat -o uid=1000,gid=1000,umask=077,noatime "$PARTITION_BY_ID" "$MOUNT_POINT"
  mounted_here=1
fi
[[ -w $MOUNT_POINT ]]

available_bytes=$(df -B1 --output=avail "$MOUNT_POINT" | tail -n 1 | xargs)
(( available_bytes > 266319069522 ))

log "COPY_START source=${SOURCE} destination=${DESTINATION} usb_link=$(lsusb -t | grep -F 'Mass Storage' | xargs)"
rsync \
  -rlt \
  --append-verify \
  --partial \
  --modify-window=1 \
  --no-perms \
  --no-owner \
  --no-group \
  --human-readable \
  --stats \
  "${SOURCE}/" \
  "${DESTINATION}/"
sync
log "COPY_COMPLETE"

echo "${EXPECTED_ARCHIVE_SUMMARY_SHA256}  ${DESTINATION}/archive-summary.json" | sha256sum -c --quiet -
echo "${EXPECTED_MANIFEST_SHA256}  ${DESTINATION}/.archive-state/manifest.jsonl" | sha256sum -c --quiet -
echo "${EXPECTED_SHA256SUMS_SHA256}  ${DESTINATION}/SHA256SUMS" | sha256sum -c --quiet -
[[ $(wc -l < "${DESTINATION}/.archive-state/manifest.jsonl") == "$EXPECTED_FILE_COUNT" ]]
[[ $(wc -l < "${DESTINATION}/SHA256SUMS") == "$EXPECTED_FILE_COUNT" ]]

log "SSD_HASH_VERIFICATION_START files=${EXPECTED_FILE_COUNT}"
(
  cd "$DESTINATION"
  sha256sum -c --quiet SHA256SUMS
)
log "SSD_HASH_VERIFICATION_COMPLETE files=${EXPECTED_FILE_COUNT}"

[[ -d $SOURCE ]]
[[ -f $SOURCE/archive-summary.json ]]
rm -rf --one-file-system "$SOURCE"
[[ ! -e $SOURCE ]]
sync
log "SOURCE_REMOVED path=${SOURCE}"
