#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SOURCE_PKL ARCHIVE_ROOT OUTPUT_DIR TEMPORARY_DIR" >&2
  exit 2
fi

source_pkl=$1
archive_root=$2
output_dir=$3
temporary_dir=$4
project_dir=$(cd "$(dirname "$0")/.." && pwd)
python_bin="$project_dir/.venv/bin/python"
status_dir="$output_dir/job"
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p "$status_dir" "$temporary_dir"

write_status() {
  local state=$1
  local exit_code=$2
  "$python_bin" - "$status_dir/status.json" "$state" "$exit_code" "$started_at" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, state, exit_code, started_at = sys.argv[1:]
payload = {
    "state": state,
    "exitCode": int(exit_code),
    "startedAt": started_at,
    "updatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

finish() {
  exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    write_status completed "$exit_code"
  else
    write_status failed "$exit_code"
  fi
  exit "$exit_code"
}
trap finish EXIT

write_status running 0
cd "$project_dir"
exec > >(tee -a "$status_dir/pipeline.log") 2>&1

make \
  PYTHON="$python_bin" \
  SOURCE_PKL="$source_pkl" \
  ARCHIVE_ROOT="$archive_root" \
  OUTPUT_DIR="$output_dir" \
  TEMPORARY_DIR="$temporary_dir" \
  run
