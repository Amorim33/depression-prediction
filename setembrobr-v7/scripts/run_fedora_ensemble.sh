#!/usr/bin/env bash
set -euo pipefail

python_bin=${1:?python path is required}
archive_root=${2:?archive root is required}
baseline_output=${3:?baseline output is required}
ensemble_output=${4:?ensemble output is required}
temporary_dir=${5:?temporary directory is required}
project_dir=$(cd "$(dirname "$0")/.." && pwd)
status_dir="$ensemble_output/job"
status_file="$status_dir/status.json"
log_file="$status_dir/run.log"

mkdir -p "$status_dir" "$temporary_dir"
cd "$project_dir"

write_status() {
  local state=$1
  local stage=$2
  local code=${3:-0}
  "$python_bin" - "$status_file" "$state" "$stage" "$code" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
payload = {"state": sys.argv[2], "stage": sys.argv[3], "exitCode": int(sys.argv[4])}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, path)
PY
}

stage=starting
trap 'code=$?; write_status failed "$stage" "$code"; exit "$code"' ERR

run_stage() {
  stage=$1
  shift
  write_status running "$stage" 0
  "$python_bin" scripts/setembrobr_v7_ensemble.py "$stage" \
    --config ensemble-config.json \
    --baseline-config config.json \
    --baseline-output "$baseline_output" \
    --archive-root "$archive_root" \
    --output-dir "$ensemble_output" \
    --temporary-dir "$temporary_dir" \
    --feature-helper shared/raw_ternary_prepare_setembrobr.py "$@"
}

exec > >(tee -a "$log_file") 2>&1
run_stage prepare-train
run_stage train-tabular-oof
run_stage train-sequence-oof
run_stage train-stack-oof
run_stage audit-oof
run_stage lock
run_stage prepare-test
run_stage score-test
run_stage audit-test
run_stage evaluate
stage=complete
write_status complete complete 0
