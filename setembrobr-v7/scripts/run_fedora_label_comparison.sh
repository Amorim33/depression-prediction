#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 PYTHON SOURCE_PKL SPECIALIST_LABELS BASELINE_OUTPUT COMPARISON_OUTPUT" >&2
  exit 2
fi

python_bin="$1"
source_pkl="$2"
specialist_labels="$3"
baseline_output="$4"
comparison_output="$5"
config="label-comparison-config.json"
pipeline="scripts/setembrobr_v7_label_comparison.py"
common=(--config "$config" --baseline-output "$baseline_output" --output-dir "$comparison_output")

"$python_bin" "$pipeline" prepare "${common[@]}" \
  --source-pkl "$source_pkl" --specialist-labels "$specialist_labels"
"$python_bin" "$pipeline" train-oof "${common[@]}"
"$python_bin" "$pipeline" audit-oof "${common[@]}"
"$python_bin" "$pipeline" lock "${common[@]}"
"$python_bin" "$pipeline" fit-full "${common[@]}"
"$python_bin" "$pipeline" score-test "${common[@]}"
"$python_bin" "$pipeline" audit-test "${common[@]}"
"$python_bin" "$pipeline" evaluate "${common[@]}"
"$python_bin" "$pipeline" audit-final "${common[@]}" \
  --source-pkl "$source_pkl" --specialist-labels "$specialist_labels"
