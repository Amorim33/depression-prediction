#!/usr/bin/env bash
set -uo pipefail

run_dir=${1:?run directory is required}
python_bin=${2:?python path is required}
hf_home=${3:?Hugging Face cache path is required}
config=${4:-configs/setembrobr.seed42.raw-qwen3-anxiety-embeddings.json}

log_dir="$run_dir/logs"
repo_dir="$run_dir/repo"
mkdir -p "$log_dir"
rm -f "$log_dir/full-embed.exit"
printf 'running %s\n' "$(date --iso-8601=seconds)" > "$log_dir/full-embed.status"

cd "$repo_dir"
HF_HOME="$hf_home" \
TRANSFORMERS_CACHE="$hf_home" \
PYTHONUNBUFFERED=1 \
"$python_bin" scripts/raw_qwen3_embeddings_anxiety_setembrobr.py \
  --config "$config" \
  --mode embed \
  --dataset-dir ../data/anxiety_tweets \
  --output-dir ../artifacts \
  --device cuda > "$log_dir/full-embed.log" 2>&1
exit_code=$?

printf '%s\n' "$exit_code" > "$log_dir/full-embed.exit"
if [[ $exit_code -eq 0 ]]; then
  printf 'completed %s\n' "$(date --iso-8601=seconds)" > "$log_dir/full-embed.status"
else
  printf 'failed exit=%s %s\n' "$exit_code" "$(date --iso-8601=seconds)" > "$log_dir/full-embed.status"
fi
exit "$exit_code"
