#!/usr/bin/env bash
# Throughput sweep against a running server, the workload used for every README number:
# vllm-bench, chat API, ShareGPT prompts, 1024 output tokens (ignore EOS), temperature 0, 4 x conc prompts per point.
#
#   bench/bench.sh <label> [concurrencies]          # default 1,2,4,8,16,32
#   DATASET=zh bench/bench.sh <label> 8,32          # DATASET = sharegpt (default) | zh | /path/to/sharegpt-format.json
#   python3 bench/summarize.py <label> [<label>...] # table of output tok/s, TPOT, TTFT
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); source "$REPO/bench/common.sh"
LABEL=$1; CONCS=${2:-1,2,4,8,16,32}
need_bench
mkdir -p "$RESULTS_DIR/$LABEL"
exec "$VLLM_BENCH" --backend openai-chat --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --dataset-name sharegpt --dataset-path "$(dataset_path "${DATASET:-sharegpt}")" --sharegpt-output-len 1024 --ignore-eos \
  --temperature "${TEMPERATURE:-0}" --seed 0 \
  --sweep-max-concurrency "$CONCS" --sweep-num-prompts-factor 4 --num-warmups 2 \
  --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir "$RESULTS_DIR/$LABEL" --label "$LABEL" --disable-tqdm
