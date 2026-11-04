# Shared settings for the bench scripts (sourced). Override any of these in the environment.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PORT=${PORT:-8000}
BASE_URL=${BASE_URL:-http://127.0.0.1:$PORT}
MODEL=${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
RESULTS_DIR=${RESULTS_DIR:-$REPO/results}
DATASETS_DIR=${DATASETS_DIR:-$REPO/datasets}
VLLM_BENCH=${VLLM_BENCH:-$(command -v vllm-bench || echo "$REPO/bench/bin/vllm-bench")}
# vllm-bench tokenizes locally: point it at the model snapshot in the HF cache.
TOKENIZER=${TOKENIZER:-$(ls -d "$HF_HUB_CACHE/models--${MODEL//\//--}/snapshots/"* 2>/dev/null | head -1)}

# dataset_path <sharegpt|zh|/path/to/file.json>
dataset_path() {
  case "$1" in
    sharegpt|sg) echo "$DATASETS_DIR/ShareGPT_V3_unfiltered_cleaned_split.json" ;;
    zh)          echo "$DATASETS_DIR/sharegpt_zh_bench.json" ;;
    *)           echo "$1" ;;
  esac
}

need_bench() {
  [ -x "$VLLM_BENCH" ] || { echo "vllm-bench not found ($VLLM_BENCH): run bench/install-vllm-bench.sh" >&2; exit 1; }
  [ -n "$TOKENIZER" ] || { echo "model snapshot not found under $HF_HUB_CACHE (set TOKENIZER=<dir>)" >&2; exit 1; }
}

# warmup <dataset> [num_prompts] [concurrency]: unmeasured traffic so JIT compiles and the adaptive expert cache settle
warmup() {
  need_bench
  "$VLLM_BENCH" --backend openai-chat --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --dataset-name sharegpt --dataset-path "$(dataset_path "$1")" --sharegpt-output-len 1024 --ignore-eos \
    --temperature 0 --seed 7 --num-prompts "${2:-96}" --max-concurrency "${3:-16}" --disable-tqdm
}
