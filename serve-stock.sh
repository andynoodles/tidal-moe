#!/usr/bin/env bash
# Stock baseline: the same vLLM v0.33.0 image and model with vLLM's own layer-wise UVA expert offload and no TidalMoE
# changes. This is the "stock" configuration in the README results (124 / 282 / 432 output tok/s at conc 1 / 8 / 32).
#
#   ./serve-stock.sh [extra 'vllm serve' args...]     # container $NAME (default tidal-moe-stock) on :$PORT (default 8000)
#   DRY_RUN=1 ./serve-stock.sh                        # print the docker command only
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NAME=${NAME:-tidal-moe-stock}
PORT=${PORT:-8000}
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e}  # v0.33.0
MODEL=${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/tidal-moe}
HOST_NODE=${HOST_NODE:-0}

ARGS=(
  --tokenizer-mode deepseek_v41
  --offload-backend uva
  --cpu-offload-gb 60                          # stock: whole MoE layers' expert weights live in host memory
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight
  --engram-config '{"cpu_offload":true}'
  --gpu-memory-utilization "${GPU_UTIL:-0.96}"
  --max-model-len 1048576
  --max-num-batched-tokens 8192
  --long-prefill-token-threshold 6144
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,6,8,12,16,18,24,32,40,48,64,96,128]}'
  --tensor-parallel-size 1
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41"}'
  --kv-cache-dtype fp8_ds_mla
  --max-num-seqs "${MAX_SEQS:-24}"
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v41
  --mm-encoder-tp-mode data
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,24,1]]}'
  --host 0.0.0.0 --port 8000
)
DOCKER=(docker run -d --name "$NAME" --gpus all --ipc=host --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864
  -p "$PORT:8000"
  -v "$HF_HUB_CACHE:/root/.cache/huggingface/hub" -v "$HOME/.cache/vllm:/root/.cache/vllm"
  -v "$CACHE_DIR/.triton:/root/.triton" -v "$CACHE_DIR/.tilelang:/root/.tilelang"
  -v "$CACHE_DIR/.nv:/root/.nv" -v "$CACHE_DIR/.config:/root/.config" -v "$CACHE_DIR/.humming:/root/.humming"
  -v "$CACHE_DIR/flashinfer:/root/.cache/flashinfer"
  -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} -e VLLM_ENGINE_READY_TIMEOUT_S=14400 -e VLLM_USE_RUST_FRONTEND=1
  --entrypoint numactl "$IMAGE" --membind="$HOST_NODE" vllm serve "$MODEL" "${ARGS[@]}" "$@")

if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "${DOCKER[@]}"; echo; exit 0; fi
if docker ps --format '{{.Names}} {{.Ports}}' | grep -v "^$NAME " | grep -q ":$PORT->"; then
  echo "another container already publishes port $PORT; stop it first (docker ps)" >&2; exit 1
fi
mkdir -p "$CACHE_DIR"/{.triton,.tilelang,.nv,.config,.humming,flashinfer} "$HOME/.cache/vllm"
python3 "$REPO/tools/evict-cache.py" "$HF_HUB_CACHE" "$CACHE_DIR" >/dev/null || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
"${DOCKER[@]}"
echo "started $NAME on :$PORT (stock vLLM layer offload); wait: bench/wait-ready.sh $NAME"
