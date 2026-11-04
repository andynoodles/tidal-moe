#!/usr/bin/env bash
# TidalMoE: optimized DeepSeek-V4.1-Flash-1017 server on a single NVIDIA GB300 (Grace + B300), vLLM v0.33.0 in docker.
#
#   ./serve.sh [extra 'vllm serve' args...]          # starts container $NAME (default tidal-moe) on :$PORT (default 8000)
#   DRY_RUN=1 ./serve.sh                             # print the docker command instead of running it
#   curl -s localhost:8000/health                    # ready after ~6.5 min (first start: + FlashInfer autotune, see README)
#
# Everything below the knobs reproduces the configuration that was benchmarked (see README.md "Results").
# Feature switches (all default on):
#   MOE_PLACEMENT=apply   per-expert hot/cold placement (hot experts in HBM, cold ones in pinned Grace LPDDR5X)
#   MOE_ADAPT=1           adaptive expert cache (online hot/cold swaps every 3 s)
#   MOE_PERSIST=1         host-aware persistent MoE kernels for decode-size batches (<= 128 tokens)
#   MOE_AV=1              DSpark adaptive verification with the cost curve measured on GB300
#   MOE_TACTICS=...       forced TRT-LLM MoE tactics for the remaining TRT-LLM calls (empty = autotuned)
#   MOE_HOST_PARAMS=...   token embedding on host memory (empty = off)
#   KV_BYTES / CPU_KV_BYTES  GPU KV sized to exactly 1M context + a CPU KV tier (0 = no CPU tier)
# Set MOE_PLACEMENT=off MOE_PERSIST=0 MOE_AV=0 MOE_TACTICS= MOE_HOST_PARAMS= KV_BYTES= CPU_KV_BYTES=0 to get (almost)
# the stock behaviour; ./serve-stock.sh is the exact stock baseline used in the benchmarks.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- environment (override as needed) -------------------------------------------------------------------------
NAME=${NAME:-tidal-moe}
PORT=${PORT:-8000}
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e}  # v0.33.0
MODEL=${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash-1017}
HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}   # host HF cache holding the model
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/tidal-moe}    # JIT / autotune caches (reuse it: FlashInfer autotune takes ~1 h)
HOST_NODE=${HOST_NODE:-0}                         # Grace LPDDR5X NUMA node (the B300 HBM is exposed as node 1)
PLAN=${PLAN:-plan-e207-g8.json}                   # placement plan under plans/

# ---- optimization knobs (defaults = benchmarked configuration) ------------------------------------------------
HOST_PARAMS_DEFAULT='embed_tokens\.weight$'
TACTICS_DEFAULT=8=8:437,16=16:388,32=8:71,64=32:119   # FlashInfer 0.7.2.post1 indices; re-check after image changes
KV_BYTES=${KV_BYTES-2254857830}                   # 1,085,568 tokens = 1M context; empty = size KV from GPU_UTIL
CPU_KV_BYTES=${CPU_KV_BYTES:-34359738368}         # 32 GiB pinned CPU KV tier (SimpleCPUOffloadConnector); 0 = off
MOE_AV=${MOE_AV:-1}
if [ "$MOE_AV" = 1 ]; then
  SPEC_DEFAULT='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,32,5]],"enable_adaptive_verification":true}'
  # AV cost model = draft(requests) + verify(target tokens), measured on GB300 with the persistent kernels.
  export MOE_AV_CURVE=${MOE_AV_CURVE-builtin}
  export MOE_AV_DRAFT_MS=${MOE_AV_DRAFT_MS-1:0.1,2:0.2,3:0.3,4:0.4,5:0.5,6:0.6,7:0.7,8:0.8,9:0.9,10:1.0,11:1.1,12:1.2,13:1.3,14:1.4,15:1.5,16:1.6,17:1.7,18:1.8,19:1.9,20:2.0,21:2.1,22:2.2,23:2.3,24:2.4,25:2.5,26:2.6,27:2.7,28:2.8,29:2.9,30:3.0,31:3.1,32:3.2}
  export MOE_AV_VERIFY=${MOE_AV_VERIFY-1:7.3,2:7.6,4:8.8,6:9.2,8:11.2,12:12.7,16:14.0,18:15.2,24:17.0,32:19.1,40:21.5,48:23.2,64:26.4,96:32.9,128:39.0,192:150,256:160,1024:170,8192:400}
  export MOE_AV_MIN_REQS=${MOE_AV_MIN_REQS-2}
  export MOE_AV_UNIFORM_CG=${MOE_AV_UNIFORM_CG-1}
else
  SPEC_DEFAULT='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,8,3],[9,16,2],[17,32,1]]}'
fi
SPEC=${SPEC:-$SPEC_DEFAULT}

ARGS=(
  --tokenizer-mode deepseek_v41
  --offload-backend uva
  --cpu-offload-gb 60                          # load-time offload only; the plugin then re-places experts per plan
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight
  --engram-config '{"cpu_offload":true}'
  --gpu-memory-utilization "${GPU_UTIL:-0.98}"
  --max-model-len 1048576
  --max-num-batched-tokens 8192
  --long-prefill-token-threshold 6144
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,6,8,12,16,18,24,32,40,48,64,96,128]}'
  --tensor-parallel-size 1
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41"}'
  --kv-cache-dtype fp8_ds_mla
  --max-num-seqs "${MAX_SEQS:-32}"
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v41
  --mm-encoder-tp-mode data
  --speculative-config "$SPEC"
  --host 0.0.0.0 --port 8000
)
[ -n "$KV_BYTES" ] && ARGS+=(--kv-cache-memory-bytes "$KV_BYTES")
[ "$CPU_KV_BYTES" != 0 ] && ARGS+=(--kv-transfer-config "{\"kv_connector\":\"SimpleCPUOffloadConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use_per_rank\":$CPU_KV_BYTES,\"lazy_offload\":${CPU_KV_LAZY:-false}}}")

PXENV=(); for v in $(compgen -e | grep -E "^MOE_(PERSIST|AV|STEP)_" || true); do PXENV+=(-e "$v=${!v}"); done
DOCKER=(docker run -d --name "$NAME" --gpus all --ipc=host --cap-add SYS_NICE --ulimit memlock=-1 --ulimit stack=67108864
  -p "$PORT:8000"
  -v "$HF_HUB_CACHE:/root/.cache/huggingface/hub" -v "$HOME/.cache/vllm:/root/.cache/vllm"
  -v "$CACHE_DIR/.triton:/root/.triton" -v "$CACHE_DIR/.tilelang:/root/.tilelang"
  -v "$CACHE_DIR/.nv:/root/.nv" -v "$CACHE_DIR/.config:/root/.config" -v "$CACHE_DIR/.humming:/root/.humming"
  -v "$CACHE_DIR/flashinfer:/root/.cache/flashinfer"
  -v "$REPO/plugin:/moe/plugin:ro" -v "$REPO/plans:/moe/data:ro"
  -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} -e VLLM_ENGINE_READY_TIMEOUT_S=14400 -e VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-1}
  -e PYTHONPATH=/moe/plugin -e "MOE_PLACEMENT=${MOE_PLACEMENT:-apply}" -e "MOE_PLACEMENT_PLAN=/moe/data/$PLAN"
  -e "MOE_ADAPT=${MOE_ADAPT:-1}" -e "MOE_ADAPT_POLICY=${MOE_ADAPT_POLICY:-z}" -e "MOE_ADAPT_EPOCH_S=${MOE_ADAPT_EPOCH_S:-3}"
  -e "MOE_ADAPT_SWAPS=${MOE_ADAPT_SWAPS:-64}" -e "MOE_ADAPT_HALFLIFE=${MOE_ADAPT_HALFLIFE:-10}"
  -e "MOE_ADAPT_MARGIN=${MOE_ADAPT_MARGIN:-0.25}" -e "MOE_ADAPT_Z=${MOE_ADAPT_Z:-3}" -e "MOE_BW_TRACE=${MOE_BW_TRACE:-0}"
  -e "MOE_ADAPT_COUNT=${MOE_ADAPT_COUNT:-touch}" -e "MOE_ADAPT_FUSED=${MOE_ADAPT_FUSED:-1}"
  -e "MOE_PERSIST=${MOE_PERSIST:-1}" "${PXENV[@]}"
  -e "MOE_HOST_PARAMS=${MOE_HOST_PARAMS-$HOST_PARAMS_DEFAULT}" -e "MOE_TACTICS=${MOE_TACTICS-$TACTICS_DEFAULT}"
  --entrypoint numactl "$IMAGE" --membind="$HOST_NODE" vllm serve "$MODEL" "${ARGS[@]}" "$@")

if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "${DOCKER[@]}"; echo; exit 0; fi

# Only one server of this size fits on the GPU.
if docker ps --format '{{.Names}} {{.Ports}}' | grep -v "^$NAME " | grep -q ":$PORT->"; then
  echo "another container already publishes port $PORT; stop it first (docker ps)" >&2; exit 1
fi
mkdir -p "$CACHE_DIR"/{.triton,.tilelang,.nv,.config,.humming,flashinfer} "$HOME/.cache/vllm"
# On GB300 the HBM is NUMA node 1: page cache that spills onto it steals GPU memory. Drop it before starting.
python3 "$REPO/tools/evict-cache.py" "$HF_HUB_CACHE" "$CACHE_DIR" >/dev/null || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
"${DOCKER[@]}"
echo "started $NAME on :$PORT (plan $PLAN, MOE_PERSIST=${MOE_PERSIST:-1}, MOE_AV=$MOE_AV, MOE_ADAPT=${MOE_ADAPT:-1}," \
     "KV ${KV_BYTES:-from GPU_UTIL} + CPU tier $CPU_KV_BYTES B); wait: bench/wait-ready.sh $NAME"
