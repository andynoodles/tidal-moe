#!/usr/bin/env bash
# One complete A/B data point, as used for the README tables: start a fresh server, wait until healthy, then
# ShareGPT warm-up + bench, then zh warm-up + bench. GPU / host-memory snapshots go to results/<label>-mem.txt.
#
#   bench/run-config.sh <label> <serve script> <sharegpt concs|-> <zh concs|-> [VAR=val ...] [-- extra vllm args]
#   bench/run-config.sh tidal  ./serve.sh       1,8,32 8,32
#   bench/run-config.sh stock  ./serve-stock.sh 1,8,32 8,32
#   bench/run-config.sh noav   ./serve.sh       1,8,32 -    MOE_AV=0
# Run-to-run noise is about +-2.5% at conc 8/32: alternate configs (A B B A) and repeat before trusting a difference.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); source "$REPO/bench/common.sh"
LABEL=$1; SERVE=$2; SGC=$3; ZHC=$4; shift 4
envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
NAME=${NAME:-tidal-bench}
mkdir -p "$RESULTS_DIR"; R=$RESULTS_DIR/$LABEL-mem.txt
snap() {
  echo "-- $1 $(date +%T)" >> "$R"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> "$R"
  awk '/MemFree|Active\(file\)|Inactive\(file\)|Shmem:/{printf "%s %.1fGiB ", $3, $4/1048576} END{print ""}' \
    /sys/devices/system/node/node0/meminfo >> "$R"
}
echo "== $LABEL: $SERVE env: ${envs[*]} args: $*" > "$R"
env NAME="$NAME" PORT="$PORT" "${envs[@]}" "$SERVE" "$@" >/dev/null
timeout 2400 "$REPO/bench/wait-ready.sh" "$NAME" | tee -a "$R"
docker logs "$NAME" 2>&1 | grep -E "KV cache size|Maximum concurrency|placed in|host_rebind|persist|adaptive-verification" \
  | cut -c1-300 >> "$R" || true
snap startup
if [ "$SGC" != - ]; then
  warmup sharegpt > "$RESULTS_DIR/$LABEL-sg-warmup.log" 2>&1; snap sg-warm
  DATASET=sharegpt "$REPO/bench/bench.sh" "$LABEL-sg" "$SGC" > "$RESULTS_DIR/$LABEL-sg.log" 2>&1; snap sg-bench
fi
if [ "$ZHC" != - ]; then
  warmup zh > "$RESULTS_DIR/$LABEL-zh-warmup.log" 2>&1; snap zh-warm
  DATASET=zh "$REPO/bench/bench.sh" "$LABEL-zh" "$ZHC" > "$RESULTS_DIR/$LABEL-zh.log" 2>&1; snap zh-bench
fi
docker logs "$NAME" > "$RESULTS_DIR/$LABEL-docker.log" 2>&1 || true
[ "$SGC" != - ] && python3 "$REPO/bench/summarize.py" "$LABEL-sg"
[ "$ZHC" != - ] && python3 "$REPO/bench/summarize.py" "$LABEL-zh"
echo "done $LABEL (server $NAME left running)"
