#!/usr/bin/env bash
# Correctness + memory probe against a running server (run while nothing else is sending requests):
#   1. teacher-forced prefill logprobs vs tf_final.json (reference from the unmodified model: must be bit-identical)
#   2. greedy decode vs ref-identity.json (decode differs at near-ties; see README "Correctness")
#   3. a ~200k-token prompt (memory headroom), then GPU / NUMA-node-0 memory and /health
#   correctness/check.sh <label> [container]
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); source "$REPO/bench/common.sh"
L=$1; CT=${2:-tidal-moe}; C=$REPO/correctness
export BASE_URL MODEL
python3 "$C/tf_logprobs.py" compare "$C/tf_final.json" "$L"
python3 "$C/correctness.py" compare "$C/ref-identity.json" "$L" | tail -8
python3 "$C/longprompt.py" "${LONG_TOKENS:-200000}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
awk '/MemFree|Active\(file\)|Inactive\(file\)|Shmem:/{printf "%s %.1fGiB ", $3, $4/1048576} END{print ""}' /sys/devices/system/node/node0/meminfo
curl -s -o /dev/null -w "health %{http_code}\n" "$BASE_URL/health"
docker logs "$CT" 2>&1 | grep -Ei "out of memory|Traceback" | head -3
