#!/usr/bin/env bash
# Unmeasured warm-up traffic (96 prompts at conc 16, ~2.5 min): first-request JIT compiles and the adaptive expert
# cache adapting to the workload. Always warm up with the same kind of traffic before benchmarking.
#   bench/warmup.sh [sharegpt|zh|file]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); source "$REPO/bench/common.sh"
warmup "${1:-sharegpt}" "${NUM_PROMPTS:-96}" "${CONC:-16}"
