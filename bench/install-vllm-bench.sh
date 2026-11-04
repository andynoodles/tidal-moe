#!/usr/bin/env bash
# Installs the vllm-bench client (Rust, static binary) into bench/bin/. It was used for every number in the README.
# The standalone repo is archived; its last release binaries still work. Upstream now lives in vllm-project/vllm (rust/).
set -euo pipefail
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bin
mkdir -p "$D"
curl -fsSL "https://github.com/vllm-project/vllm-bench/releases/latest/download/vllm-bench-$(uname -m)-linux-musl" -o "$D/vllm-bench"
chmod +x "$D/vllm-bench"
"$D/vllm-bench" --version
