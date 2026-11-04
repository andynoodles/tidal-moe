<h1 align="center">TidalMoE</h1>

<p align="center"><a href="report/tidalmoe.pdf">Technical Report</a></p>

TidalMoE serves `deepseek-ai/DeepSeek-V4.1-Flash-1017` (552B backbone + 196B Engram parameters, 1M context) on one
NVIDIA GB300 desktop superchip, **2.4–4.3× faster than stock vLLM** on the same hardware, image and model.

It is a [vLLM general plugin](plugin/moe_placement) plus a launch script for the unmodified official
`vllm/vllm-openai` v0.33.0 image. No vLLM source changes, no rebuild.

## Results

Output tokens/s from [vllm-bench](https://github.com/vllm-project/vllm-bench): ShareGPT prompts, 1024 output tokens,
temperature 0, warm server. Each step was A/B-tested against the previous one with interleaved runs.

| Configuration | conc 1 | conc 8 | conc 32 | Chinese conc 32 |
|---|---:|---:|---:|---:|
| Stock vLLM v0.33.0 (`serve-stock.sh`) | 124 | 282 | 432 | 419 |
| **TidalMoE (`serve.sh`)** | **293** | **982** | **1862** | **1821** |
| Speed-up | 2.4× | 3.5× | 4.3× | 4.3× |

Prefill logprobs are bit-identical to the unmodified model, and the 1M-token context is preserved.

## How it works

The routed experts (~253 GiB in MXFP4) do not fit in HBM. Stock vLLM offloads whole MoE layers and is bound by
host-memory reads. TidalMoE instead uses:

1. **Per-expert hot/cold placement.** Each expert weight tensor is one CUDA virtual-address range whose pages live in
   HBM (hot experts) or pinned Grace LPDDR5X (cold experts). Experts are permuted together with their routing biases,
   so routing is unchanged.
2. **Adaptive expert cache.** Hot and cold experts are swapped online, gated by a significance test on usage counts.
   Swaps happen in place, so CUDA graphs stay valid.
3. **Kernel-aware layout.** Cold experts sit in evenly spread groups of 8; per-layer host budgets come from a convex
   cost model.
4. **Bit-exact forced MoE tactics.** Deeper TRT-LLM pipelines for the calls that still use TRT-LLM.
5. **HBM reclaimed for experts.** GPU KV is sized to exactly 1M tokens with a 32 GiB CPU KV tier behind it, and the
   token embedding moves to host memory.
6. **Host-aware persistent MoE kernels.** In one launch, a few CTAs stream host-resident experts over NVLink-C2C
   (~378 GB/s) while the rest stream HBM (~6 TB/s).
7. **Calibrated DSpark adaptive verification.** A cost model measured on GB300 replaces vLLM's built-in profile, which
   leaves out the MoE layers.

## Requirements

- **Hardware:** NVIDIA GB300 desktop superchip (Grace + B300 over NVLink-C2C). Tested on a Supermicro Super AI Station
  GB300 with 496 GB LPDDR5X.
- **OS and driver:** Linux 64K-page kernel (`7.0.0-1024-nvidia-64k`), driver 595.99.
- **Containers:** Docker with the NVIDIA Container Toolkit.
- **Image:** `vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e` (vLLM 0.33.0).
- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` in the Hugging Face cache (~476 GB).
- **Pinned host memory:** ~277 GiB. Keep NUMA node 0's `MemAvailable` above ~40 GiB.

## Quick start

```bash
git clone https://github.com/andynoodles/tidal-moe && cd tidal-moe
export HF_HOME=/data0/huggingface            # where the HF cache lives (model read from $HF_HOME/hub)
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017

./serve.sh                                    # container "tidal-moe" on :8000
bench/wait-ready.sh tidal-moe                 # ~6.5 min; a first start on a new machine also autotunes FlashInfer (~1 h)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"Hello!"}]}'
```

- `DRY_RUN=1 ./serve.sh` prints the `docker run` command without starting the container.
- Extra arguments to `serve.sh` go to `vllm serve`.
- Keep `CACHE_DIR` (default `~/.cache/tidal-moe`) across starts; it holds the JIT and FlashInfer autotune caches.

### Main switches

| Variable | Default | Meaning |
|---|---|---|
| `NAME`, `PORT` | `tidal-moe`, `8000` | container name, host port |
| `HF_HUB_CACHE` / `HF_HOME`, `CACHE_DIR` | `~/.cache/huggingface/hub`, `~/.cache/tidal-moe` | model cache, JIT/autotune caches |
| `PLAN` | `plan-e207-g8.json` | placement plan in `plans/` |
| `MOE_PLACEMENT` | `apply` | `off` = vLLM's layer offload |
| `MOE_ADAPT` | `1` | adaptive expert cache (`MOE_ADAPT_*` tune it; `MOE_ADAPT_COUNT=sel` also learns from prefill) |
| `MOE_PERSIST` | `1` | host-aware persistent MoE kernels |
| `MOE_AV` | `1` | DSpark adaptive verification with the GB300 cost curve; `0` = static schedule |
| `MOE_TACTICS` | tuned table | forced TRT-LLM tactics; empty = autotuned |
| `MOE_HOST_PARAMS` | `embed_tokens\.weight$` | parameters moved to host memory; empty = none |
| `KV_BYTES`, `CPU_KV_BYTES` | 1M-token KV, 32 GiB | GPU KV size; CPU KV tier (0 = off) |

`serve.sh` documents every knob; `serve-stock.sh` is the exact stock baseline.

## Benchmarking and correctness

```bash
bench/install-vllm-bench.sh && bench/fetch-datasets.sh     # client binary + ShareGPT / Chinese prompt sets
bench/warmup.sh sharegpt && bench/bench.sh mytest 1,8,32    # against a running server
python3 bench/summarize.py mytest
bench/run-config.sh tidal ./serve.sh 1,8,32 32              # fresh server + warm-up + ShareGPT and Chinese benches
bench/measure-bw.sh tidal tidal-moe 8 32                    # HBM + LPDDR5X/C2C bandwidth (Grace PMUs, needs perf access)
correctness/check.sh mycheck tidal-moe                      # prefill bit-exactness, decode, long prompt, memory
```

- Run-to-run noise is ~±2.5% at conc 8/32. Interleave configurations (A B B A) before trusting a difference.
- `correctness/check.sh` needs an otherwise idle server: batching with other traffic changes numerics.
- To rebuild the placement plan from your own traffic, see [docs/REPLANNING.md](docs/REPLANNING.md).

## Caveats

- **Tied to this image.** The forced tactics, generated persistent kernels and adaptive-verification cost curve are
  specific to vLLM 0.33.0 / FlashInfer 0.7.2.post1. After an upgrade, start with `MOE_TACTICS= MOE_PERSIST=0 MOE_AV=0`
  and re-validate.
- **Decode output changes slightly.** The persistent kernels are deterministic but not bit-identical to TRT-LLM, so
  greedy outputs change at near-ties. Measured quality is unchanged.
- **Page cache can take GPU memory.** HBM appears as NUMA node 1. Bind host processes to node 0
  (`numactl --membind=0`), or page cache may spill into GPU memory.
- **One checkpoint.** Only the MXFP4 checkpoint was evaluated.

## Repository layout

```
serve.sh, serve-stock.sh   optimized and stock launchers
plugin/moe_placement/      the vLLM plugin (placement, adaptive cache, tactics, persistent kernels, AV)
plans/, data/              placement plans and the routing statistics they were built from
bench/, correctness/       benchmark harness and correctness checks
tools/                     routing capture, plan builders, page-cache eviction, persistent-kernel generator
report/                    technical report (LaTeX source, figures, PDF; report/build.sh rebuilds it)
docs/                      raw engineering notes: architecture measurements, Nsight analysis, full log
```

## License

Apache-2.0 ([LICENSE](LICENSE)). `plugin/moe_placement/persist_moe_cute.py` derives from vLLM (Apache-2.0) and
FlashInfer (BSD-3-Clause); see [NOTICE](NOTICE). Model weights and datasets are not included and have their own
licenses.
