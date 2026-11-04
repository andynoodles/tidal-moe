# TidalMoE

**Hot/cold expert tiering and host-aware MoE kernels for DeepSeek-V4.1-Flash on a single NVIDIA GB300.**

TidalMoE serves `deepseek-ai/DeepSeek-V4.1-Flash-1017` (552B backbone + 196B Engram parameters, 1M context) on one
GB300 Grace Blackwell Ultra desktop superchip (Grace CPU + B300 GPU). It runs **2.4–4.3× faster than stock vLLM** on
the same hardware, image and model.

The model's routed experts do not fit in the GPU's HBM. TidalMoE lets the hot experts live in HBM and the cold ones in
the Grace CPU's LPDDR5X, and keeps moving experts between the two tiers as traffic changes. On top of that it adds a
MoE kernel that reads both tiers at the same time and a speculative-decoding scheduler calibrated to this machine.

Everything is a [vLLM general plugin](plugin/moe_placement) plus a launch script on top of the unmodified official image
`vllm/vllm-openai` v0.33.0. No vLLM source changes and no rebuild are needed.

## Results

Output tokens/s, measured with [vllm-bench](https://github.com/vllm-project/vllm-bench): ShareGPT prompts, 1024 output
tokens, temperature 0, warm server. Each "→ next" step was A/B-tested against the previous one in the same session
(fresh server starts, alternating configs, at least 2 runs each).

| Configuration | conc 1 | conc 8 | conc 32 | zh conc 32 |
|---|---:|---:|---:|---:|
| Stock vLLM v0.33.0 (`serve-stock.sh`, layer-wise UVA offload) | 124 | 282 | 432 | 419 |
| + per-expert hot/cold placement, serving knobs | 216 | 627 | 931 | 644¹ |
| + adaptive expert cache | 219 | 735 | 1170 | 1150 |
| + cache refinements, HBM reclaimed, groups of 8, forced MoE tactics | 259 | 789 | 1313 | 1316 |
| + host-aware persistent MoE kernels | 302 | 935 | 1599 | 1617 |
| **+ DSpark adaptive verification (current `serve.sh`)** | **293²** | **982** | **1862** | **1821** |
| **Speed-up vs stock** | **2.4×** | **3.5×** | **4.3×** | **4.3×** |

¹ Chinese traffic with a static plan built without Chinese data. Hot expert sets are domain-specific, which is why the
adaptive cache matters.
² Conc 1 runs the same code path as the row above; the difference is run-to-run noise.

More points from the final verification, current configuration vs the previous row:

| ShareGPT conc 2 | conc 4 | conc 16 | sampled (T=0.7) conc 32 |
|---:|---:|---:|---:|
| 433 (+1.8%) | 653 (+4.7%) | 1334 (+9.9%) | 1830 (+13.1%) |

- Median TPOT at conc 32 is ~15.5 ms (18.9 ms without adaptive verification).
- 1M-token context still fits: GPU KV is sized for exactly one 1,048,576-token request, plus a 32 GiB CPU KV tier.
- Prefill logprobs are bit-identical to the unmodified model. See [Correctness](#correctness) for decode.

## The problem

| | |
|---|---|
| GPU | B300: 152 SMs @ 2.07 GHz, 115.5 MiB L2, HBM3e ~249.5 GiB usable, 7.16 TB/s |
| CPU | Grace: 72 Neoverse V2 cores, ~496 GB LPDDR5X, 396 GB/s spec (360 GB/s per firmware) |
| Link | NVLink-C2C, 10 × 44.7 = 447 GB/s per direction, cache-coherent |
| Model | 40 layers (20-layer causal encoder + 20-layer decoder, CED), MoE in every layer: 1 shared + 384 routed experts, top-6, MXFP4 |
| Routed experts | 40 × 384 × 16.9 MiB ≈ **253 GiB**: more than the whole HBM |
| Engram | 196B FP8 parameters (~189 GiB), kept in pinned host memory by vLLM |

Stock vLLM offloads whole MoE layers (layers 0–8) to host memory and reads them over NVLink-C2C, so ~24% of every
token's expert reads come from LPDDR5X. Measured with the Grace uncore PMUs at conc 32:
- GPU reads of host memory plateau at ~258 GB/s;
- SMs are 92% busy, mostly stalled on those reads;
- HBM sits at ~15% activity.

Stock is bound by host-memory reads, not by compute or HBM.

Measured facts about this machine that shaped the design (see [docs/GB300-ARCHITECTURE.md](docs/GB300-ARCHITECTURE.md)):
- **The ceiling is LPDDR5X, not the link.** With enough bytes in flight, GPU reads of pinned host memory
  (`cuMemCreate` with `HOST_NUMA`) reach 383–390 GB/s.
- **A few SMs saturate host bandwidth.** One SM tops out at ~51 GB/s through TMA, so 8–16 SMs are enough.
- **Host reads are slow and not cached.** Latency is ~650–700 ns versus ~140 ns for HBM, and the GPU does not cache
  host memory across kernels.
- **Host and HBM reads can run fully concurrently on different SMs.** A microbenchmark read host memory at 389 GB/s
  on 16 SMs while the other 136 SMs read HBM at 6.62 TB/s.
- **The HBM is exposed to Linux as NUMA node 1.** Host processes must be bound to node 0, or page cache spills into
  the GPU's memory.

**Routing is skewed.** The top 20% of experts take 60–75% of decode routing; the coldest 24% take only ~3.5%. This is
an "80/20 rule" for experts, and it holds per workload. But the hot sets are domain-specific: English chat, Chinese
and code overlap with a Jaccard index of only 0.2–0.38.

## What TidalMoE does

### 1. Per-expert hot/cold placement (`MOE_PLACEMENT=apply`)
After vLLM's MoE weight post-processing, the plugin permutes each layer's 384 experts so the hot ones come first.
- Gate rows, routing biases (text and image), scales and clamp limits are permuted identically, so routing decisions
  are unchanged.
- `w13`/`w2` are re-backed by one CUDA VMM virtual-address range per tensor. Hot experts sit on HBM pages, cold experts
  on pinned LPDDR5X pages (`cuMemCreate` with `HOST_NUMA`, NUMA node 0), at 2 MiB granularity.
- The MoE kernel sees an ordinary CUDA tensor.
- About 3,300 cold experts (54 GiB) live on the host, chosen per layer from measured routing statistics (`plans/`).

### 2. Adaptive expert cache (`MOE_ADAPT=1`)
The plan is only a starting point; traffic changes the hot set.
- A fused Triton kernel inside each decode-size MoE call counts the distinct experts touched.
- Every 3 s the engine thread updates an EMA of these counts (half-life 10 epochs).
- It then swaps the hottest host-resident experts with the coldest HBM-resident ones, at most 64 swaps per epoch, most
  beneficial first. A swap needs both a >25% margin and >3σ Poisson significance.
- A swap exchanges the two slots' rows in every per-expert tensor. Addresses never change, so CUDA graphs stay valid,
  and routing is unchanged.
- Ideas from DALI, TriMoE, HybriMoE and Fate (see [docs/EXPERT-CACHE-ANALYSIS.md](docs/EXPERT-CACHE-ANALYSIS.md)).
- It recovers a mismatched plan within minutes: Chinese traffic on a plan built without Chinese data went from 644 to
  1150 tok/s at conc 32.

### 3. Layout and budgets that suit the kernel
- **Cold experts in groups of 8, spread evenly across each layer.** The TRT-LLM MoE kernel launches CTAs in expert-slot
  order, so clustered cold experts tie up many SMs on host reads while HBM idles. Groups of 8 gave +3–6% at conc 32.
- **Per-layer host budgets from a convex step-time model** (`tools/plan_convex.py`). Nsight showed a layer costs about
  `max(t_hbm, 64 µs × n_cold)`, so host budget moves from layers that are host-bound to layers with slack.

### 4. Forced TRT-LLM MoE tactics (`MOE_TACTICS`)
FlashInfer's autotuner, which ran on synthetic uniform routing, picked shallow 3-stage pipelines. TidalMoE forces
4–6-stage tactics (or a 512-wide K tile) per token bucket. These keep more bytes in flight, which helps both small-batch
HBM latency and host reads. Every forced tactic was chosen from an exhaustive sweep to be **bit-identical** to the
autotuned one. The tactics apply to the calls that still use TRT-LLM: prefill chunks and the DSpark drafter.

### 5. HBM reclaimed for hot experts
- GPU KV is sized to exactly 1M context (`--kv-cache-memory-bytes 2254857830` = 1,085,568 tokens), plus a 32 GiB CPU KV
  tier (`SimpleCPUOffloadConnector`) for prefix reuse beyond that.
- The 1.23 GiB token embedding, which is gather-only, moves to pinned host memory (`MOE_HOST_PARAMS`).
- The freed HBM holds more hot experts.

### 6. Host-aware persistent MoE kernels (`MOE_PERSIST=1`): the biggest kernel-level win
Every routed-expert MoE call of up to 128 tokens (decode, speculative verification, short prefill chunks) runs as:
1. a Triton front that builds a routing record (cold tiles first) and gathers MXFP8 activations;
2. a persistent GEMM1 and a persistent GEMM2 (CuTe DSL, one CTA per SM, 6/10-stage TMA pipelines);
3. a Triton finalize.

The kernels are chained with PDL and captured in CUDA graphs. Inside each GEMM launch, CTAs `[0, kc)` (4–16 of them,
chosen per call from a bandwidth model) stream all host-resident experts over NVLink-C2C while the other CTAs stream
HBM. This reproduces the hardware's concurrent-read capability inside one launch.

Nsight Compute, one MoE layer at 64 tokens with 12 cold experts:

| | GEMM1 | host read | HBM read |
|---|---:|---:|---:|
| TRT-LLM (stock kernel) | 534 µs | 265 GB/s | 4.10 TB/s |
| TidalMoE persistent | 375 µs | 378 GB/s | 5.97 TB/s |

A whole MoE layer takes 24–30% less time with 4–18 cold experts.

The kernels are generated from vLLM's own CuTe-DSL routed-expert GEMMs, which in the image only support 4 GPUs (TP4)
and ≤8 rows. They are extended to a single GPU and up to 128 rows (`tools/persist/make_persist_cute.py`), so they read
exactly the weight layout vLLM prepares for TRT-LLM. Design: [docs/persistent-moe/DESIGN.md](docs/persistent-moe/DESIGN.md).

### 7. DSpark adaptive verification calibrated to this machine (`MOE_AV=1`)
vLLM's DSpark speculative decoding can size each request's verification length from the draft confidence head and an
engine cost curve (`enable_adaptive_verification`).
- **The problem:** vLLM's built-in cost profile runs padding tokens that route to no expert, so it misses the MoE cost.
  On this machine every extra verified token may pull more cold experts from host memory.
- **The fix:** the plugin replaces the curve with step times measured on GB300 with the persistent kernels. It prunes
  drafts only when at least 2 requests run, and keeps the uniform graph at 1 request.
- **Result:** +5.6% at conc 8 and +13.9% at conc 32. Acceptance length at conc 32 rose from 1.77 to 2.49.
- Details: [docs/ADAPTIVE-VERIFICATION.md](docs/ADAPTIVE-VERIFICATION.md).

### 8. Serving settings
- `--max-num-seqs 32`.
- DSpark schedule `[[1,32,5]]` with adaptive verification (`MOE_AV=0` falls back to `[[1,4,5],[5,8,3],[9,16,2],[17,32,1]]`).
- All containers bound to NUMA node 0, with page-cache eviction before start.
- The FlashInfer autotune cache is persisted.

### Measured and rejected
| Idea | Outcome |
|---|---|
| Two concurrent TRT-LLM calls (hot / cold) on separate streams | Slower: the cold-only call read host memory at ~200 GB/s with a shallow pipeline |
| SM partitioning with CUDA green contexts | Loses at ≤12 cold experts per layer, and is not bit-identical |
| Computing cold experts on the Grace CPU | 0.80 ms vs 0.61 ms for the GPU to read them; CPU DRAM traffic also halves GPU host bandwidth |
| Copy-engine prefetch of cold experts | Routing is known only at the layer; copy engines give 188–303 GB/s for 1–4 MB chunks |
| DeepGEMM Mega-MoE (DeepSeek's persistent MoE) | 3–14% slower with host-resident experts (its dynamic queue does not hide host tiles); +0.9 GiB of buffers |
| Mega-mHC for ≤16-token steps | No gain |
| Saving and restoring the adaptive-cache state across restarts | +2–3% right after a restart; most of the slump is first-request JIT. Off by default (`MOE_ADAPT_SAVE`/`_RESUME`) |
| CPU KV tier | Throughput-neutral in these benchmarks; kept for prefix reuse beyond 1M tokens |

The full engineering log, with every measurement and A/B table, is in [docs/OPERATIONS-LOG.md](docs/OPERATIONS-LOG.md).
Profiling notes are in [docs/NSIGHT-BOTTLENECKS.md](docs/NSIGHT-BOTTLENECKS.md), and a review of the DeepSeek-V4.1 tech
report against vLLM's implementation is in [docs/TECH-REPORT-LEADS.md](docs/TECH-REPORT-LEADS.md).

## Requirements
- **Hardware:** NVIDIA GB300 desktop superchip (Grace + B300 over NVLink-C2C). Tested on a Supermicro Super AI
  Station GB300 with 496 GB LPDDR5X.
- **OS and driver:** Linux with the NVIDIA 64K-page kernel (tested `7.0.0-1024-nvidia-64k`) and driver 595.99. In this
  driver mode the HBM shows up as NUMA node 1.
- **Containers:** Docker with the NVIDIA Container Toolkit, plus image
  `vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e` (vLLM 0.33.0,
  FlashInfer 0.7.2.post1, PyTorch 2.14 / CUDA 13.0).
- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` in the Hugging Face cache (~476 GB).
- **Pinned host memory:** about 277 GiB (Engram ~189 GiB, cold experts 54 GiB, embedding 1.2 GiB, CPU KV tier 32 GiB).
  There is no swap on the test machine, so keep NUMA node 0's `MemAvailable` above ~40 GiB.

## Quick start
```bash
git clone https://github.com/andynoodles/tidal-moe && cd tidal-moe
export HF_HOME=/data0/huggingface            # wherever the HF cache lives; the model is read from $HF_HOME/hub
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017

./serve.sh                                    # container "tidal-moe" on :8000
bench/wait-ready.sh tidal-moe                 # ~6.5 min; the first start on a new machine also autotunes FlashInfer (~1 h)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"Hello!"}]}'
```
- `CACHE_DIR` (default `~/.cache/tidal-moe`) holds the JIT and FlashInfer autotune caches. Reuse it across starts.
- `DRY_RUN=1 ./serve.sh` prints the full `docker run` command.
- Only one server of this size fits on the GPU. Both scripts refuse to start while another container publishes the same port.
- The server is OpenAI-compatible, with DeepSeek-V4.1 tool calling and reasoning parsers enabled.

### Configuration
| Variable | Default | Meaning |
|---|---|---|
| `NAME`, `PORT` | `tidal-moe`, `8000` | container name and host port |
| `HF_HUB_CACHE` / `HF_HOME` | `~/.cache/huggingface/hub` | host Hugging Face cache with the model |
| `CACHE_DIR` | `~/.cache/tidal-moe` | Triton / TileLang / FlashInfer caches |
| `PLAN` | `plan-e207-g8.json` | placement plan in `plans/` (`plan-b16.json` = older 16-expert blocks, 56 GiB on host) |
| `MOE_PLACEMENT` | `apply` | `off` = vLLM's own layer offload, `inspect` = log only |
| `MOE_ADAPT` | `1` | adaptive expert cache; `MOE_ADAPT_*` tune it (epoch 3 s, ≤64 swaps, half-life 10, margin 0.25, z 3) |
| `MOE_ADAPT_COUNT` | `touch` | `touch` = distinct experts per decode step; `sel` = selections incl. prefill (better for repeated huge prompts) |
| `MOE_PERSIST` | `1` | host-aware persistent MoE kernels; `MOE_PERSIST_MAX_TOKENS` (128), `MOE_PERSIST_LAYERS` (bisecting) |
| `MOE_AV` | `1` | DSpark adaptive verification with the GB300 cost curve (`MOE_AV_*` override the curve); `0` = static schedule |
| `MOE_TACTICS` | `8=8:437,16=16:388,32=8:71,64=32:119` | forced TRT-LLM tactics; empty = autotuned |
| `MOE_HOST_PARAMS` | `embed_tokens\.weight$` | parameters moved to host memory; empty = none |
| `KV_BYTES` | `2254857830` | GPU KV bytes (1M context); empty = size from `GPU_UTIL` (needs `PLAN=plan-b16.json`) |
| `CPU_KV_BYTES`, `CPU_KV_LAZY` | 32 GiB, `false` | CPU KV tier size (0 = off) and lazy offload |
| `MOE_BW_TRACE` | `0` | `1` = log expert bytes read from host/HBM per epoch |
| `SPEC` | (from `MOE_AV`) | full speculative-decoding JSON override |
| `IMAGE`, `MODEL`, `HOST_NODE`, `GPU_UTIL`, `MAX_SEQS` | see `serve.sh` | |

Any extra arguments to `serve.sh` are passed to `vllm serve`.

## Benchmarking
```bash
bench/install-vllm-bench.sh                   # static vllm-bench binary -> bench/bin/
bench/fetch-datasets.sh                       # ShareGPT + the 4000-prompt Chinese set -> datasets/

# against a running server
bench/warmup.sh sharegpt                      # always warm up first (JIT + adaptive cache)
bench/bench.sh mytest 1,8,32                  # -> results/mytest/*.json
DATASET=zh bench/bench.sh mytest-zh 8,32
python3 bench/summarize.py mytest             # output tok/s, TPOT, TTFT per concurrency

# full A/B points (fresh server, warm-up, ShareGPT + zh benches, memory snapshots)
bench/run-config.sh stock ./serve-stock.sh 1,8,32 32
bench/run-config.sh tidal ./serve.sh       1,8,32 32
bench/run-config.sh noav  ./serve.sh       1,8,32 32 MOE_AV=0

# bandwidth under load: HBM (GPM counter) + Grace LPDDR5X / NVLink-C2C (uncore PMUs, needs perf_event_paranoid <= 0)
bench/measure-bw.sh tidal tidal-moe 1 8 32
```
Run-to-run noise is about ±2.5% at conc 8/32. Alternate configurations (A B B A) and repeat before trusting a difference.

## Correctness
`correctness/check.sh <label> [container]` checks a running server:
- teacher-forced prefill logprobs against a reference recorded on the unmodified model;
- greedy decode;
- a ~200k-token prompt;
- memory headroom.

What we verified:
- **Placement, adaptive swaps, forced tactics, embedding move, KV sizing:** prefill logprobs are **bit-identical**
  (352/352 tokens), including after >1,100 live expert swaps.
- **Persistent kernels:**
  - Prefill stays bit-identical.
  - Decode is deterministic but not bit-identical to TRT-LLM. At ≤8 tokens TRT-LLM uses a split-K reduction; the new
    kernels match its non-split-K tactics. Greedy outputs therefore change at near-ties.
  - Quality statistics match TRT-LLM: 2.93% vs 3.01% of greedy tokens are not the teacher-forced argmax over 13.3k
    tokens, and the mean |Δlogprob| is 0.048 for both.
  - Long generations (up to 2048 tokens; English, Chinese, code, math) are coherent.
- **Adaptive verification** is lossless, as speculative decoding should be: 30 concurrent greedy requests re-scored
  teacher-forced gave 2.77% non-argmax tokens vs 3.57% without it.
- **Stability:**
  - 10-minute mixed load at conc 32 (ShareGPT + Chinese + 210k / 528k-token prompts): 2,528 requests, 0 failures.
  - A 1.04M-token prompt under conc-32 load: first token after 47 s, no errors.

## Re-planning for your traffic
The adaptive cache converges online, so re-planning only gives a better starting point after restarts:
1. Start a routing-capture server:
   ```bash
   MOE_PLACEMENT=off MOE_PERSIST=0 MOE_AV=0 VLLM_USE_RUST_FRONTEND=0 ./serve.sh --enable-return-routed-experts
   ```
   Returning routed experts needs the Python frontend.
2. Collect routing counts with `tools/collect_routing.py`. Edit its prompt lists first; it needs numpy, so run it
   inside the container:
   ```bash
   docker cp tools/collect_routing.py tidal-moe:/tmp/ && docker cp datasets/ShareGPT_V3_unfiltered_cleaned_split.json tidal-moe:/tmp/sg.json
   docker exec tidal-moe python3 /tmp/collect_routing.py /tmp/sg.json /tmp/routing.npz 200 384 16
   docker cp tidal-moe:/tmp/routing.npz data/routing-new.npz
   ```
3. Build the plan in the vLLM image:
   ```bash
   IMG=vllm/vllm-openai@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e
   docker run --rm --entrypoint python3 -v "$PWD:/w" $IMG /w/tools/plan_convex.py /w/data/routing-new.npz \
     /w/plans/plan-b16.json 207 /w/plans/plan-new.json sharegpt:0.5,zh:0.3,code:0.2 mix
   docker run --rm --entrypoint python3 -v "$PWD:/w" $IMG /w/tools/relayout.py /w/plans/plan-new.json /w/plans/plan-new-g8.json 8
   ```
   The `207` is the host budget in 16-expert-block units: 54.2 GiB of cold experts, which fits the default KV sizing.
4. Start with `PLAN=plan-new-g8.json ./serve.sh`.

## Caveats
- **Pinned to this image.**
  - The tactic indices, the generated persistent kernels and the AV cost curve are specific to vLLM 0.33.0 /
    FlashInfer 0.7.2.post1 and to this plan.
  - The plugin wraps vLLM internals: `process_weights_after_loading`, `TrtLlmMxfp4ExpertsModular.apply`,
    `GPUModelRunner.execute_model`, the adaptive-verification manager and the CUDA-graph manager.
  - After an image upgrade, start with `MOE_TACTICS= MOE_PERSIST=0 MOE_AV=0` and re-validate.
  - Regenerate the persistent kernels with `tools/persist/make_persist_cute.py` and re-calibrate AV
    (docs/ADAPTIVE-VERIFICATION.md).
- **Tight GPU memory under extreme load.** With huge prefills mixed with heavy decode, free HBM dipped to ~0.6 GiB; no
  OOM was seen. `MOE_AV=0` gives back ~1.3 GiB of peak.
- **Page cache on the HBM node.** If node-0 free memory falls below its watermark, unbound processes get page cache on
  node 1, which is GPU memory. Run large file copies with `numactl --membind=0`, or run
  `tools/evict-cache.py <dirs>` first.
- **Untested combinations:** structured output / guided decoding and tool-call streaming together with adaptive
  verification.
- **Model coverage:** only DeepSeek-V4.1-Flash-1017 (MXFP4 experts) is supported. The NVFP4 checkpoint was not evaluated.

## Repository layout
```
serve.sh              optimized launcher (docker run of the official vLLM image + plugin)
serve-stock.sh        stock baseline launcher used for the "stock" numbers
plugin/moe_placement/ vLLM general plugin: placement, adaptive cache, tactics, host params, persistent kernels, AV
plans/                placement plans (plan-e207-g8.json = default)
data/routing-v1.npz   routing statistics (ShareGPT / zh / code; prefill and decode) used to build the plans
bench/                vllm-bench install, dataset fetch, warm-up, sweeps, A/B runner, bandwidth measurement, summary
correctness/          prefill/decode reference checks, concurrent losslessness check, long-prompt probe
tools/                routing capture, plan builders, page-cache eviction, persistent-kernel generator
docs/                 architecture measurements, Nsight analysis, cache study, kernel design, full engineering log
```

## License
Apache-2.0 (see [LICENSE](LICENSE)). `plugin/moe_placement/persist_moe_cute.py` is derived from vLLM (Apache-2.0) and
FlashInfer CuTe-DSL kernels (BSD-3-Clause, NVIDIA); see [NOTICE](NOTICE). DeepSeek-V4.1 model weights are subject to
their own license.
