> **Provenance:** engineering notes written during the TidalMoE tuning work (Nov 2026) on the GB300 test machine.
> Paths such as `~/dsv41/...`, `~/dsv41/tuning/...` and `~/dsv41/moe/...` refer to that original workspace;
> in this repository the production pieces live in `serve.sh`, `plugin/`, `plans/`, `bench/`, `correctness/` and `tools/`.

# Inference bottlenecks on the DGX Station GB300 (224): Nsight Systems + Nsight Compute

Measured 2026-11-03 on the production setup: `~/dsv41/moe/serve-moe.sh` with `plan.json`, `MOE_ADAPT=1` (policy z), GPU memory utilization 0.98, `--max-num-seqs 32` and the dspark S1 schedule. The workload was ShareGPT with 1024 output tokens, `--ignore-eos`, and a warm server.

Every number below was measured on 224, except where a line says "model", "estimate" or "spec".

Tools:
- **Nsight Systems 2025.6.3** and **Nsight Compute 2026.1.0**, both from the host install under `/opt/nvidia`.
- GPU counters are admin-only on this box (`RmProfilingAdminOnly=1`), so both tools ran as root inside `--privileged` containers. No sudo was used.
- Grace uncore counters come through nsys `--cpu-socket-metrics` and through `perf` (`perf_event_paranoid=-1`).

## 0. Ranked bottlenecks

| # | Bottleneck | Where | Evidence (measured) |
|---|---|---|---|
| 1 | **The MoE GEMMs at conc ≥ 8 are limited by host (LPDDR5X) reads, and inside the kernel those reads neither reach the DRAM limit nor overlap with HBM reads** | `bmm_MxE4m3_MxE2m1` (GEMM1, w13) and `bmm_Bfloat16_MxE2m1` (GEMM2, w2) | **conc 32:** GEMM1 + GEMM2 take 38.8 ms of a 49.0 ms step (78% of kernel time). Inside the GEMMs, C2C reads average 216 / 207 GB/s.<br>**Same kernels and tactics, standalone:** host reads saturate at 277-280 GB/s, while a plain TMA stream with the same 128 x 128 B box pattern gets 386-393 GB/s.<br>**Within-kernel timeline:** long phases at 251-264 GB/s host while HBM sits at 1% and 100% of SMs are active, i.e. every SM is parked on a host tile.<br>**Microbenchmark `mix`:** 16 SMs read host at 389 GB/s *while* 136 SMs read HBM at 6.6 TB/s. |
| 2 | **Cold experts are spread unevenly across layers** | expert placement (plan + adaptive cache) | **conc 32:** the cold experts touched per layer per step range from 4.5 (L39) to 21.1 (L18); the mean is 11.8 and the total is 474 per forward (the plugin counts about 460-500).<br>**Cost model from the standalone (64 tokens, 175 distinct experts):** a layer costs about max(605 µs HBM, 64 µs × cold experts). Up to about 9.5 cold experts per layer the cost is hidden; above that each one costs about 64 µs.<br>Layers 2-18 are host-bound and layers 22-39 waste their free overlap. |
| 3 | **Low-batch (conc 1) decode kernels are latency-bound and use little of HBM** | MoE GEMMs at 6 tokens, dense projections, mHC, elementwise | **conc 1 step is 13.2 ms**, GPU busy 96%.<br>**MoE GEMMs (5.6 ms):** GEMM1 runs at 4.5 TB/s. GEMM2 (splitK2, 2880 CTAs mostly empty, 17% warps active) runs at **2.0 TB/s**.<br>**Dense GEMMs (3.5 ms, 314 kernels/step, 9 µs average):** some use only 40 CTAs.<br>**mHC (1.8 ms, 321 kernels/step):** grid 6 = one CTA per token on 152 SMs.<br>**About 900 single-CTA kernels/step (1.4-1.65 ms):** most are the moe_placement plugin's own per-layer counters (§3). |
| 4 | Per-step fixed costs that do not scale with the batch | plugin counter kernels (§3), launch gaps, draft model, sampler | **Plugin counter kernels:** 7 per layer in production and 18 with `MOE_BW_TRACE`, about 0.45 / 1.4 ms per step.<br>**GPU idle:** 0.5-0.6 ms/step.<br>**Eager draft / sampler / lm-head kernels at conc 32:** about 2 ms/step. |
| 5 | *Not a bottleneck:* CPU, launch overhead, and waiting on the host | engine thread | **conc 1:** the engine thread spends 7.3 ms of each 13.2 ms step blocked in `cudaEventSynchronize`.<br>**conc 32:** blocked 21-23 ms of each 49 ms step.<br>**GPU idle:** 3.8% at conc 1 and 1.2% at conc 32. The GPU waits on the host only in 1-2 gaps of more than 50 µs per step, in front of eager sampler/draft kernels.<br>**CPU LPDDR5X traffic while serving:** about 1 GB/s. |

## 1. Per decode step (nsys trace, production config)

Method: `scripts/serve-moe-nsys.sh` runs the production server under `nsys profile` with these settings:
- `--trace=cuda,nvtx --cuda-graph-trace=node`, so every kernel inside the CUDA graphs is recorded.
- CPU, context-switch and Python sampling.
- `gb11x-ct` GPU metrics at 20 kHz.
- Grace socket metrics at 1 ms.
- `--capture-range=cudaProfilerApi`, with vLLM `--profiler-config '{"profiler":"cuda"}'` and `/start_profile` and `/stop_profile` around 5 s of steady conc-32 decode and 4 s of conc-1 decode.

Tracing overhead was within noise: 75.6 steps/s traced against 73.8 untraced at conc 1, and 20.4 against 19.4-20.6 at conc 32.

**How the trace is segmented:** the main decode graph's kernel records carry correlationId 0, because the graph was instantiated before the capture. Steps are therefore cut on the GPU timeline at each occurrence of the graph's first node. This is done in `scripts/trace_analyze.py`.

| per step | conc 1 (6 tokens: 1 seq x (1+5 draft)) | conc 32 (64 tokens: 32 x 2) |
|---|---|---|
| step period (GPU) | **13.23 ms** (p50 12.74, p90 13.27), 75.6 steps/s | **48.97 ms** (p50 48.4, p90 54.5), 20.4 steps/s |
| GPU busy / idle | 96.2% / 0.51 ms (gaps > 50 µs: 1.6/step = 0.24 ms) | 98.8% / 0.58 ms (0.2 ms before eager `_zero_kv_blocks_kernel`) |
| kernels per step | 2202 in the target graph + 126 in the draft graph + 54 eager | 2011 in the target graph + 187 eager |
| sum of kernel durations vs union | 15.28 vs 12.73 ms (multi-stream overlap saves 2.5 ms; ≥ 2 kernels active 14.7% of the time) | 50.29 vs 48.39 ms (≥ 2 active 3.5%) |
| CPU (engine thread) | execute range 3.9 ms; 52 eager launches + 2.1 graph launches per step; blocked in `cudaEventSynchronize` 7.3 ms/step | execute range 5.9 ms; 188 eager launches; blocked 21-23 ms/step |
| MoE GEMM1 (w13) | 2.82 ms (40 + 3 draft calls, 66 µs each) | **26.12 ms** (643 µs/call, p10 421, p90 927) |
| MoE GEMM2 (w2) | 2.79 ms (67 µs each) | **13.31 ms** (327 µs/call) |
| dense GEMMs (attention projections, shared expert, indexer, lm_head, draft) | 3.51 ms (314 kernels) | 4.04 ms |
| mHC | 1.79 ms (321 kernels, tilelang, grid 6) | 1.15 ms (`sm100_mega_mhc_impl`, 152 CTAs) |
| elementwise / norm (1-CTA kernels; ≈ 85% are moe_placement counters plus `MOE_BW_TRACE`, §3) | 1.65 ms (≈ 900 kernels) | 1.84 ms |
| attention core (FlashMLA `fwd_kernel`, indexer, rope/quant/insert) | 1.45 ms (`fwd_kernel` 20.5 µs x 40, grid 6) | 1.46 ms (`fwd_kernel` 22.6 µs, grid 64) |
| routing + finalize | 0.44 ms | 1.05 ms |
| MXFP8 quantize | 0.40 ms | 0.44 ms |
| dspark draft (graph2 at conc 1, eager at conc 32) | ≈ 1.0 ms | ≈ 2.0 ms (incl. two 240 µs `nvjet` lm-head GEMMs) |
| spec / sample | 0.26 ms | 0.48 ms (`_mask_candidates_kernel` 70 µs x 4) |
| engram | 0.17 ms | 0.39 ms (`_engram_lookup_kernel` 96 µs x 2) |

**GPU metrics inside each kernel class.** These are time-weighted means of the 50 µs samples, so for kernels shorter than about 20 µs they are smeared over neighbouring kernels.

| class | conc 32: SM active / issue / tensor / HBM read % / C2C GB/s |
|---|---|
| GEMM1 | 97 / 12.7 / 8.5 / 45 / 216 |
| GEMM2 | 94 / 7.3 / 7.6 / 40 / 207 |
| dense GEMMs | 20 / 1.7 / 5.1 / 13 / 2 |
| mHC | 13 / 1.5 / 1.4 / 3.6 / 2 |

**Production, sampled system-wide with zero tracing overhead** (`scripts/sys_sample.sh`, nsys GPU metrics + Grace socket metrics, 15 s at conc 32):
- SMs active 89.9%, SM issue 9.5%, tensor 7.3%.
- HBM read: mean 35.6% of peak, p90 74%.
- Grace C2C0 GPU reads: mean 192 GB/s, p90 267, 1 ms peak 325.
- C2C0 read latency at the Grace port: p50 1.31 µs, against 219 ns for the CPU's own DRAM reads. The queueing is in the memory system.
- **Host reads and HBM reads mostly do not overlap.** Samples with HBM ≥ 50% *and* CTC RX ≥ 30%: 13%. Samples with CTC ≥ 30% and HBM < 50%: 49%.
- Bench during this run: 975 tok/s at conc 32 (TPOT 30.5 ms), measured soon after a restart (§5). 215 tok/s at conc 1.

## 2. MoE GEMMs: hot vs cold layers

**From the server trace.** Each GEMM call is classified by the mean CTC-RX (C2C read) inside it, converted to GB/s with the Grace C2C0 socket metric (1% ≈ 6.26 GB/s).

| GEMM1 calls, conc 32 | share of calls | mean duration | C2C rate | HBM read % |
|---|---|---|---|---|
| no host reads | 0.1% | 394 µs | – | 78 |
| medium host | 30% | 506 µs | 153 GB/s | 61 |
| heavy host | **66%** | **719 µs** | 241 GB/s | 41 |

GEMM2 shows the same pattern: 201 / 264 / 374 µs.

At conc 1:
- 71% of GEMM1 calls read nothing from host and take 58 µs (HBM 54%). The 13% heavy-host calls take 94 µs.
- The host share of GEMM time is small (about 14% for GEMM1, 7% for GEMM2). The plugin counts about 1 cold expert per layer per step (39.5 per forward).

**Per layer at conc 32.** Cold experts touched per step are derived from GEMM1 host bytes ÷ 11.8 MB:

```
L0:12.3 L1:10.7 L2:20.6 L3:17.4 L4:16.3 L5:20.5 L6:18.6 L7:10.7 L8:15.5 L9:17.7 L10:18.9 L11:11.8 L12:16.8 L13:16.3
L14:18.4 L15:5.0 L16:17.4 L17:15.7 L18:21.1 L19:8.5 L20:10.6 L21:13.2 L22:8.2 L23:5.9 L24:13.5 L25:9.4 L26:9.1 L27:6.3
L28:10.7 L29:8.8 L30:5.8 L31:9.3 L32:5.4 L33:5.9 L34:7.1 L35:8.4 L36:8.5 L37:7.8 L38:4.9 L39:4.5   (sum 474/forward)
```

**Standalone replica** (`scripts/moe_bench.py`):
- Same FlashInfer `trtllm_fp4_block_scale_routed_moe` call as vLLM `TrtLlmMxfp4ExpertsModular`, with tactics replayed from the server's autotune store. The kernel names and grids match the server exactly.
- Weights are VMM-backed through the production plugin's own `_alloc_segments`, using plan layer 10's 16-expert host blocks.
- Four layer copies are rotated and L2 is flushed between calls.
- Whole-call times without nsys match the server: 0.137 ms at conc-1 shape against about 0.128 ms in the server; 0.625 ms at conc-32 shape against 0.62 ms.

| case (tokens / distinct / cold) | GEMM1 µs | GEMM2 µs | host rate in GEMM | HBM % | samples with host and HBM both high |
|---|---|---|---|---|---|
| 64 / 175 / 0 | 403 | 202 | – | 77 / 73 | – |
| 64 / 175 / 6 | 414 | 216 | 174 / 162 GB/s | 73 / 67 | 51-56% |
| 64 / 175 / 12 | 519 | 278 | 275 / 248 | 55 / 50 | 63% |
| 64 / 175 / 24 | 1030 | 526 | **276 / 265** | 26 / 25 | 19% |
| 64 / 175 / 48 | 2038 | 1013 | **277 / 277** | 11 / 11 | 8-10% |
| 6 / 22 / 0 | 66 | 64 | – | 57 / 25 | – |
| 6 / 22 / 1 | 71 | 70 | 171 / 81 | 51 / 22 | |
| 6 / 22 / 3 | 132 | 95 | 267 / 187 | 25 / 15 | |

So per layer, GEMM time ≈ max(HBM time, cold bytes / ~277 GB/s):
- **conc 32:** the crossover is about 9.5 cold experts for GEMM1 (403 µs vs 42.6 µs per cold expert).
- **conc 1:** the crossover is about 1.5 cold experts.

**Nsight Compute** (`ncu/moe-*.ncu-rep`, full set, boost clocks, caches flushed each pass):

| kernel | duration | HBM | host reads | warps | issue | long-scoreboard share of stalls | grid / SM residency |
|---|---|---|---|---|---|---|---|
| GEMM1, 64 tok, all HBM | 399 µs | 5.51 TB/s (77% of the 7.16 spec) | – | 31% | 20.8% | 83% | 13824 CTAs, 1 CTA/SM (137 KB smem, 640 thr) |
| GEMM2, 64 tok, all HBM | 206 µs | 5.36 TB/s | – | | | 83% | |
| GEMM1, 64 tok, 48 cold | 2037 µs | 0.80 TB/s | 566 MB = **278 GB/s** | 29.7% | 4.4% | **96%** | per-SM TMA bytes min 12.8 / max 44 MB (3.4x imbalance) |
| GEMM1, 6 tok | 61 µs | 4.50 TB/s | | 24% | | 82% | |
| GEMM2, 6 tok (splitK2) | 70 µs | **1.99 TB/s** | | 17% | | 50% | grid 2880 (40x36x2), most CTAs empty |

Further notes from Nsight Compute:
- Host reads of EGM pages are counted under the **peer** aperture (`syslts__t_sectors_aperture_peer_lookup_miss`), not `sysmem`.
- The sysLTS path is at only 27% of its peak, so the GPU-side port is not the limit.

**Why the single call loses overlap.** CTAs are ordered by (N-tile, expert). The 36-40 CTAs of a cold expert each hold an SM for a long time. When host bandwidth saturates, SMs fill up with cold CTAs, because hot CTAs finish fast and their SMs pick up the next cold CTA. From then on all 152 SMs wait on host data at about 257 GB/s and HBM goes idle.

**Validation with the real kernels** (`scripts/green_split.py`):
- Setup: CUDA green contexts (torch 2.14 `GreenContext`, disjoint SM partitions via `cuDevSmResourceSplitByCount`) and a hot-prefix layout (292 HBM + 92 host, the tactic-tuned shape).
- The cold-expert call runs on K SMs and the hot-expert call on the remaining SMs, concurrently, and the outputs are added.

| cold experts/layer (64 tok) | single call, hot-prefix layout | split, no partition | **green split**, cold on 16 / 24 / 32 SMs | production interleaved single call |
|---|---|---|---|---|
| 6 | 1.028 ms | 1.002 | 0.877 / 0.885 / 0.893 | **0.649** |
| 12 | 1.389 | 1.359 | 0.910 / **0.872** / 0.879 | **0.810** |
| 20 | 1.763 | 1.729 | 1.304 / 1.139 / **1.099** | ≈ 1.32 (interpolated) |
| 30 | 2.203 | 2.176 | 1.790 / 1.584 / **1.544** | ≈ 1.95 (extrapolated at 62 µs/expert) |

What the split shows:
- Partitioning the SMs lifts the cold call to at least 344 GB/s (531 MB in 1.544 ms on 32 SMs) and restores the overlap.
- Without a partition the split is barely faster than a single call, which reproduces the earlier MOE_SPLIT result.
- With the stock kernels it wins only above about 15 cold experts per layer. Two calls plus the add cost about 0.15-0.25 ms, and the hot call loses SMs.
- The split changes the summation order (max |diff| 0.0039 in bf16), so it is **not bit-identical** to the single call.

## 3. Non-MoE kernels (Nsight Compute on the live server, conc 1)

**Nsight Compute could not profile the live server.**
- `scripts/serve-moe-ncu.sh` starts production under `ncu --profile-from-start off --graph-profiling node`. Even with profiling off, ncu's per-node interception slowed serving from about 75 to about 9 steps/s.
- At `/start_profile`, profiling the first matched graph node failed: `==ERROR== Failed to profile "vectorized_elementwise_kernel"` and the app was killed.
- The most likely cause is that kernel replay must save and restore device memory, and only about 1 GiB of HBM is free at GPU memory utilization 0.98. The server cannot run lower: weights and non-torch allocations already use 236 GiB, and the KV cache needs ≥ 2.03 GiB.

The non-MoE kernels are therefore characterized from the nsys trace below. The figures are durations, grids, and bytes moved, the bytes inferred from the shape tags in the CuTe kernel names. Only the MoE GEMMs got full Nsight Compute treatment, via the standalone.

**Top non-MoE kernels at conc 1** (per step; 40 MoE layers; inferred bandwidth = weight bytes / duration):

| kernel | calls/step | avg | grid x block | notes |
|---|---|---|---|---|
| FlashMLA sparse `fwd_kernel` (attention core) | 43 | 19.9 µs | 5-6 CTAs x 512 | one CTA per query token: **6 of 152 SMs** at conc 1 |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 162 | 5.2 µs | **5-6 CTAs** x 96 | mHC (hc_mult 4); about 4 launches per layer, each on 6 SMs |
| `mhc_post_tilelang_kernel` | 81 | 4.9 µs | 6 CTAs | |
| `deep_gemm::sm100_tf32_hc_prenorm_gemm_impl` | 80 | 6.1 µs | 16 CTAs | |
| FlashInfer `dense_blockscaled_gemm` (persistent) | 54 | 12.2 µs | **40 CTAs** | |
| MXFP8 warp-split-K GEMM `o1792x2560` (q_a + kv_a; 4.6 MB MXFP4) | 46 | 11.9 µs | 112 CTAs | about 0.39 TB/s |
| MXFP8 warp-split-K GEMM `o32768x640` (q_b; 21 MB) | 43 | 11.5 µs | 2048 CTAs | about 1.8 TB/s |
| `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl` | 43 | 10.2 µs | 152 CTAs | |
| MXFP8 warp-split-K `o4608x2560` (shared expert w13, 11.8 MB) | 43 | 8.2 µs | 288 CTAs | about 1.4 TB/s |
| MXFP8 warp-split-K `o5120x1152` (shared expert w2, 5.9 MB) | 43 | 6.2 µs | 160 CTAs | about 0.95 TB/s |
| `nvjet` lm-head / draft head | 7 | 66 µs (210 µs eager target lm-head) | 1010 CTAs | about 1.3 GB bf16 → about 6 TB/s (fine) |

Every dense projection runs at **0.4-1.8 TB/s at conc 1** (5-25% of HBM). They are launch- and latency-bound kernels of 6-12 µs, about 314 per step, 3.5 ms in total.

**Most of the about 900 tiny 1-CTA kernels per step come from the moe_placement plugin, not from vLLM.** The per-layer GPU counters in `_patch_adapt.apply` launch about 7 kernels per MoE call in production (`MOE_ADAPT=1`): `.long()`, two compares, `&`, `.to()`, `clamp`, `index_add_`. With `MOE_BW_TRACE=1`, which was on in this trace and in the bw-sequence, about 11 more are added: zeros, `index_add_`, `>0`, `& hm`, `~hm`, two `sum` reductions, three `+=`.

Measured at conc 1:

| kernel | per step |
|---|---|
| `direct_copy` | 206 kernels, 383 µs |
| `sum` reduce | 80, 189 µs |
| `compare` | 128, 160 µs |
| `BitwiseAnd` | 121, 159 µs |
| `scatter_gather` (`index_add_`) | 80, 127 µs |
| `add` | 129, 196 µs |
| `clamp` | 81, 96 µs |
| `not` | 41, 55 µs |
| `fill` | 40, 41 µs |
| **total** | **≈ 1.4 ms/step, 11% of the 13.2 ms step** |

About 0.45 ms of that remains in production (counters only) and about 0.95 ms exists only while `MOE_BW_TRACE=1`. **Fix:** one fused counting kernel per layer, e.g. a Triton atomic-add over `topk_ids` with the validity mask folded in. Better still, count once per step from the routing record. **Gain: about 0.4 ms/step in production, +3% at conc 1 and +1% at conc 32.** Also, bandwidth traces with `MOE_BW_TRACE=1` understate conc-1 speed by about 7%.

At conc 32 these same kernels cost about 1.8 ms/step (4%). The non-MoE GEMMs (4.0 ms, 8%), mHC (1.15 ms, through the 152-CTA `sm100_mega_mhc_impl`) and attention (1.46 ms) matter much less there than the host-bound MoE GEMMs.

## 4. Architecture microbenchmarks (measured, `microbench/quick/`)

`~/dsv41/tuning/arch/bench/run_all.sh` ran unchanged with `QUICK=1` and no fixes were needed. Every section ran under Grace PMU counting.

| what | result |
|---|---|
| HBM streaming read (LDG) | 6.43-6.57 TB/s with 152-608 CTAs (92% of the 7.16 TB/s spec); 2.7 TB/s with only 152 CTAs x 8 KB in flight |
| **GPU read of host memory, cuMemCreate HOST_NUMA (EGM, the plugin's cold-expert pages)** | **383-390 GB/s** once about 1 MB is in flight (32 CTAs x 32 KB). That is 98% of the LPDDR5X spec (396) and above the firmware's 360. |
| same, cudaHostAlloc / malloc (ATS) | 353 / 346 GB/s |
| **TMA 128 x 128 B boxes with the MoE strides (2560 B = w13, 1152 B = w2) from EGM** | **386-393 GB/s**, identical to contiguous (128 B) boxes. 16 CTAs x 5 stages or 32 x 2 stages saturate it; 8 CTAs x 3 stages gives 288 GB/s. |
| per-SM limit | TMA: 51.2 GB/s per SM for both HBM and EGM (≥ 4 stages = 64 KB in flight). LDG: 73 GB/s per SM (1024 thr x 8 x 16 B). |
| latency, pointer chase (2 MB stride, TLB-friendly) | HBM 134-145 ns, EGM 634-704 ns, hostalloc/malloc 648-816 ns. 64 B-stride chase on large footprints: EGM about 860 ns. |
| latency, 16 KB TMA box round trip | EGM 1.08 µs, HBM 0.83 µs |
| GPU L2 caching of host data | **none**: EGM 64 MB re-read 8x stays at 383 GB/s, while HBM 64 MB rises to 14.6 TB/s from L2 |
| Grace L3 (data touched by the CPU first) | no benefit (281 vs 281 GB/s) |
| **host + HBM reads on disjoint SMs at the same time (`mix`)** | host 8 SMs: 365 GB/s + HBM 144 SMs: 6.66 TB/s<br>**host 16 SMs: 389 GB/s + HBM 136 SMs: 6.62 TB/s**<br>host 32: 387 + 6.53 TB/s<br>host 96: 386 + 5.25 TB/s<br>The two are **additive**: the link and LPDDR5X can be saturated by 16 SMs without slowing HBM. |
| copy engine H2D from EGM, by chunk size | 1 MB: 188 GB/s · 2 MB: 253 · 4 MB: 303 · 16 MB: 365 · ≥ 64 MB: 386-392 · D2H 378 |
| CE copy concurrent with an HBM-read kernel | CE 394.7 GB/s + kernel 6.05 TB/s (kernel alone about 6.5) |
| CE copy concurrent with a host-read kernel | 233 + 195 GB/s (they share LPDDR5X) |
| random 256 B gathers over 16 GB (engram-like) | EGM 370-376 GB/s; cudaHostAlloc / THP-registered / malloc 286-341 GB/s (152-608 CTAs). Latency implied at low concurrency: about 950 ns. |
| GPU writes to EGM | 370 GB/s |
| Grace CPU stream, 72 threads | read 281, write 355, triad 278 GB/s |
| **CPU and GPU reading LPDDR5X at the same time** | the GPU's EGM read drops from 386 to **189 GB/s** while 72 CPU threads read 294 GB/s. It is one shared DRAM. |
| CPU MXFP4 expert compute (11 experts, 1 token, 72 cores) | 0.80 ms (260 GB/s), against the GPU's C2C read equivalent of 0.61 ms. Offloading experts to the CPU does not pay. |

What this means for this machine, as opposed to a PCIe box:
- **The ceiling is LPDDR5X itself (about 390 GB/s), not the 447 GB/s C2C link.** CPU traffic comes out of the same budget.
- **The GPU's TMA can saturate that ceiling with only 8-16 SMs.** Host latency is about 0.65-1.1 µs, only 1.3-4.6x HBM's, so deep pipelines on a few SMs suffice.
- **Host data is never cached on the GPU (L2) or by the Grace SLC on the GPU's behalf.** Every touch of a cold tile costs DRAM bandwidth.
- **HBM and host reads are fully additive when they run on different SMs.** The current MoE kernel loses this because host tiles take over all SMs.

## 5. Expert cache: what is left to tune (answer to 「關於expert cache還有能調優的嗎」)

The detailed simulations are in `~/dsv41/tuning/cache-analysis/REPORT.md`, written by another agent. The Nsight data adds three things:

1. **The cost of a cold expert is convex, not linear.** At 64 tokens/step a layer hides up to about 9.5 cold experts (about 1.5 at conc 1) behind its HBM work for free; after that each costs about 64 µs at the kernel's 277 GB/s. The objective should therefore be Σ over layers of max(0, cold_L - free_L), not the total number of DRAM touches. Today's per-layer skew (4.5-21 cold per layer) is the main waste.
   - **Model, at conc 32 with measured per-layer counts:** today about 33.0 ms; balanced about 30.3 ms; measured 38.8 ms, which includes per-step variance and is also convex.
   - **Recommendation:** re-plan the per-layer `n_hot` budgets with this cost model, moving HBM slots from layers 22-39 to layers 2-18. Let the adaptive cache rank swap candidates by marginal step time per layer, not by raw counts.
   - **Expected gain:** +5-8% at conc 32, with placement changes only, so the outputs stay bit-identical. The model gives -2.7 ms/step, and the per-step variance it also removes is not counted.
2. **A restart throws away the cache state.** After the nsys restart, epoch 3 still had a 27.9% DRAM share of routed selections. A conc-32 bench right after a restart measured 975 tok/s, against 1170 in the README. Persist the EMA scores and the slot order at shutdown and reload them at start (warm start).
3. **The per-expert cost depends on the kernel.** With the SM-partitioned host reads in §6.1 the cost drops from about 64 µs to about 45 µs (at 390 GB/s), and it becomes fully hidden while a layer has HBM work left.

Cold-expert *bytes* also matter. Each cold touch reads w13 (11.8 MB, GEMM1) and w2 (5.9 MB, GEMM2). The fp8 scales stay in HBM, which is correct, since host latency would also hit the scale loads.

## 6. The three most promising optimizations (grounded in the measurements above)

### 6.1 MoE kernel: give the host tiles a bounded set of SMs with deep TMA pipelines (largest gain at conc ≥ 8)

**Problem:**
- The single TRT-LLM call lets cold tiles occupy every SM. The host stream then runs at 257-277 GB/s and HBM goes idle (§2).
- The hardware can do 389 GB/s host on 16 SMs *plus* 6.6 TB/s HBM on the other 136 at the same time (`mix`).

**Options:**
- **(a) No kernel source changes:**
  - Use a hot-prefix plan and green contexts, with cold experts on 24-32 SMs and hot experts on the rest, as in `green_split.py`.
  - Per layer, take the split only where that layer's expected cold count is ≥ 15 (layers 2-18 today); this choice is static and safe inside a CUDA graph.
  - Tactics for the cold call need tuning on host data, not HBM.
  - It is not bit-identical (summation order).
  - **Estimate, from the measured per-layer cold counts and the split timings:** about -1.9 ms/step, i.e. about +3-5% at conc 32.
- **(b) The real fix: a persistent MXFP4 grouped GEMM with tile scheduling aware of the host/HBM split:**
  - 8-24 warp-specialized producer CTAs pinned to cold tiles, with ≥ 4-stage, ≥ 64 KB TMA pipelines (51 GB/s per SM).
  - The remaining SMs stream hot tiles.
  - One finalize, which keeps the accumulation order and makes bit-exactness possible.
  - Starting point: the CuTe-DSL tcgen05 low-batch routed-expert kernels already in the image (`vllm/models/deepseek_v41/nvidia/specialized_decode/moe/routed_expert_gemm_cute.py`).
  - **Model, using the measured per-layer cold counts:** Σ max(620 µs, 45.4 µs x cold_L) = 27.5 ms per step, against the measured 38.8 ms.
  - **Expected: up to -11 ms/step at conc 32, about +25-30% tok/s; +10-20% at conc 8-16; about 0 at conc 1.**

### 6.2 Convex-cost expert placement and a warm-started cache (cheap, bit-identical)

- Re-plan the per-layer HBM budgets and rank swaps by marginal step time, as described in §5.
- Persist the adaptive state across restarts.
- Combine with the cache report's items: about 272-320 more HBM experts from KV headroom, the vision encoder, `embed.weight` and GPU utilization 0.99, and scoring decode touches only.
- **Expected: +5-8% at conc 32 from balancing alone. The cache report estimates +12% for its #2-#4; the two only partly overlap.**
- Runs immediately after a restart improve by up to about +20%, judging from the 975 vs 1170 tok/s gap.

### 6.3 Low-batch decode kernels (conc 1-8): MoE GEMMs at 6 tokens, mHC with 6 CTAs, about 900 single-CTA elementwise kernels

All numbers are at conc 1 (13.2 ms/step).

- **MoE GEMMs: 5.6 ms/step at 2.0-4.5 TB/s.**
  - GEMM2 is splitK2, with 2880 CTAs most of which are empty, and reaches only 1.99 TB/s.
  - vLLM already contains a persistent CuTe-DSL 128x8 tcgen05 path, but it is hard-coded for the TP4 slice (I = 576/640). Port it to TP1 (I = 2304; w13 [384, 4608, 2560], w2 [384, 5120, 1152]).
  - At about 6 TB/s the per-layer MoE drops from about 125 µs to about 62 µs: **-2.4 ms/step**.
- **mHC: `mhc_pre_big_fuse_with_norm` + `mhc_post` + `hc_prenorm_gemm`, 1.7 ms/step and 321 launches/step.**
  - The grid is 6 CTAs (one per token) on 152 SMs.
  - Split the hidden dimension across ≥ 64 CTAs, or use the `sm100_mega_mhc_impl` path that conc 32 already uses: **-0.6 to -1.0 ms**.
- **Plugin counters: about 280 tiny kernels per step in production.** Fuse them into one kernel per layer, or one per step: **-0.4 ms** (§3).
- **Dense projections: 3.5 ms/step at 0.4-1.8 TB/s, 6-12 µs kernels with 40-288 CTAs.** Fusing q_a+kv_a+indexer and the shared expert into persistent split-K GEMMs that use all 152 SMs: **-1 to -1.5 ms**.
- **Expected: 13.2 → about 9-10 ms/step, +30-40% at conc 1 and +15-25% at conc 8.** Bit-exactness of a new MoE kernel against TRT-LLM-gen has to be re-validated with `tf_logprobs.py`.

Not worth doing, from the measurements:
- Computing cold experts on the Grace CPU. It reads the same DRAM at 260 GB/s, and CPU traffic halves the GPU's host bandwidth.
- Making the link faster. The link is not the limit; LPDDR5X is.
- Copy-engine prefetch inside a layer. The routing is known too late, and CE chunks under 16 MB run at only 188-303 GB/s.
- Cross-layer prefetch. It needs a router predictor (cache report #1), and it can use 16 SMs or the CE alongside the HBM GEMM without loss (`mix`, `cemix_hbm`).
- Moving engram to EGM pages (cuMemCreate HOST_NUMA). Random gathers run at 370 vs 292 GB/s, but engram is only about 1% of the step today.

## 7. Bandwidth-sequence results (`results/bw-sequence.log`, copied)

```
hot/cold+adaptive  c= 1 | GPM: SM 57.5% tensor 1.2% HBM 28.1% (max 32%) | plugin: 73.8 steps/s, experts/forward DRAM 39.5 HBM 855.7, DRAM 50.9 GB/s, HBM 1189.4 GB/s
                        | PMU: GPU reads of LPDDR5X over C2C 50.6 GB/s (max 1s 101.0; link 447)
hot/cold+adaptive  c= 8 | GPM: SM 74.8% tensor 7.1% HBM 40.2% (max 46%) | plugin: 35.2 steps/s, DRAM 205.4 HBM 3684.7, DRAM 127.2 GB/s, HBM 2445.1 GB/s
                        | PMU: 128.7 GB/s (max 1s 156.2)
hot/cold+adaptive  c=16 | GPM: SM 81.1% tensor 7.9% HBM 42.4% (max 47%) | plugin: 27.5 steps/s, DRAM 294.6 HBM 5068.5, DRAM 142.3 GB/s, HBM 2623.2 GB/s
                        | PMU: 142.8 GB/s (max 1s 166.7)
hot/cold+adaptive  c=32 | GPM: SM 84.4% tensor 7.3% HBM 37.6% (max 43%) | plugin: 19.4 steps/s, DRAM 501.4 HBM 6491.0, DRAM 170.0 GB/s, HBM 2377.8 GB/s
                        | PMU: 170.3 GB/s (max 1s 186.8)
layer-offload stock c= 1 | GPM: SM 76.2% tensor 1.0% HBM 14.3% | plugin: 42.8 steps/s, DRAM 218.5 HBM 702.6, DRAM 165.5 GB/s | PMU 165.5 (max 179.7)
layer-offload stock c= 8 | GPM: SM 91.0% tensor 3.0% HBM 14.8% | plugin: 15.1 steps/s, DRAM 928.9 HBM 3031.8, DRAM 248.4 GB/s | PMU 248.9 (max 252.4)
layer-offload stock c=16 | GPM: SM 93.2% tensor 3.0% HBM 14.9% | plugin: 11.7 steps/s, DRAM 1250.3 HBM 4116.0, DRAM 258.1 GB/s | PMU 258.0 (max 263.6)
layer-offload stock c=32 | GPM: SM 92.4% tensor 3.0% HBM 14.6% | plugin:  9.0 steps/s, DRAM 1616.6 HBM 5398.9, DRAM 256.0 GB/s | PMU 258.1 (max 268.5)
(CPU's own LPDDR5X traffic 0.0-0.2 GB/s in every row; GPU C2C writes <= 0.4 GB/s)
```

The stock layout plateaus at 256-258 GB/s. That is the same ceiling the MoE kernel reaches once host tiles fill all SMs (§2: 257-277 GB/s), well below the 390 GB/s the DRAM delivers to a TMA stream.

## 8. Files (all under `~/dsv41/tuning/nsight/`)

- `scripts/serve-moe-nsys.sh`: the production server under nsys with a CUDA-graph node trace, between `/start_profile` and `/stop_profile`.
- `scripts/trace_run.sh`: drives warm-up and two capture windows.
- `scripts/trace_analyze.py`: per-step and per-layer analysis.
- `scripts/sys_sample.sh` and `scripts/nsys_sys.sh`: GPU metrics plus Grace socket metrics against the running server, with no tracing.
- `scripts/sysmetrics.py`: summary of those metrics.
- `scripts/moe_bench.py` and `scripts/run_moe_bench.sh` (modes time / nsys / ncu): the standalone MoE layer.
- `scripts/bench_analyze.py`: per-case kernel metrics for the standalone runs.
- `scripts/ncu_cases.sh` and `scripts/ncu_summary.py`: ncu runs of the standalone and their summary.
- `scripts/green_split.py`: the SM-partition experiment.
- `scripts/serve-moe-ncu.sh` and `scripts/ncu_live_run.sh`: the production server under ncu, profiling starting at `/start_profile`.
- `reports/`:
  - `prod-trace.1` (conc 32) and `prod-trace.2` (conc 1): `.nsys-rep`, `.sqlite` and `.analysis.txt`.
  - `prodsys-c32` and `prodsys-c1`: production with no tracing.
  - `bench-nsys1`: the standalone.
- `ncu/`:
  - `moe-{c1_hbm,c1_cold3,c32_hbm,c32_cold12,c32_cold48}.ncu-rep`
  - `live-c1.ncu-rep`
  - `summary.txt`
- `microbench/quick/`: `*.results`, `*.pmu.txt`, `index.txt`.
- `logs/`

Output files written from containers are root-owned.

Caveats:
- GPU-metric attribution to kernels shorter than about 20 µs is smeared (50 µs samples).
- The CTC-RX → GB/s calibration (5.2-6.3 GB/s per %) comes from the Grace C2C0 socket metric.
- The standalone uses random weights (timing only) and synthetic routing (2.2 tokens per expert at 64 tokens).
- The green-split numbers are eager launches; inside CUDA graphs the fixed overhead would be smaller.
