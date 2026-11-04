> **Provenance:** engineering notes written during the TidalMoE tuning work (Nov 2026) on the GB300 test machine.
> Paths such as `~/dsv41/...`, `~/dsv41/tuning/...` and `~/dsv41/moe/...` refer to that original workspace;
> in this repository the production pieces live in `serve.sh`, `plugin/`, `plans/`, `bench/`, `correctness/` and `tools/`.

# Host-aware persistent MoE kernel for DeepSeek-V4.1 decode on GB300 (design + groundwork)

Status: design and CPU-verified groundwork. **No GPU was used.** Code lives in `~/dsv41/tuning/persist/src` on 224.
Baseline: the `~/dsv41/moe-cand` config (plan-e207-g8, adaptive cache with fused touch counting, MOE_TACTICS
`8=8:437,16=16:388,32=8:71,64=32:119`, embedding on host, KV for 1M tokens + 64 GiB CPU KV). Its plugin is identical to
`~/dsv41/tuning/plugin`. Production `~/dsv41/moe` is still the older plugin; nothing there was touched.

Each claim below is tagged:
- **[V]** verified in this session (CPU only)
- **[M]** measured earlier on 224 (BOTTLENECKS.md, ARCHITECTURE.md, kernel/ncu/summary.txt)
- **[S]** simulated (`sched_sim.py`)
- **[A]** an assumption that the GPU stages must check

> **Implementation status (2026-11-04): stages (i) to (iii) are done. Results are in `README.md`.**
>
> * **Stage (i).** GEMM2 rows are bit-identical to TRT-LLM's non-split-K tactics.
> * **Stage (ii).** One MoE layer runs in −24 to −30 % of TRT-LLM's time with 4 to 18 cold experts.
>   * ncu, T = 64 with 12 cold: host 378 GB/s concurrent with HBM 5.97 TB/s.
> * **Stage (iii).** Server ShareGPT tok/s against production: c1 +15 %, c8 +15 %, c32 +23 %, zh c32 +23 %.
>   * Prefill logprobs are bit-identical.
> * **Changes from this design, with evidence:**
>   * FP4 operands are stored unpacked as 8-bit in shared memory, so a stage uses twice the TMA bytes.
>     GEMM1 uses 223 KB of shared memory and GEMM2 186 KB, not 120 / 100, so GEMM1 and GEMM2 are never co-resident.
>   * The front launches with PDL and issues all its loads first (v2).
>   * The static split was kept: kc, bandwidth and stage sweeps all land within ±1 %.

## 0. TL;DR

* **Route:** extend the image's own persistent CuTe-DSL routed-expert GEMMs, which were written for TP4 at 8 or fewer rows, to TP1 and up to 128 rows, and add a host-aware schedule.
  * Those kernels already read the exact TRT-LLM-gen MXFP4 layout that vLLM prepares, with TRT-LLM-identical numerics: ascending K, no split-K, the same epilogue PTX.
  * They already use TMA with deep pipelines, warp specialization, a double-buffered TMEM accumulator and PDL.
* **Key change: per call, a small number of CTAs own all cold items.** `kc` (4 to 16) CTAs take every cold-expert item first: weights in host DRAM, read over NVLink-C2C, with 6- or 10-stage TMA pipelines. They then take `hoc` hot items from the end of the hot list. The other `152 - kc` CTAs stream only HBM experts.
  * The front kernel picks `kc` and `hoc` per call from a bandwidth model.
  * This reproduces the measured `mix` microbenchmark: 16 SMs read host at 389 GB/s while 136 SMs read HBM at 6.62 TB/s [M]. It does so inside one launch per GEMM, with no extra launches, no add and no green contexts.
* **Groundwork done and checked on CPU [V]:**
  1. Layout reference (`ref_layout.py`): matches FlashInfer's shuffle maps and vLLM's TRT-LLM conversion code.
  2. Routing record v2 spec plus the per-CTA schedule (`route_ref.py`): 300 random routings; every (tile, m-tile) item is covered exactly once.
  3. The **host-aware TP1 CuTe GEMM1 and GEMM2** (`persist_moe_cute.py`, generated from the image's file by exact edits) **compile for sm_103a**: 40 / 34 registers, 120 / 100 KiB staged smem.
  4. **Triton front and finalize kernels** (`persist_front.py`): compile for sm_103a and match the numpy spec bit for bit in the Triton CPU interpreter.
  5. Plugin binding (`persist_moe.py`) and the GPU bench for stages (i)/(ii) (`pbench.py`): written, byte-compiled and import-checked, never run.
* **Expected gain [S], anchored to [M]:**
  * **c32:** a MoE layer at 64 tokens with about 10 to 13 cold experts goes from about 0.77 to 0.80 ms (measured GEMM1+GEMM2) to about 0.55 to 0.63 ms. That is about -6 to -9 ms of a 45 ms step, **about +15 to 20 % tok/s**.
  * **c8:** MoE -30 %.
  * **c1:** MoE -25 % or more, because the TRT-LLM low-batch GEMM2 split-K path runs at only 2 to 3 TB/s [M].
* **Time estimate:** 2 to 3 GPU-days through stage (iv). The highest-risk step is stage (i), the first GPU run of the modified CuTe kernel.

## 1. What exists today (study)

### 1.1 Production path: FlashInfer TRT-LLM-gen routed MoE

* **Call chain:** vLLM `TrtLlmMxfp4ExpertsModular.apply` → `flashinfer.trtllm_fp4_block_scale_routed_moe` (FlashInfer 0.7.2.post1) [V].
  * Precomputed top-k ids go to the routing kernel.
  * GEMM1 is `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_..._clmp_geGlu_dynB_sm100f`, with the SwiGLU and the clamp fused.
  * GEMM2 is `bmm_Bfloat16_MxE2m1MxE4m3_...`, followed by finalize. For 16 tokens or fewer, finalize is deferred into the mHC kernel through `UnfinalizedMoEOutput` (`MHC_OVERLAP_MAX_TOKENS = 16`).
* **Cubins and tactics:**
  * The cubins are prebuilt in `flashinfer_cubin/cubins/8ee5…/batched_gemm-fdba669-16ba8ed/`.
  * Kernel names encode the tile (`t128x{8,16,32}x{256,512}`), the stages (`s3`–`s6`), split-K, the cluster, `ldgstsSf` (scale factors loaded with LDGSTS) and `rgTma` (routed gather of activations with TMA).
  * The autotune store is `~/dsv41/cache/flashinfer/autotune/v2/e84a378bdee44552`. MOE_TACTICS forces a (tile_n, config) pair per token bucket, through the plugin's `AutoTuner.choose_one` patch.
* **CTA mapping [M]:**
  * The grid is non-persistent. It is `(36 m-tiles, up to 384 token tiles)` for GEMM1 at 64 tokens and `(40, 384)` for GEMM2.
  * `dynB` means CTAs past the real tile count exit immediately.
  * There is 1 CTA per SM (137–223 KB smem).
  * CTAs are dispatched x-fastest, so all m-tiles of expert tile 0 go first, then tile 1, and so on, in **expert order**.
  * Each CTA computes one 128 × tile_n output tile over K = 5120 (GEMM1) or K = 2304 (GEMM2).
* **Why the overlap is lost [M]:**
  1. A CTA whose expert is cold streams about 320 KB from host. Only about 22 GB/s per CTA is effective, because of the kernel's shallow pipeline and LDGSTS scale-factor loads, so the CTA holds its SM about 2× longer than a hot one.
  2. The hot CTAs drain, and their SMs pick up the next CTAs in expert order, which are often cold.
  3. Soon all 152 SMs wait on host at about 257–280 GB/s while HBM idles. ncu shows long phases at 251–264 GB/s host with HBM at 1 %.
  4. Measured cost: 64 tokens, 175 experts → GEMM1 + GEMM2 = 605 µs with 0 cold, 630 with 6 cold, 797 with 12 cold, 1556 with 24 cold. With the production tactics at 10 cold it is 518 + 248 µs, with in-GEMM host reads at only 228 / 238 GB/s.
* **Low batch [M]:**
  * At 6 tokens, GEMM1 takes 98 µs and runs at 4.2 TB/s.
  * GEMM2 is split-K2 with 2880 mostly-empty CTAs. It takes 69 µs and runs at 3.0 TB/s.

### 1.2 The image's persistent CuTe-DSL routed-expert path

The path is `vllm/models/deepseek_v41/nvidia/specialized_decode/moe/`: `routed_expert_gemm_cute.py`, `routed_expert_gemm.py`, `routed_expert_route_cute.py` and `routed_expert_front_cute.py`. A copy is in `~/dsv41/tuning/persist/ref/`.

* **Kernel structure.** It is a persistent `tcgen05` block-scaled GEMM with 128 × 8 output tiles. Work items are (weight m-tile, 8-row token tile). CTA `b` of `G = #SMs` takes items `b, b+G, …`. The warps are:
  * warp 0 loads A and its scales with TMA, with an evict-first hint;
  * warp 1 loads B (and its scales);
  * warp 2 issues the MMAs;
  * warps 4–7 run the epilogue (warp 3 is idle).
* **Pipeline and accumulator.** GEMM1 has 6 stages with `cta_k = 256`; GEMM2 has 10 stages with `cta_k = 128`. The accumulator is double-buffered in TMEM (128 columns).
* **PDL.** GEMM2's A warp starts streaming weights before GEMM1 finishes; only its B warp waits.
* **Numerics.**
  * The MMAs run in ascending K with no split-K, as the TRT-LLM `t128x8` tactics do.
  * The GEMM1 epilogue (clamped SwiGLU, UE8M0 scale, E4M3 quantization) is TRT-LLM's arithmetic, written out in PTX.
  * The docstring claims bitwise equality with TRT-LLM-gen.
* **What it is hard-coded for (TP4 low batch):**
  * 8 rows or fewer. GEMM1's B is the whole `x` (≤ 8 rows), and the epilogue picks each tile's tokens.
  * Intermediate 576 padded to 640, with the padding tile zeroed.
  * The route kernel supports at most 8 rows × top-8.
  * It only runs under `decode_backend=latency_specialized`, which requires TP4, FlashInfer MLA and multicast.
* **It compiles on CPU [V].** With `CUTE_DSL_ARCH=sm_103a` and no GPU, `cute.compile` produces real sm_103a cubins in 0.3–0.5 s, for the TP4 shapes and also for TP1 shapes.

### 1.3 Other options available in the image [V]

* **DeepGEMM** is vendored as `vllm/third_party/deep_gemm` (JIT C++ headers under `include/deep_gemm`).
  * It includes **Mega-MoE** (`impls/sm100_fp8_fp4_mega_moe.cuh`, `scheduler/mega_moe.cuh`). This is a fused persistent dispatch + L1 + SwiGLU-clamp + L2 + combine kernel, with an atomic task scheduler, L1→L2 readiness masks (`L2KBlockDependency`), per-die locality domains and work stealing. It runs with EP = 1 (a plain buffer when `group.size() == 1`).
  * vLLM only enables it with `--enable-expert-parallel`, and it uses its own weight layout: gate/up interleaved by 8, UE8M0 scales packed into int32 and transposed for UTCCP.
  * Hot experts take about 199 GiB of HBM, so a second copy is impossible. Adopting Mega-MoE would therefore mean converting every expert, sending prefill through DeepGEMM too, and rewriting the plugin's slot and swap logic. Its scheduler is not host-aware either.
  * **Use it as a design reference** for the dynamic queue (v2) and the fused L1→L2 version (v3), not as the base.
* **CUTLASS DSL** 4.8.0 (`nvidia-cutlass-dsl`) and quack-kernels 0.6.5.
* **Triton** 3.8.0. It compiles for sm_103a and supports `TRITON_INTERPRET=1` on CPU [V].
* **tilelang** 0.1.12 and cuda-tile 1.7.
* **nvcc:** 13.0 in the image and 13.2 on the host. CUTLASS C++ headers ship inside FlashInfer and DeepGEMM.
* **ncu** 2026.1 and **nsys** 2025.6 on the host. Profiling needs the privileged-container route in `nsight/scripts`.

## 2. Route choice

| route | first result | risk | verdict |
|---|---|---|---|
| (a) extend the image's CuTe-DSL persistent kernels to TP1 + host-aware schedule | stage (i) in about 1 day | kernel already reads the TRT-LLM layout, numerics known; changes are local (item enumeration, B source, epilogue row map); compiles on CPU | **chosen** |
| (b) new persistent kernel from scratch (CUDA C++/CUTLASS or CuTe DSL) with host-loader warps/CTAs | about 1 week | new MMA/TMA/TMEM plumbing, layout and numerics to re-derive | no; (a) *is* (b) minus the plumbing |
| (c) two co-scheduled TRT-LLM launches on green-context SM partitions + fused finalize | measured | lost at ≤ 12 cold/layer (two calls + add = 0.15–0.25 ms fixed, hot call loses SMs statically) [M] | rejected |
| DeepGEMM Mega-MoE + host-aware scheduler | about 1 week | needs a second layout for every expert (HBM full) or a full switch of prefill and plugin; C++ JIT internals | reference only |

**Why this design avoids the earlier failures:**
* **MOE_SPLIT (two streams) was slower** because both TRT-LLM calls spread over all SMs, the cold call used a shallow tactic and parked SMs at about 200 GB/s, and the split added a second routing, a second finalize and an add.
  * Here there is one routing (the front), one GEMM1 and one GEMM2. The cold work is confined to `kc` CTAs with 6- or 10-stage TMA pipelines, about 96 KiB in flight per SM on GEMM1, against about 64 KiB needed for the 51 GB/s per-SM TMA cap at about 1.1 µs host latency [M]. There is one finalize, or none when vLLM defers it.
* **The green-context split lost at 12 or fewer cold experts** because of its fixed overhead (two calls plus an add) and a static SM partition, in which the hot call lost SMs even when the cold call had finished.
  * Here the partition costs nothing: it is the CTA index inside one launch.
  * It is sized per call: `kc = 0` when no cold expert is touched, and `kc` shrinks with the cold work.
  * Cold CTAs that finish their cold items take hot items (`hoc`), so no SM idles.
  * In the simulated sweep (§3.4), the new schedule loses only with 0 cold experts and an assumed HBM rate of 5.0 TB/s: +1 % at T = 64 and +6 % at T = 6 [S]. That case means the new kernel streams HBM less efficiently than TRT-LLM, which stage (ii) checks ("never slower at 0 cold").

## 3. Design

### 3.1 Data layout (no second copy of any weight)

* **Weights.** These are exactly the tensors vLLM prepared for TRT-LLM-gen, re-backed by the plugin's VMM ranges. HBM and host slots are fixed per layer by the plan's segments; adaptive swaps move contents, not addresses.
  * `w13` is `[384, 4608, 2560]` u8: E2M1 values, K-major, low nibble = even k.
  * Rows are interleaved (row 2i = w3/up_i, row 2i+1 = w1/gate_i), then shuffled within blocks of 32: stored row `s` holds row `(s & ~31) | ((s & 7) << 2) | ((s >> 3) & 3)`.
  * `w2` is `[384, 5120, 1152]`, shuffled the same way.
  * This was checked against `flashinfer.utils.get_shuffle_matrix_a_row_indices(..., 128)` and against vLLM's `convert_weight_to_mxfp4_moe_kernel_format` TRTLLM branch [V].
* **Block scales.**
  * `w13_weight_scale` is `[384, 4608, 160]` and `w2_weight_scale` is `[384, 5120, 72]`. Both are UE8M0 bytes, row-shuffled the same way, in the 128x4 layout: `off(m, j) = ((m//128)·(SFK/4) + j//4)·512 + (m%32)·16 + ((m%128)//32)·4 + j%4` [V].
  * **The scales always stay in HBM.** The plugin permutes them in place and re-backs only the two weight tensors, so a cold item's scale loads are HBM-latency.
* **Clamp.** `gemm1_clamp_limit` is FP32 `[384]`, per slot. There are no expert biases, no alpha and no beta [V: plugin inventory logs].
* **Per-call buffers.** These are allocated with `torch.empty` and come from the graph pool under capture. At T = 64 they total about 39 MB; at T = 128, about 46 MB. Only about 1 GiB of HBM is free, so watch it. `M = max_tiles(T)`, which is 384 at T = 64.
  * `xp [8M, 5120]` u8: MXFP8 input rows, permuted expert-major and padded to 8-row tiles, gathered by the front.
  * `xpsf [M, 20480]`: their UE8M0 scales in the per-tile 128-row atom layout, the same layout as GEMM2's HS, so GEMM1's B path is GEMM2's B path (TMA for B and SFB).
  * `h [8M, 2304]` (as u16 pairs) and `hs [M, 9216]`: GEMM1 output and GEMM2 input, as in the original kernel.
  * `rows [T·6, 5120]` bf16: **compact** GEMM2 output (route order), which is the `UnfinalizedMoEOutput.gemm2_permuted` contract.
* **Routing record v2** (int32; `route_ref.py` is the spec, `persist_front.py` the kernel). `C = round_up(M, 4)`.
  * Header: `[0]` padded rows, `[1]` tiles, `[2]` cold tiles, `[3]` T, `[4..8)` GEMM1 `kc, hoc`, `[8..12)` GEMM2 `kc, hoc`.
  * Tables of `C` entries each:
    * tile → expert slot
    * tile → padded row limit
    * tile → compact row
    * `order`: position → tile, cold tiles first, each class in ascending expert order
  * Then padded row → token.
  * Tiles are expert-ascending, with each expert's rows in ascending token order and padded to 8, as TRT-LLM routing does.
  * `permuted[t, k]` is the compact row, or −1.

### 3.2 Work decomposition and schedule

* **Items.**
  * GEMM1 item = (tile, m-tile of 128 rows of w13); 36 per tile, each 20 TMA k-steps of 16 KiB A + 1 KiB SFA + 2 KiB B + 1 KiB SFB.
  * GEMM2 item = (tile, m-tile of w2); 40 per tile, each 18 k-steps of 8 KiB + 0.5 KiB + 1 KiB + 0.5 KiB.
  * Item list order is `[cold tiles × m-tiles | hot tiles × m-tiles]`, tile-major. 36 or 40 consecutive items of one expert go to different CTAs at the same time, so HBM rows are fetched once (L2) and host experts are streamed as contiguous ranges.
* **Static split (v1).**
  * Cold CTAs (`bid < kc`) take cold items `bid, bid+kc, …`, then `hoc` hot items from the end of the hot list with the same stride.
  * Hot CTAs take the remaining hot items with stride `G − kc`.
  * Every warp evaluates `schedule()` and `item()` identically. These are CuTe `@cute.jit` helpers that are also mirrored in `route_ref.cta_items`, where coverage is proven by test [V].
* **Choosing `kc`.**
  * The front evaluates, for `kc ∈ [4, 16]` (`MOE_PERSIST_KC`): `t_cold = bytes/min(cta_bw, host_bw/kc)`, `t_hot = bytes/min(cta_bw, hbm_bw/(G−kc))`, cold phase `ceil(cold_items/kc)·t_cold`, and the join time `(hot·t_hot + kc·tc)/G`.
  * It keeps the minimum makespan and derives `hoc`.
  * The defaults are 385 GB/s host, 6.0 TB/s HBM and 46 GB/s per CTA (`MOE_PERSIST_BW`); calibrate them in stage (iii).
  * `kc = 0` when there are no cold tiles.
  * The split is computed separately for GEMM1 and GEMM2.
* **Cold loads go first.** Cold CTAs start on cold items in the first microseconds. With the GEMM2 PDL launch, low block indices (the GEMM2 cold CTAs) get the first SMs that GEMM1 frees.
* **Front and finalize.**
  * The front is one Triton launch of 32 programs. Each program recomputes the routing (a 384-bin histogram and scans) and gathers its own slots' rows; program 0 writes the tables and the split. There is no inter-CTA sync.
  * GEMM1 waits on the front with `griddepcontrol.wait`.
  * Finalize (T > 16) is one Triton launch: `out[t] = Σ_k w·rows[permuted]` in fp32, ascending k.

### 3.3 Pipeline depth, SMEM, TMEM, registers

| | GEMM1 | GEMM2 |
|---|---|---|
| tile / cta_k / k-steps per item | 128×8 / 256 / 20 | 128×8 / 128 / 18 |
| TMA bytes per stage (A+SFA, B+SFB) [V] | 17 408 + 3 072 | 8 704 + 1 536 |
| stages (default) → staged smem [V] | 6 → 120 KiB | 10 → 100 KiB |
| A bytes in flight per CTA | 96 KiB | 80 KiB |
| registers / thread (256 threads) [V cubin] | 40 | 34 |
| TMEM columns | 128 (acc 2×8 + SFA + SFB) | 128 |

* **Host side.** About 64 KiB in flight per SM reaches the 51 GB/s TMA per-SM cap at about 1.1 µs host latency [M], so 6 stages leave margin for jitter.
  * The host ceiling (385–390 GB/s) is reached with 8–16 CTAs [M].
  * With `kc = 16`, each cold CTA gets about 24 GB/s.
* **HBM side.** 136 CTAs × 46 GB/s ≈ 6.3 TB/s of capability against the 6.0–6.6 TB/s HBM rate seen in `mix` [M].
* **Co-residency.** GEMM1 + GEMM2 come to 120 + 100 + about 4 KiB (static + reserved), which is at most 228 KiB. GEMM2 CTAs can therefore be co-resident during GEMM1's tail, and its weight streaming then starts earlier [A]: check with ncu's "Block Limit Shared Mem". If they do not fit, use stages 6 + 9 or 5 + 10.

### 3.4 Expected bandwidths and time per layer

* **Calibration [S].** `sched_sim.py` is a fluid model: tiers shared evenly, per-CTA caps, and additive tiers. Fitted to the measured TRT-LLM standalone times at T = 64 with 0, 6, 12, 24 and 48 cold experts, it reaches 6.9 % rms error with 5.6 TB/s HBM, 300 GB/s host and 22 GB/s per cold CTA.
* **New schedule, synthetic (T = 64, 175 experts) [S].** Host 385 GB/s, HBM 5.5 TB/s (in parentheses: 6.0):

| cold experts | 0 | 6 | 10 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| TRT-LLM model (µs) | 679 | 715 | 790 | 882 | 1081 | 1314 | 1565 |
| host-aware static (µs) | 628 (582) | 605 (561) | 602 (557) | 593 (592) | 787 | 982 | 1177 |
| host-aware dynamic (µs) | 628 | 605 | 592 | 593 | 775 | 958 | 1142 |

* **Real routing [S].** ShareGPT decode distribution of every layer (routing-v1.npz), plan-e207-g8 slots, 2 samples per layer; dynamic schedule with HBM 5.5 TB/s:
  * **T = 64:** 13.2 cold / 175 distinct per layer; 1087 → 718 µs (−34 %); 40 layers: 43.5 → 28.7 ms.
  * **T = 32:** 660 → 449 µs (−32 %).
  * **T = 6:** 169 → 126 µs (−26 %).
  * The model overestimates TRT-LLM by about 10 % (43.5 against a measured 38.8 ms per step).
* **Anchored to measurements:** the c32 step loses about 6–9 ms of about 45 ms; **c32 +15–20 %, c8 +12–18 %, c1 +8–15 %**.
* **What the result depends on.** Above about 15 cold experts per layer, the layer is host-bound regardless of the kernel, at 46 µs per expert at 385 GB/s [S, M]. Gains there come from 277 → 385 GB/s, so the convex per-layer budgets of plan-e207 still matter. The biggest unknown is the HBM rate the CuTe kernel reaches at 64 tokens: 5.0 TB/s halves the gain at 0–6 cold.

### 3.5 Numerics and the correctness bar

* **Same arithmetic as TRT-LLM-gen `t128x8` non-split-K tactics:**
  * MXFP4 × MXFP8 block-scaled MMA, fp32 accumulation in ascending K;
  * the clamped SwiGLU / UE8M0 / E4M3 PTX, copied verbatim;
  * BF16 RN for the GEMM2 rows.
* **Unit level (stage i):**
  * The per-route GEMM2 rows should be **bit-identical** to TRT-LLM with a non-split-K GEMM2 tactic. They may differ by 1–2 bf16 ulps against split-K tactics (e.g. 8:468).
  * Finalize may differ by 1 ulp: Triton fp32 FMA order against TRT-LLM's finalize.
  * Check independently against `ref_layout.ref_moe`, a CPU fp32 kernel-faithful reference: cosine above 0.9999.
* **Model level (stage iv; the task's bar, since bit-exactness is not required):**
  * Teacher-forced prompt logprobs: **mean |Δ| < 0.01 and max < 0.5** against the TRT-LLM path at identical chunking.
  * Greedy decode must stay coherent. Divergences are allowed only at near-ties (`verify/deccmp.py` prints the probabilities at the divergence).
* **Making prefill hit the kernel.** Prefill chunks are larger than 128 tokens and would take the TRT-LLM fallback, so the logprob check runs a server with `--max-num-batched-tokens 128 --max-num-seqs 8` (8 × 6 spec rows ≤ 128) twice, `MOE_PERSIST=0` then `1`. Run `tf_logprobs.py record` on the first and `compare` on the second.

### 3.6 CUDA graphs, streams, PDL

* **Capture safety.**
  * Every launch runs on the current stream with shapes that are static per captured batch size.
  * The record and the split are computed on the GPU (front), so no host readback happens.
  * The CuTe kernels are compiled once at bind time, outside capture, and take shapes as runtime Int32s.
  * The Triton front and finalize specialise per T. **Warm every capture size at bind** with a dummy call (TODO in `persist_moe.py`).
* **Interaction with the adaptive cache.**
  * Host-mask bytes are static per slot.
  * Adaptive swaps happen between steps on the same stream and only exchange row contents, so the kernel always reads the slot's current expert.
  * The adaptive cache's counting wrapper must stay outside the persistent wrapper; `patch_apply` must be installed **before** `_patch_adapt`.
* **PDL.** front (plain) → GEMM1 (PDL; all warps `griddepcontrol.wait` before reading the record) → GEMM2 (PDL; only its B warp waits, because the record is final once GEMM1 passed its wait — the original kernel's argument) → finalize (plain).
* **Persistent grid.** It is G = #SMs. A side-stream kernel (shared expert overlap) can delay some CTAs. With the static split those CTAs' items finish late (tail) [A]. Measure this in the server trace; v2's dynamic queue removes it.

### 3.7 Plugin integration (`persist_moe.py` → copy into `tuning/plugin/moe_placement/`)

* **Environment:**
  * `MOE_PERSIST=1` (needs `MOE_PLACEMENT=apply`)
  * `MOE_PERSIST_MAX_TOKENS=128`
  * `MOE_PERSIST_STAGES=6,10`
  * `MOE_PERSIST_KC=4,16`
  * `MOE_PERSIST_BW=385,6000,46`
  * `MOE_PERSIST_LAYERS=all|a-b,...` (for bisecting)
  * `serve-tune.sh` already passes `MOE_*` through.
* **Hooks:**
  * `register()`: if `MOE_PERSIST=1`, call `persist_moe.patch_apply(log)` **before** `_patch_adapt(log)`.
  * `run()`: after the placement loop, call `persist_moe.bind(placed, E, dev, log)`. That attaches `fe._moe_persist`, with the host mask from the layer's segments.
* **Wrapper.** It takes the call when `expert_map is None`, `a1q_scale` is MXFP8 and `T ≤ MAX`. It honours vLLM's deferral: `should_defer_moe_finalize(T)` → return `UnfinalizedMoEOutput(rows, topk_weights, permuted)` (compact rows); otherwise it finalizes into `output`. Everything else goes to the original `apply`: prefill chunks above 128 tokens, the DSpark draft MoE (128 experts, not placed), and unplaced layers.
* **To verify at bind:** `a1q_scale` is linear `[T, 160]` (not swizzled or padded), `w1_scale is fe.w1_scale` permuted with the weights, and `gemm1_clamp_limit` is FP32 `[384]`.

### 3.8 Later versions (only if stage iii shows the need)

* **v2, dynamic queue.**
  * The idle warp 3 becomes a scheduler warp: it claims items from two global atomic counters (cold, hot; reset by the front) and publishes them to a 4–8-slot smem ring with mbarriers, the way DeepGEMM's `MegaMoEScheduler::publish_task` / `get_next_task` do.
  * Cold CTAs drain cold items, then steal hot; hot CTAs drain hot, then steal cold.
  * This removes the model dependence and the late-CTA tail (simulated gain 0–3 %).
* **v2, J-runs.** An expert with more than 8 tokens re-reads its weights once per 8-row tile. From L2 for HBM experts that is fine; from host it costs real bandwidth. Let one A stage feed J ≤ 4 N=8 MMAs, with B and SFB stages for J tiles and TMEM 256 columns.
* **v3, one launch.** GEMM1 and GEMM2 items go in one persistent grid, with per-tile H-ready counters that the front zeroes: GEMM2's B warp spins on `ld.acquire` and then `fence.proxy.async` before its TMA. That removes the boundary (about 5–15 µs per layer) and lets cold CTAs stream w2 while their H is produced. Deadlock-free only with claim-order queues (v2).
* **Die locality.** B300 has two dies. Measure host bandwidth per die using `%smid`-based roles; DeepGEMM's `get_sm_locality_domains` gives the map. Place `kc` CTAs on the die nearer the C2C port if it differs.

## 4. Staged plan (GPU needed from here; stop dsv41-moe first and restore it at the end)

**(i) Standalone correctness: 0.5–1 day.**
1. Command: `MODE=exact TS=1,6,8,32,64,128 COLDS=0,1,6 ~/dsv41/tuning/kernel/run_py.sh ../persist/src/pbench.py`.
2. The script mounts tuning at `/work`; `pbench.py` adds `/work/persist/src` and `/moe/plugin` to `sys.path`.
3. Expected first-run fixes: CuTe pointer dtypes, Triton specialisation, the `UnfinalizedMoEOutput` import.
4. **Pass:** per-route rows bit-identical to TRT-LLM at non-split-K tactics, otherwise 2 bf16 ulps or fewer; CPU-reference cosine above 0.9999; no NaN; host-resident experts give the same bits as HBM-resident ones.
5. **If GEMM1 mismatches,** first test GEMM2 alone with a synthetic H checked against `ref_layout` (TRT-LLM does not expose its H). Then check the `xpsf` atom layout, the most likely bug, against the `route_ref` offsets.

**(ii) Full MoE layer standalone: 0.5 day.**
1. Command: `MODE=time TS=6,24,32,48,64 COLDS=0,1,3,6,10,16,24`.
2. Compare against TRT-LLM with the production tactic table, using the same g8 cold layout as plan-e207.
3. **Pass:** T = 64 with 10 cold ≤ 0.65 ms against about 0.77; T = 6 with 0–1 cold ≤ about 0.10 ms against about 0.14; never slower at 0 cold.
4. Sweep `MOE_PERSIST_STAGES` (5–8, 8–12) and `MOE_PERSIST_KC` max (8, 12, 16, 24).

**(iii) ncu / nsys: 0.5 day.**
1. Use `kbench.py`'s ncu flow (`kernel/ncu_kb.sh`) with `MODE=prof`. The kernel names contain `RoutedExpertGemm1` / `RoutedExpertGemm2`.
2. Check:
   * the HBM rate of the hot CTAs (aim ≥ 5.5 TB/s);
   * host reads (`syslts__t_sectors_aperture_peer_lookup_miss`; aim ≥ 370 GB/s while the cold phase runs);
   * per-SM TMA bytes, to confirm the partition;
   * stall reasons in the epilogue;
   * the front's duration (aim ≤ 5 µs, or reduce programs / move it to CuTe);
   * whether GEMM2 is co-resident.
3. Calibrate `MOE_PERSIST_BW` from the measured rates.
4. Decide on v2/v3 from the tail and boundary gaps (an nsys trace with `--cuda-graph-trace=node`).

**(iv) Server integration: 0.5–1 day.**
1. Copy `persist_moe.py`, `persist_moe_cute.py` and `persist_front.py` into `tuning/plugin/moe_placement/`, wire `register()` / `run()`, and add bind-time warm-up for all capture sizes.
2. Correctness: run the §3.5 logprob A/B, then `cache-exp/check.sh` (decode correctness, the 210k-token prompt and memory).
3. Performance: `cache-exp/run-both.sh` with the moe-cand env plus `MOE_PERSIST=1`, at the same args as `kernel/kq.sh` (`g8tac`), **twice**, against two baseline runs (noise ±2.5 %).
4. Bisect regressions with `MOE_PERSIST_LAYERS`.
5. Restore production at the end: `docker rm -f dsv41-tune; ~/dsv41/moe/serve-moe.sh`, then wait for `/health` to return 200.

## 5. Risks and mitigations

| risk | mitigation |
|---|---|
| First GPU run of a modified CuTe kernel hangs: wrong expect-tx bytes for the new SFB TMA in GEMM1, or the barrier count | `tma_bytes_b` now includes SFB (17 408 / 3 072 printed [V]); run under `compute-sanitizer --tool racecheck` at T = 6 first; timeouts in pbench |
| HBM rate at 64 tokens below 5.5 TB/s (kernel designed for 8 rows or fewer) | more stages, `cta_k` 512 for GEMM1 (t128x8x512 tactics exist and are bit-identical [M]), J-runs |
| Static split mis-modelled, or late CTAs from side streams | calibrate `MOE_PERSIST_BW`; v2 dynamic queue |
| Front kernel too slow at c1 (recomputation in 32 programs) | fewer programs; or port to CuTe like the image's fused front (it already fuses top-k + quant + record for 8 rows or fewer) |
| Workspace HBM (about 46 MB at T = 128) with about 1 GiB free | buffers come from the graph pool; shrink `xpsf` (only 8 of 128 atom rows used) if needed |
| `a1q_scale` layout differs from linear `[T, 160]` | assert at bind; the front can read any 2-D stride |
| Persistent kernel + spin waits (v3) deadlock | only with claim-order queues (v2); v1 has no spin waits |

## 6. Files (all on 224, `~/dsv41/tuning/persist/`)

**Helpers:**
* `cpu_py.sh <script>`: runs a script from `src/` in the vLLM image **without GPUs**, with `CUTE_DSL_ARCH=sm_103a`.

**`ref/`:** copies of the image's `specialized_decode/` (with `moe/`), `trtllm_mxfp4_moe.py`, DeepGEMM `include/` + `mega/`, `deep_gemm_mega_moe.py` and `prepare_megamoe.py`.

**`src/` (each item says what was checked on CPU):**
* `ref_layout.py`: TRT-LLM MXFP4 layout ↔ logical, SF 128x4, the UE8M0/E4M3 epilogue math, and `ref_moe`, a CPU reference. Self-test [V].
* `route_ref.py`: record v2 spec, `choose_split`, `cta_items`. Property tests [V].
* `sched_sim.py`: fluid schedule simulator. Modes `cal`, `synth` and `plan <hbm> <Ts>` [S].
* `make_persist_cute.py` → `persist_moe_cute.py`: the host-aware TP1 GEMM1/GEMM2, made by exact edits of the image file. Each edit must match exactly once; re-run after an image change.
* `compile_persist.py`: compiles both for sm_103a and prints the resources [V].
* `compile_probe.py`: shows that the original kernels compile on CPU [V].
* `persist_front.py`: the Triton front and finalize.
* `test_front.py`: `run` (interpreter, against `route_ref`, 12 trials [V]) and `compile` (sm_103a [V]).
* `persist_moe.py`: the binding plus the plugin hooks (not yet wired into the plugin).
* `pbench.py`: the stage (i)/(ii)/(iii) GPU bench (never run).
* `pycheck.py`: byte-compiles everything [V].

**Commands to re-run the CPU checks:**
```
cd ~/dsv41/tuning/persist
./cpu_py.sh ref_layout.py; ./cpu_py.sh route_ref.py; ./cpu_py.sh make_persist_cute.py
CUTE_DSL_NO_CACHE=1 CUTE_DSL_KEEP=cubin CUTE_DSL_DUMP_DIR=/tmp/pd ./cpu_py.sh compile_persist.py
TRITON_INTERPRET=1 CPUS=16 ./cpu_py.sh test_front.py run 12; ./cpu_py.sh test_front.py compile
CPUS=16 MEM=24g ./cpu_py.sh sched_sim.py plan 5.5e6 64,32,6
```
