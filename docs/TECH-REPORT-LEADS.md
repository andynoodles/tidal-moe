> **Provenance:** engineering notes written during the TidalMoE tuning work (Nov 2026) on the GB300 test machine.
> Paths such as `~/dsv41/...`, `~/dsv41/tuning/...` and `~/dsv41/moe/...` refer to that original workspace;
> in this repository the production pieces live in `serve.sh`, `plugin/`, `plans/`, `bench/`, `correctness/` and `tools/`.

# DeepSeek-V4.1 tech report: mechanisms we do not exploit yet (GB300, TP1)

Written 2026-11-03 from CPU-only code reading. No GPU was used.

- **Code:** image `sha256:d825baf9ce4b…` (vLLM v0.33.0). Paths are relative to
  `/usr/local/lib/python3.12/dist-packages/vllm/`.
- **Report:** `DeepSeek_V41_Tech_Report.pdf` (§ = report section, p = page).
- **Baseline:** production `~/dsv41/moe/serve-moe.sh` (plan.json, adaptive cache) and the best tuning config in
  `kernel/kq.sh` (g8tac: plan-e207-g8 + MOE_TACTICS).
- **Numbers marked "est."** are estimates from the code plus the BOTTLENECKS.md and kbench measurements. They have not
  been measured.

## Summary (in recommended order for the GPU agent)

| # | Mechanism (report §) | Verdict on our server | Expected gain on our box | Effort |
|---|---|---|---|---|
| **M** | **Mega-MoE + Mega-Gate**, the persistent DeepGEMM MoE kernel (§3.2 p18) | **Not used.** It needs `moe_backend=deep_gemm_mega_moe` + `--enable-expert-parallel`. The kernel ships in the image and supports a single rank. | c32: up to about +20% (est.) if its dynamic task queue hides the host tiles. **The sign is unknown until it is benchmarked standalone.** | M1 standalone bench: 0.5 day. Plugin port: 1-2 days. |
| **2** | Single-Pass mHC / Mega-mHC (§2.4.1) | **Partly used.** Steps with ≤16 tokens in FULL graphs (c1, c2) take a 4-kernel overlap path tuned for GB200 TP4. All other steps already use `sm100_mega_mhc_impl`. | c1: −0.6 to −1.0 ms of 13.2 ms/step, **+5-8%** (est.). c≥3: 0. | One-line monkeypatch |
| **3b** | "11 kernels per Reuse layer" fused decode: the persistent CuTe-DSL low-batch MoE (§3.2) | **Not used.** The specialized decode is TP4-only, and its MoE GEMMs are hard-coded to I=576/640. | c1: about −2.4 ms/step, **+15-20%** (est., BOTTLENECKS §6.3) | Port: about 2 days |
| **6** | CED + SWA bounded replay (§2.2, §3.2.2) | **Active** (log). The placement does not take it into account: production plan.json puts *more* cold experts in the encoder layers (1920 vs 1504). | Long-prompt TTFT −25 to −50% (est.) from bigger prefill chunks and an encoder-light plan. c32 decode +3-6% (est.) from equalizing cold experts touched per layer. | Flags plus plan generator |
| **1** | DSpark confidence-scheduled verification (§2.4.3) | **Not used.** The built-in profiler records a cost curve **without routed-MoE cost**: its dummy tokens are padding, so their top-k ids are −1. | c8-c32 +2-5% after the cost curve is fixed (est.). c1 about 0. **Raising the per-batch draft caps without the fix: about −10% at c32.** | Plugin: about 1 day |
| **7a** | FP4 main KV cache (§2.4.4) | **Not used.** Our explicit `--kv-cache-dtype fp8_ds_mla` overrides mega-attention's default `nvfp4_ds_mla`. | About 0.6 GiB of HBM freed, or about 35 more hot experts: +0.5-1% at c32 (est.) | Flag plus KV/plan re-budget |
| **7b** | MXFP4 indexer K, sparse logits, DeepSelect (§2.3.2) | The candidate pool is used (dense logits + mask). The MXFP4 cache and sparse logits are not. | ShareGPT: about 0 (+0.17 GiB). Large for long-context (≥32K) decode. | Flags |
| **4** | Engram background prefetch (§2.4.2) | **Already used.** Lookups run on a side stream with half the SMs and overlap layer 0. | About 0. `use_thp` is optional. | – |
| **5** | Modality-specific routing biases (§2.1.1) | **Used, and correctly permuted by our plugin. No correctness bug.** | – | One image smoke test |
| 7c | Reasoning-effort control (§5.1.4) | Set by the client. The default is `high` (75). | Changes output length, not engine tok/s | – |

The relayed user request asks for a **resident ("persistent") MoE kernel**: about 0.79 → 0.60 ms/layer.
- Lead **M** is that kernel, already shipped in the image as DeepSeek's own design.
- Lead **3b** is a second persistent MoE (CuTe DSL, editable Python source) that reads exactly the TRT-LLM tensors our
  plugin already places. It is the cheapest sandbox for host-aware scheduling.

Benchmark both before writing a kernel from scratch (recipes in M3 below).

---

## M. Mega-MoE (DeepGEMM persistent MoE) + Mega-Gate: not used, top lead for the persistent-kernel work

### Evidence
- **Report §3.2 p18:** "the Mega-Gate, Mega-mHC, and Mega-MoE kernels in DeepGEMM … Reuse Mode layers execute with
  only 15 kernels during prefill and 11 during decode".
- **Selecting it:**
  - `config/kernel.py:313`: `"deep_gemm_mega_moe"`.
  - `models/deepseek_v4/nvidia/model.py:227-236`: `use_native_mega_moe` raises unless `--enable-expert-parallel` is
    set. The DSv4.1 MoE subclasses this code.
  - TP1 → EP group of 1.
- **Shared expert fused into the same launch at TP1:** `models/deepseek_v4/nvidia/model.py:374-392`
  (`fuse_shared_experts = … tp_size == 1`).
- **Gate:**
  - ≤16 tokens: `fused_topk_bias`.
  - >16 tokens: `bf16_mega_gate(gate.weight, bias, image_bias=bias_vl, …)`.
  - See `models/deepseek_v4/nvidia/model.py:467-516`.
- **Experts class:** `model_executor/layers/fused_moe/deep_gemm_mega_moe.py`.
  - `MEGA_MOE_MXFP4` = `fp8_fp4_mega_moe`: MXFP8 activations × MXFP4 weights, shared expert MXFP8 (:146-154).
  - Parameters are `w13_weight` [E, 4608, 2560] int8 and `w2_weight` [E, 5120, 1152] int8, plus int32 UTCCP
    scales:
    - `w13_weight_scale` storage [E, 40, 4608], viewed transposed.
    - `w2_weight_scale` storage [E, 18, 5120], viewed transposed.
    - See :433-449.
  - Checkpoint shards are written straight into the kernel layout, with gate/up interleaved by 8 rows (:585-605).
  - `capture_fn(topk_ids)` hook: :762.
- **Single rank works:** `third_party/deep_gemm/mega/__init__.py:59` has
  `allocator = torch if group.size() == 1 else symm_mem`.
- **Scheduler** (`third_party/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`):
  - Persistent 2-CTA clusters (BLOCK_N 128 per CTA).
  - Tasks are taken **dynamically** from global atomic counters (:348, `atomic_add(task_count_ptr…)`).
  - Linear1 and Linear2 tasks of different experts are interleaved after L1 warm-up waves. An L2 K-block starts as
    soon as its L1 N-blocks are done (`L2KBlockDependency`, :143-165).
- **Offload matching is by name segment** (`model_executor/offloader/uva.py:102-111`). In mega mode the parameters
  are `…ffn.experts.w13_weight`, so the flag must be `--cpu-offload-params experts.w13_weight experts.w2_weight`.
  `routed_experts.*` does not match.
- **Mega-mHC from the same library already runs** on this GPU at c32 (`sm100_mega_mhc_impl`, BOTTLENECKS §1), so
  the DeepGEMM JIT works on SM103.

### Why it fits our gap
BOTTLENECKS #1 found three things:
- CTAs are dispatched in expert order, and CTAs that hit host tiles park their SMs.
- GEMM1 and GEMM2 are separate kernels, so GEMM1's host-read tail cannot overlap GEMM2's HBM reads.
- Routing, finalize, MXFP8 quantization and the shared expert are separate kernels: about 1.5-2 ms per c32 step.

Mega-MoE addresses each one:
- A pair stalled on a host task does not block the queue. Other pairs keep pulling HBM tasks.
- L2 of early experts runs while L1 of later experts is still streaming.
- Gate+top-k, quantization, shared expert and combine run in one launch.

### Risks
- L2 tasks spin-wait on their L1 producers, so pairs can idle behind host-bound L1 tasks.
- At 64 tokens most experts get only 1-2 tokens, which wastes BLOCK_M.
- The numerics differ: L2 activations are FP8 here, whereas TRT-LLM's GEMM2 takes BF16. Expect a small, not bit-exact
  difference.
- The symmetric buffer is sized for `max_num_batched_tokens` = 8192 × top-6. Estimated 0.4-0.6 GiB of HBM.
- Our plugin's discovery and counting are tied to `TrtLlmMxfp4ExpertsModular`.

### Recipe
**M1. Standalone, decides go/no-go.** Write `kernel/kbench_mega.py`, a copy of `kbench.py`. Run it with
`kernel/run_py.sh kbench_mega.py`.

1. Set up a single-rank group:
   `torch.distributed.init_process_group("nccl", init_method="tcp://127.0.0.1:29511", rank=0, world_size=1)`, then
   `g = dist.group.WORLD`.
2. `dg = vllm.utils.deep_gemm._import_deep_gemm()` loads the vendored DeepGEMM.
3. `buf = dg.get_symm_buffer_for_mega_moe(g, 384, 8192, 6, 5120, 2304, num_shared_experts=0 /*then 1*/, mma_type="fp8xfp4", activation="swiglu")`.
   Print `buf.buffer.nbytes`.
4. Weights: random bytes are enough for timing.
   - `w13` uint8 [384, 4608, 2560] and `w2` [384, 5120, 1152], allocated with kbench's VMM layout:
     `layout("8")`, `NCOLD=80`, the same `build()`.
   - Scales in HBM:
     `w13_sf = torch.randint(…int32).view(384, 40, 4608).transpose(1, 2)` and
     `w2_sf = (384, 18, 5120).transpose(1, 2)`.
   - Fill each scale byte with values near 127 (UE8M0 ≈ 1.0).
5. Inputs:
   `vllm.model_executor.layers.fused_moe.prepare_megamoe.prepare_megamoe_inputs(h_bf16, topk_w, topk_ids, buf.x[:T], buf.x_sf[:T], buf.topk_idx[:T], buf.topk_weights[:T])`.
   Use the default `hidden_quant=kMxfp8Dynamic` and kbench's `routing(C, hot, cold, seed)`.
6. Call `dg.fp8_fp4_mega_moe(y, (w13, w13_sf), (w2, w2_sf), buf, activation_clamp=10.0, fast_math=True)`.
   Capture prepare+kernel in a CUDA graph and time it with kbench's `timed()`.
7. Sweep the same T and C grid as `cx-T*.log`:
   - T=6: C 0/1/3.
   - T=32: C 0/3/8.
   - T=64: C 4/10/18.
   - Also T=96 and T=128.
8. Repeat with `num_shared_experts=1` (MXFP8 shared weights [4608, 5120]/[5120, 2304] + 1x32 UE8M0 scales). Compare
   against TRT-LLM + 2 shared GEMMs (8.2 + 6.2 µs at c1).
9. Optionally run ncu or the nsight scripts on the T=64/C10 case to see whether host and HBM reads overlap inside the
   kernel.

**Go criterion:**
- T=64/C=10 ≤ 0.70 ms. TRT-LLM is 0.79 ms, plus about 0.05 ms of routing, quantize, finalize and shared-expert
  kernels that Mega-MoE replaces.
- T=6/C=1 ≤ 0.20 ms.

**M2. Server, if M1 passes.** Add to the serve-tune extra args:
`--enable-expert-parallel --kernel-config '{"moe_backend":"deep_gemm_mega_moe"}' --cpu-offload-params experts.w13_weight experts.w2_weight`.

The plugin needs `MOE_MEGA=1` handling:
- **a. Discovery:** `m.experts` is `DeepGemmMegaMoEExperts`. Its per-expert tensors are:
  - big: `w13_weight`, `w2_weight` (`_BIG` names unchanged);
  - small: `w13_weight_scale`, `w2_weight_scale` (transposed int32 views; `index_select` + `copy_` handles the
    strides), plus `gate.weight`, `gate.e_score_correction_bias`, `gate.bias_vl`.

  There is no `gemm1_clamp_limit`: the clamp is the scalar `activation_clamp`.
- **b. Counting:** set `m.experts.capture_fn = partial(_count_fused, c=counts[i], touch=_COUNT_TOUCH)`. It is called
  inside forward, so it is captured into the graphs. Drop the `TrtLlmMxfp4ExpertsModular.apply` patch for these layers
  and do not set `MOE_TACTICS`.
- **c.** The DSpark MoE (128 experts) is already skipped by the `n_routed_experts` check.

Then run `run-both` with g8 + adapt (sg 1,8,32 + zh 32, ×2) and `check.sh`. Expect logprob deltas within
run-to-run noise, not bit-exactness.

**M3. Our own persistent kernel, only if M1/M2 still show host stalls.** Start from the DeepGEMM headers under
`third_party/deep_gemm/include/deep_gemm/{scheduler,impls}/…mega_moe…`. They are JIT-compiled at run time, so a
modified copy can be mounted over them.
- Split the task queue into a **host queue** and an **HBM queue**. The plugin passes a per-slot host bitmap; it
  already knows the segments.
- Pin about 8 CTA pairs to the host queue with deeper TMA stages. The rest work the HBM queue.

This is the measured `mix` microbench configuration (16 SMs reading host at 389 GB/s while 136 SMs read HBM at
6.6 TB/s), i.e. the 0.60 ms/layer "ideal overlap" target.

For small T, apply the same idea to the CuTe-DSL kernel in 3b (Python source, simpler to edit).

---

## 2. Single-Pass mHC / Mega-mHC: partly used; c1-c2 take the slower 4-kernel overlap path

### Evidence
- **Report §2.4.1 p12-13:** one fused kernel reads and writes the residual once, half the traffic of the original
  4-kernel implementation.
- **`models/deepseek_v41/nvidia/ops/mhc.py:37`:** `MHC_OVERLAP_MAX_TOKENS = 16`, commented "GB200 TP4/FlashInfer
  improves through 16 tokens".
- **`ops/mhc.py:44-57`:** `supports_mhc_overlap` is true on SM10x with DeepGEMM, so `model.py:751` creates
  `mhc_stream` at TP1 too.
- **`model.py:509-516`:** the overlap path is used only when capturing FULL graphs with ≤16 tokens. Otherwise
  `mhc_stream=None`.
- **Overlap path, `ops/mhc.py:256-282`:** four kernels per boundary, two boundaries per layer:
  - `mhc_post_tilelang`;
  - the `MHC_PRE_NORM_KERNEL` "input" epilogue;
  - on the side stream, `tf32_hc_prenorm_gemm` + the "stats" epilogue.
- **Otherwise, `ops/mhc.py:284-303`:** Mega-mHC (`deep_gemm.mega_mhc`, `third_party/deep_gemm/_C.py:360`).
  - Requirements (`ops/mega_mhc.py:17-43`): hidden % 1024 == 0, hc_mult 4, not `capture_aux`.
  - The `capture_aux` exception covers the DSpark aux layers 37-39, which fall back to TileLang fused.
- **Fused all-reduce + mHC** (`ops/cute_dsl/all_reduce_mhc.py`) is TP4-only (`supports_mhc_all_reduce`,
  `ops/mhc.py:59-72`). It does not apply at TP1, where there is no all-reduce.
- **Measured (BOTTLENECKS §1, §6.3):**
  - c1: mHC 1.79 ms/step, 321 kernels, grid 6 (one CTA per token).
  - c32: 1.15 ms via `sm100_mega_mhc_impl`, 152 CTAs. It splits the hidden dimension across CTAs, so its parallelism
    does not depend on the token count.
- c2 = 2×6 = 12 tokens, so it also takes the overlap path. c3 = 18 tokens already uses Mega-mHC.

### Recipe (E2)
Add to the plugin's `register()`:
```python
if os.environ.get("MOE_MHC_MEGA") == "1":
    import vllm.models.deepseek_v41.nvidia.model as m41
    m41.supports_mhc_overlap = lambda vllm_config: False   # no mhc_stream -> Mega-mHC for every token count
```
- `model.py` imported the name, so it must be patched there. The plugin loads before model init.
- Equivalent alternative: `m41.MHC_OVERLAP_MAX_TOKENS = 0`.

Run `kq.sh`-style: `run-both kt-mhcmega 1,2,8,32 - <g8tac BASE> MOE_MHC_MEGA=1`, twice, plus `check.sh`.
- With `nsys --cuda-graph-trace=node`, confirm that c1 now runs `sm100_mega_mhc_impl` instead of `mhc_pre_big_fuse…`
  (grid 6).
- Expected: c1 +5-8% tok/s, c2 +2-4%, c≥3 unchanged.
- Prefill is unchanged; it already runs piecewise, i.e. Mega-mHC.

---

## 3. "11 kernels per Reuse layer": which fused paths are disabled at TP1, and what porting takes

### Evidence
The specialized decode is enabled by `additional_config={"decode_backend":"latency_specialized"}` (`model.py:1638`).
Its requirements are listed in `models/deepseek_v41/nvidia/specialized_decode/__init__.py:75-130`:

| Requirement | Status on our server |
|---|---|
| SM103 | ✓ |
| Model Runner V2 | ✓ |
| **TP == 4** | ✗ |
| DSpark **without** adaptive verification | ✓ |
| **FLASHINFER_MLA_SPARSE_DSV41** backend + fp8 cache + **`indexer_kv_dtype=mxfp4` + `indexer_sparse_logits`** | ✗ |
| FULL decode graphs | ✓ |
| **DeepGEMM + NVLink multicast for the fused all-reduce + mHC** | ✗ |

Missing items make it raise at startup. It is a whole-step executor built around the TP4 all-reduce.

Also, `__init__.py:43` pins `FLASHINFER_VERSION = "0.7.2"`, but the image has `0.7.2.post1`. So
`uses_checked_flashinfer()` disables these sub-features even at TP4:
- persistent KV counters;
- pinned MXFP8 tactics;
- the publishing `wo_b`;
- the fused MXFP8 input.

### Portable piece: the low-batch routed-expert MoE
Files: `specialized_decode/moe/routed_expert_gemm.py`, `…_gemm_cute.py`, `…_front_cute.py`, `…_route_cute.py`.
- **Kernels:** a persistent CuTe-DSL `tcgen05` 128×8 GEMM1 and GEMM2 (CTA b takes items b, b+G, …). PDL lets GEMM2's
  weights stream before GEMM1 finishes. A fused "front" kernel does DSv4 top-k + MXFP8 quantization + routing record
  in one launch.
- **Exactness:** the output is bitwise equal to TRT-LLM `t128x8`.
- **Inputs:** the **same TRT-LLM-prepared w13/w2 tensors and scales**, so our VMM hot/cold placement works unchanged.
- **Hard-coded for TP4** (`routed_expert_gemm.py:76-87`): `SLICE=576`, `PADDED=640`, `MAX_ROWS=8`.
  `check_weights` (:424-440) expects (E, 2·PADDED, H/2).
- **At TP1:** w13 is [384, 4608, 2560] and w2 is [384, 5120, 1152], so SLICE = PADDED = 2304. That is 18×128, so there
  is no padding tile. The kernel constructors take these as arguments (`RoutedExpertGemm1(HIDDEN, SLICE, PADDED)`,
  :376-403).
- **Missing at TP1:** the output is an `UnfinalizedMoEOutput` meant for the fused AR+mHC. At TP1 it needs a small
  finalize kernel: weighted top-k sum of `gemm2_permuted` rows + shared output.

### Recipe (E3b, about 2 days)
1. Plugin `MOE_LOWBATCH=1`: set `routed_expert_gemm.SLICE = PADDED = 2304` before
   `build_routed_expert_gemms([l.ffn.experts for l in layers])`, after placement.
2. Wrap `TrtLlmMxfp4ExpertsModular.apply`:
   - for 1 ≤ T ≤ 8: front (or route) → gemm1 → gemm2 → finalize;
   - otherwise: the original path.
3. Validate bitwise against TRT-LLM with forced tactic t128x8 in kbench (MODE=exact style).
4. Bench c1/c2 twice.

Expected: c1 MoE 5.6 → about 3 ms/step (−2.4 ms, BOTTLENECKS §6.3), +15-20% at c1.

The work list is static round-robin, so host-aware ordering is an easy edit here, e.g. put host items first on
dedicated CTAs. That is a cheap test of the persistent-kernel idea at small T.

### Other report fusions
- Mega-attention (fused RoPE + attention + inverse RoPE + cast) is already used: `FLASHMLA_MEGA_ATTN_DSV41`.
- Mega-mHC: lead 2.
- Mega-MoE / Mega-Gate: lead M.
- `kernel_config.monokernel` is AMD-only (`config/kernel.py:299`).

---

## 1. DSpark confidence-scheduled verification: not used; the built-in cost curve is wrong for our box

### How vLLM implements it
**Flag:** `SpeculativeConfig.enable_adaptive_verification` (`config/speculative.py:585`), dspark only.
- It needs the confidence head (`v1/worker/gpu/spec_decode/dspark/speculator.py:106`). The checkpoint has
  `mtp.2.confidence_head.proj.weight`.
- **Profiling happens once, at startup, inside `capture_model`** (`v1/worker/gpu/model_runner.py:1140-1148`). There
  is no online update.
  - Batch shapes: every captured token count (our 1…128) plus tail sizes 192, 256, 512, 1024, 2048, 4096 and 8192.
  - Each is replayed 5 times through `_dummy_run(num_tokens, context_len=VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=8192)`
    (`adaptive_verification.py:211-233`; `envs.py:1103`), with num_reqs = min(tokens, 32).
  - Medians give the verify curve (forward ms by target tokens) and the draft curve (drafter ms by requests, FULL-graph
    samples only) (:235-277).
  - Cost tables (:80-128): a step function up to the capture limit, interpolated above it.
- **Per step** (`model_runner.py:1352` → `get_num_tokens`, `adaptive_verification.py:305-373`):
  1. Take the previous step's confidences and compute survival = cumprod.
  2. Sort all slots globally.
  3. Pick the budget b that maximizes
     (num_sampling_reqs + Σ top-b survival) / (draft_cost[num_reqs] + verify_cost[non_draft + b]) (:367).
  4. The GPU admits the top-b slots by survival (:46-72) and compacts `query_start_loc` (:375-488).
- **Composition with our `num_speculative_tokens_per_batch_size`:** the scheduler still proposes k(bs) drafts per
  request, and adaptive verification only **trims** within that cap. To let it *add* drafts at c≥5, the caps must be
  raised.
- **CUDA graphs** (`resolve_adaptive_cudagraph_mode`, :36-43): FULL_AND_PIECEWISE stays.
  - Decode graphs become **varlen** graphs, one per capture size, with num_reqs = min(size, 32) and max_query_len = 6
    (`v1/worker/gpu/cudagraph_utils.py:302-313`). They replace today's per-tier uniform graphs (44 FULL graphs).
  - Every backend we use declares ALWAYS:
    - mega-attention builder: `models/deepseek_v41/sparse_mla.py:257`;
    - SWA FlashMLA builder: `models/deepseek_v41/nvidia/flashmla.py:39-42`;
    - V4.1 indexer: `v1/attention/backends/mla/indexer.py:992`; the log shows `supports_varlen=True`;
    - compressor.

    So the manager should build (`adaptive_verification.py:491-552`).
- **Plugin compatibility:**
  - The execute_model wrapper skips dummy runs, and the counters ignore −1 ids: OK.
  - Steps of 65-192 tokens fall outside the MOE_TACTICS buckets; their autotuned tactics apply.
  - Raise `MOE_ADAPT_MAX_TOKENS` to 192, or touch counting skips those steps.

### Problem
The profiling batches are pure padding:
- `InputBatch.make_dummy(..., is_padding=not is_profile)` (`model_runner.py:1895`) with zero input_ids
  (`input_batch.py:182`).
- With `VLLM_MOE_SKIP_PADDING=1` (default, `envs.py:1642`), the router masks them
  (`fused_topk_bias_router.py:28-33,179`) and `dsv4_topk` writes `topk_ids=-1` (`dsv4_topk.py:111-114`).

So **no routed expert runs while profiling**. The verify curve therefore lacks routed-MoE cost, which is about 75% of
our c32 step and its steepest part. kbench shows MoE going from 0.435 to 0.79 ms/layer between T=32 and T=64, i.e.
about **+0.44 ms per extra token per step** over 40 layers.

`randomize_inputs` (not used here) would overshoot instead. Random tokens touch about 243 distinct experts at
64 tokens, about 21% of them in host memory, versus 5-7% for real traffic.

### What our cost curve implies (est.)
| | step | marginal cost | admit a draft slot only if survival exceeds about | effect |
|---|---|---|---|---|
| c32 | 45 ms @ 64 tokens | about 0.5 ms/token | 0.5 × 1.25 tok/ms ≈ **0.6** | 2nd/3rd drafts go only to high-confidence requests; weak 1st drafts are dropped. Expected **+2-5%**. |
| c1 | 13 ms | — | about 0.1 | All 5 drafts are admitted: no change. |

### Recipe
**E1a: compatibility and overhead, no code.**
- `SPEC='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,8,3],[9,16,2],[17,32,1]],"enable_adaptive_verification":true}'`.
- Run sg 1,8,32 twice.
- Expected:
  - About ±0 throughput: caps are unchanged and the underestimated cost means it trims little.
  - Startup about +1-2 min for the 7 eager tail sizes up to 8192 tokens.
- Any regression means the varlen graphs cost something.
- `serve-tune.sh` forwards only `MOE_*` variables, so add `-e VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` if it is
  changed.

**E1b: fix the curve in the plugin (`MOE_AV_CURVE=online`), then raise the caps.**
1. Wrap `AdaptiveVerificationManager.set_initial_cost_curves`: keep the draft curve and replace the verify curve with
   a seed curve, either from `file:/tuning/av_curve.json` or roughly `verify_ms(n) ≈ 9 + 0.55·n` for n ≤ 64.
2. Online, in the plugin's 3-s epoch. vLLM's `StepTimingCollector` cannot be reused: `forward_end`, `drafter_*` are
   called only in `_dummy_run` (`model_runner.py:945-986`), and real steps record only `forward_start` (:2045-2048).
   1. In the plugin's existing `execute_model` wrapper, record a CUDA event on the current stream at entry. Store it
      with the step's (`total_num_scheduled_tokens`, number of requests).
   2. The GPU is about 96% busy, so start(i) → start(i+1) is the full step cost.
   3. Each epoch, drain the events whose `query()` is True and keep an EMA of step ms per (tokens, requests).
   4. Set verify_curve[n] = step_ms − draft_curve[num_reqs], with the draft curve from the startup profile. That
      curve has the same padding bias, but the drafter's MoE is only 3 layers × top-3 of 128 experts.
   5. Call `runner.adaptive_verification.set_cost_curves(draft, verify)`. It is cheap, and `broadcast_object` is a
      no-op at TP1.

   This is the report's "profiled engine throughput curve under current load".
3. Then:
   - SPEC caps `[[1,8,5],[9,32,3]]` + `enable_adaptive_verification`;
   - capture sizes add 80, 112, 160 and 192 (max 32×6);
   - `MOE_ADAPT_MAX_TOKENS=192`;
   - bench sg 1,8,16,32 + zh 32, twice.

**Do not raise the caps with the built-in curve.** It would admit almost every draft: est. about −10% at c32 (128 tokens
per step for about 82 expected tokens).

---

## 6. CED + SWA bounded replay: active; implications for expert placement and prefill

### Evidence
- **Report:** §2.2 p9 and §3.2.2 p20.
- **Default:** `config/cache.py:367` `swa_bounded_replay = True`.
- **Production log** (`model.py:886`, 10:22:16): "Decoder SWA bounded replay: in eager prefill steps, layer 20 writes
  its KV for every token, and the rest of it and layers 21-39 run on each request's last 128 tokens only."
- Mechanism: `models/deepseek_v41/decoder_replay_layers.py`.
- SWA KV is kept out of prefix caching, so a prefix hit replays only 128 tokens.

### Implications
1. **Prefill MoE work belongs almost entirely to the encoder layers 0-19.**
   - Decoder layers 21-39 see at most 128 rows per request per prefill step. Layer 20 only writes KV.
   - Each long-prompt chunk is ≤6144 tokens because of `--long-prefill-token-threshold 6144`. It touches all 384
     experts in every encoder layer and therefore **reads every encoder cold expert once per chunk**:
     - plan.json: 1920 cold × 16.875 MiB = 31.6 GiB;
     - plan-e207-g8: 1648 cold, 27.2 GiB.

     That is about 70-110 ms per chunk at 280-390 GB/s.
   - Measured: the 210,797-token prompt took 4.1 s = 35 chunks × 117 ms (`results/kt-g8tacx-check.log`).
   - So **host reads are about 60-90% of long-prompt TTFT** (est.). Compute per encoder layer is only about 0.5 ms per
     6144-token chunk.
2. **Decode:** every layer costs the same per token, but cold experts touched per layer per c32 step range from 21.1
   (L18) to 4.5 (L39) (BOTTLENECKS #2).
   - Production `plan.json` gives the encoder **more** cold experts: 1920 vs 1504.
   - plan-e207-g8 is balanced by count (1648 / 1664), not by touch rate.
   - The adaptive cache only swaps within a layer, so it cannot fix this.

### Recipes
**E6a: bigger prefill chunks, flags only.**
- Two configurations:
  - `--long-prefill-token-threshold 8192`;
  - `--max-num-batched-tokens 16384 --long-prefill-token-threshold 16384`.
- Measure:
  - `check.sh` long-prompt time (210k);
  - a 64k and a 128k prompt TTFT;
  - sg c32 decode. Watch "placed … GPU free": bigger activation buffers take HBM from the experts.
- Expected long-prompt TTFT: −25% at 8192 and −40 to −50% at 16384 (est., if host-bound as computed).

**E6b: per-layer "water-filled" cold budget in `analyze_routing.py`.**
- Choose cold groups per layer so that the predicted cold experts touched per c32 step (from routing npz or saved
  adaptive EMA scores) are equal across layers, near the about 9.5 hiding threshold.
  - Cost model, BOTTLENECKS #2: a layer costs about max(605 µs, 64 µs × cold touched).
  - Add a penalty on encoder layers for prefill.
  - Keep the same total DRAM budget.
- First cut: move about 8 groups of 8 per layer from the hottest encoder layers (2-18) to decoder layers 22-39.
- Expected: c32 +3-6%, long-prompt TTFT −10 to −20% (est.).
- Minor: touch counting also counts the ≤128-row decoder replay calls of prefill steps. Restricting counting to
  pure-decode steps is optional.

---

## 7a. FP4 main KV cache: not used, because our flag forces FP8

### Evidence
- **Report §2.4.4 p14:** E2M1 values with one E4M3 scale per 16, no global scale, trained with QAT in post-training.
- **Mega-attention default:** `models/deepseek_v41/nvidia/flash_mla_mega_attn.py:204-207` sets
  `packed_kv_cache_dtype = "nvfp4_ds_mla"`. `auto` and `fp8` resolve to it (`models/deepseek_v41/attention.py:140-160`).
- **Record size:** the compressed record is 288 B/token instead of 528 B (`attention.py:525-527`). The SWA record stays
  MXFP8.
- **Our setting:** serve-moe.sh and serve-tune.sh pass `--kv-cache-dtype fp8_ds_mla`. The log says "Using DeepSeek's
  fp8_ds_mla KV cache format."

### Gain (est.)
- Compressed main KV is about 2.5 entries/token: layers 2/8/14 at ratio 2 give 0.5 each, layer 20 at ratio 1 gives 1.
- 528 → 288 B saves about 600 B/token, about **0.59 GiB** of the 2.10 GiB budget for 1.05M tokens.
- That is room for about 35 more hot experts (about 1% fewer cold experts, the hottest of them), plus a little less
  attention KV traffic.
- About +0.5-1% at c32.

### Recipe (E7a)
1. Append `--kv-cache-dtype nvfp4_ds_mla`. argparse keeps the last occurrence.
2. Read "GPU KV cache size: N tokens".
3. Set `--kv-cache-memory-bytes = 2254857830 × 1,085,568 / N`.
4. Give the freed HBM to experts: regenerate the plan with the DRAM budget lowered by the same amount.
5. Run `check.sh`. Expect small tf_logprob shifts versus FP8. The 210k prompt must still pass. Add a needle test at
   200k+ tokens.

## 7b. Indexer: MXFP4 cache, sparse logits, DeepSelect (Hierarchical Sparse Indexer §2.3.2)

What we run today:
- The candidate pool from layer 20 is used. Indexers 24/28/32/36 compute dense logits over the whole context and mask
  them.
- Log: "Using FP8 indexer cache", `use_fp4_cache=False`.

Options:
- `indexer_kv_dtype="mxfp4"`: Blackwell, allowed on SM10x (`v1/attention/backends/mla/indexer.py:56-100`).
- `indexer_sparse_logits=true`: requires mxfp4 (`models/deepseek_v41/attention.py:1430-1440`;
  `config/attention.py:93-110`). The docs say it is slower below about 32K context.
- `--kernel-config '{"sparse_indexer_topk_backend":"deep_select"}'` (`config/kernel.py:351`;
  `_deepselect_C.abi3.so` is present).

Recipe for 1M-context traffic only:
`--attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`.
Measure decode TPOT at 128k-1M contexts. ShareGPT gain is about 0, plus about 0.17 GiB of KV freed.

## 7c. Reasoning effort (§5.1.4)

- **Implementation:**
  - `tokenizers/deepseek_v41.py:88-111`: `reasoning_effort` comes from `chat_template_kwargs`.
  - `tokenizers/deepseek_v41_encoding.py:191-226`: the prompt prefix is "Reasoning Effort: {b} (range 1-100, …)", in
    thinking mode only.
- **Values:** `low`=50, `high`=`xhigh`=75, `max`=100, or an integer 1-100. `"none"` means non-thinking. The default
  is **high (75)**.
- **Client:** `"chat_template_kwargs": {"reasoning_effort": "low"}`. Verify through the Rust frontend once, by
  checking the rendered prompt via `/tokenize` or `return_token_ids`.
- **Effect:** it changes output length, not engine tok/s. Benchmarks must keep the same effort.

## 4. Engram prefetch (§2.4.2): already used

`models/deepseek_v41/common/engram.py`:
- **Side stream:** with `cpu_offload`, every Engram layer gets a prefetch stream (:1665-1678).
- **Start:** `prepare_embeddings` is called for both modules before layer 0 (`nvidia/model.py:1043-1060`).
  `_start_prefetch` (:1691-1715) runs `lookup(background=True)` on the side stream using half the SMs (:1212).
- **Inline case:** only batches of ≥4096 tokens look up inline (:98).
- **Wait:** each layer waits only for its own event.

The FULL graph is split at `@eager_break_during_capture` points. Measured Engram cost: 0.17 ms at c1, 0.39 ms at c32,
mostly hidden. Gain about 0.

Optional, low value: `--engram-config '{"cpu_offload":true,"use_thp":true}'` (:1123) to cut GPU TLB misses on the
189 GiB table. With 64 KiB base pages, THP pages are 512 MiB. Check
`/sys/kernel/mm/transparent_hugepage/enabled` first.

## 5. Modality-specific routing biases (§2.1.1): loaded, and permuted correctly. No bug.

- **Config:** `transformers_utils/configs/deepseek_v41.py:52-53` sets `vision_n_layers = 32`.
- **Parameter:** `models/deepseek_v4/nvidia/model.py:311-318` creates `gate.bias_vl` [384] fp32 on every MoE layer.
  The checkpoint has `layers.{0..39}.ffn.gate.bias_vl` (and `mtp.*`).
- **Routing:** image-sentinel tokens use `bias_vl` (`dsv4_topk.py` HAS_VL path; `fused_topk_bias_router.py:327-341`).
  The router holds the same Parameter object (:408).
- **Plugin:**
  - `_expert_groups` permutes every dim0==384 CUDA tensor under the MoE module, including the gate submodule (tuning
    plugin :269-274; production :219).
  - Production log 10:25:19: `permuted ['weight', 'e_score_correction_bias', 'bias_vl', 'w13_weight_scale', 'w2_weight_scale', 'gemm1_clamp_limit', 'w13_weight*', 'w2_weight*']`.
  - The adaptive cache swaps "8 per-expert tensors/layer", `bias_vl` included.
- **Under Mega-MoE (M):** `bias_vl` reaches `bf16_mega_gate(image_bias=…)` as the same Parameter, so it stays correct.
- **Smoke test:** one image chat request, comparing top-logprobs between production and `MOE_PLACEMENT_IDENTITY=1`.
  The difference should be within decode noise.
