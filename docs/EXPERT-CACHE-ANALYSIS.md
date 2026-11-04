> **Provenance:** engineering notes written during the TidalMoE tuning work (Nov 2026) on the GB300 test machine.
> Paths such as `~/dsv41/...`, `~/dsv41/tuning/...` and `~/dsv41/moe/...` refer to that original workspace;
> in this repository the production pieces live in `serve.sh`, `plugin/`, `plans/`, `bench/`, `correctness/` and `tools/`.

# Expert cache (placement + adaptive swapping): what is left to tune

## Summary
- **What limits throughput:** the number of distinct DRAM-resident experts touched per forward (D). In today's measurement, the plugin's count of DRAM bytes matches the Grace C2C performance counter (PMU) almost exactly: 50.9 vs 50.6 GB/s at conc 1, 127 vs 129 at conc 8, 142 vs 143 at conc 16, and 170.0 vs 170.3 at conc 32. So minimizing D is the right goal for the cache.
- **The host-memory path is limited by latency, not bandwidth.** At conc 32 the PMU's 1-second peak is 187 GB/s. The LPDDR5X limit is 396 GB/s (DGX Station spec) and the link is 447 GB/s per direction, so neither is close to full. The cold tiles in the GEMM simply don't keep enough reads in flight.
- **Each DRAM touch costs about 45 µs of step time (range 35-60 µs).** Each HBM touch costs about 3 µs. I estimated this three ways:
  - Today's conc 32 run against the README run: 501 vs 431 DRAM experts per forward, 51.5 vs 48.5 ms per step, so about 43 µs each.
  - Static plan vs adaptive cache at conc 8 and conc 32: about 45-59 µs.
  - A linear fit across concurrencies: about 37 µs.

  At conc 32 the DRAM experts are only 6-7% of the experts touched but take 40-45% of the step.
- **Throughput estimates below** are change in D × 45 µs ÷ step time. The baseline is the README's conc 32 run: D = 431, step = 48.5 ms.

## Ranked candidates (ShareGPT bench, conc 32, cumulative in this order except #1)

| # | Candidate | Simulated change in D (held-out data) | Throughput gain at conc 32 | Confidence | Effort |
|---|---|---|---|---|---|
| 1 | Cross-layer prefetch into a small HBM staging cache (Fate-style gate predictor, SM copy kernel, expert-id remap) | hides 30-70% of the remaining D | **+15%** (+7 to +24%) on top of #2-#4 | low until the predictor is measured | high |
| 2 | Free HBM for about 272-320 more hot experts: KV headroom (+64), `embed.weight` to host (+64), GPU memory utilization 0.99 (+144), vision encoder off for text-only traffic (+48) | -15% (+272 experts), -18% (+320) | **+6.4%** (+5 to +9%); +8% together with #4 | high | low-medium |
| 3 | Score by decode touches: count distinct experts per MoE call, skip steps with more than 128 tokens | -9.7% at the bench's prefill:decode ratio of 0.23; -22% at 1:1; -35% at 4:1 | **+4%** (+3 to +5.5%); +9 to +16% for prefill-heavy production traffic | high | trivial |
| 4 | Re-plan per-layer budgets from persisted adaptive scores (warm start) | -4.2% (ShareGPT); -6 to -10% when traffic doesn't match the plan | +1.7%; +3 to +5% for mismatched traffic | high | low |
| 5 | Move cold experts' fp8 scales to host (+144 experts, but +4.4% bytes per cold touch) | -3.8% net | +1.6% | medium | medium |
| 6 | Asynchronous epochs (no `cuda.synchronize`), batched swaps, minimum residence time | – | +0.5 to +1.5% | medium | low |
| 7 | Put cold blocks first in slot order (longest jobs start first) | 0 | 0 to +3%, unknown; plan-file-only experiment | low | trivial |
| 8 | Re-tune dspark draft length k at conc 32 | k=2 raises D per step by 45% | only pays if the second draft token's acceptance is above about 40% | low | env only |
| 9 | Co-activation-aware grouping | 0 | about 0 (the kernel reads each expert independently) | high | – |
| 10 | Expert substitution / cache-aware routing | large | +30-50%, but changes model outputs and breaks the bit-identical prefill check; excluded | – | – |

**Items #2, #3, #4 and #6 together:**
- At conc 32: simulated D goes 369 → 333 → 318 → 270, a 27% drop. Applied to the measured 431, that is about -5.2 ms per step, or **+12% (+9 to +17%)**.
- At conc 8: about +9%. At conc 1: about +4%.
- Adding the vision encoder and cold scales: D -33%, about +15%.
- All of these only change placement, so prefill logprobs stay bit-identical.
- The zh and code simulations show larger gains, but their prefill samples are tiny (1.4k-1.9k tokens), so treat those numbers as optimistic.

**What the simulations say about the specific questions:**
- **Touch scoring vs selection counts:**
  - With independent tokens, ranking experts by touches gives the same order as ranking by counts. The real problem is that the live counter includes prefill tokens, each counted 6 times.
  - That pulls the hot set toward experts used by prompts. Prefill steps touch almost every expert regardless of placement.
  - At conc 32 the 64 tokens in a step behave as about 63.5 independent tokens, so the independence model is reasonable there.
- **Churn of about 45 swaps per epoch:**
  - I replayed the exact "z" swap policy on steady, randomly sampled traffic. It converges within 13 epochs to within 0.4% of the ideal hot set, then makes about 0.2 swaps per epoch.
  - Changing half-life, margin, z or the swap cap doesn't change that.
  - So the live churn comes from real traffic changes and bursty requests, not from sampling noise.
  - The live log shows the measured DRAM share (3.0-4.9%) slightly above what the cache predicts (3.0-3.5%), meaning the cache lags the traffic.
- **Per-layer budgets:** per-expert and per-16-block allocation give the same result (-4.8% vs -4.6%). The loss comes only from budgets planned for a different traffic mix.
- **Splitting w13 and w2 across tiers:** no gain; keeping whole experts together is optimal.
- **dspark verification tokens:** already counted as ordinary tokens. The draft model's MoE has its own 128 experts, so its routing can't be reused.

**Where the HBM goes:**

| Item | Size | Note |
|---|---|---|
| Hot experts | 196.7 GiB | |
| Scales for all 15,360 experts | 15.8 GiB | all in HBM |
| dspark experts + scales | 6.3 + 0.4 GiB | used every step; keep |
| `embed.weight` (bf16 [129280, 5120]) | 1.23 GiB | only gathered: about 0.6 MB read per step at 64 tokens; can move to host |
| `head.weight` | 1.23 GiB | read in full every step; keep |
| Vision encoder + aligner | about 0.92 GiB | also profiled with 8 max-size images at startup |
| KV cache | 3.32 GiB (1.72M tokens) | 1M context needs only 2.03 GiB |
| CUDA graphs | 1.96 GiB | |
| Activation peak (my estimate) | about 6.6 GiB | |

One 16-expert block is 0.264 GiB; 1 GiB holds about 60.7 experts.

## Design A: free HBM for more hot experts, and re-plan per-layer budgets (#2 + #4)

**A1 – move the embedding to host.**
- Add `_host_rebind(model, dev, log)` at the end of `run()`, before `_AdaptiveCache` is created.
- Select parameters by regex from env `MOE_HOST_PARAMS`, default `embed_tokens\.weight$`. Also require shape [129280, 5120], bf16, on CUDA. Never match `lm_head` / `head.weight`.
- For each match:
  1. `_alloc_segments(nbytes, [(0, nbytes, "host")], dev)`
  2. `new.copy_(p.data)`
  3. `p.data = new`
  4. `gc.collect()` / `empty_cache()` / `_host_emptyCache()`
  5. Log free GPU memory before and after.
- This runs before memory profiling and graph capture, and the address never changes, so CUDA graphs stay valid.
- The dspark draft model loads after this hook. Wrap `GPUModelRunner.load_model` and log any [129280, 5120] CUDA parameter still in HBM; if there is one, re-bind it too.
- Values are bit-identical, so `tf_logprobs.py compare` must still pass.

**A2 – server knobs.**
- `GPU_UTIL=0.99` gives +2.5 GiB. It needs a 1-hour soak at conc 32 plus one 256k-token prompt, watching for out-of-memory.
- `TEXT_ONLY=1` → `--limit-mm-per-prompt '{"image":0}'`, only if images are not needed. Confirm the "Encoder cache will be initialized" log line disappears.
- Then set the DRAM budget so "Available KV cache memory" lands at **2.15-2.25 GiB**. vLLM fails to start below 2.03 GiB.
  - Formula: `n_cold = 3424 - 16*floor((KV_observed - 2.2)/0.2637)`.
  - Expect about 3088 cold experts (193 blocks) with the vision encoder off, about 3152 (197 blocks) with it on.

**A3 – budgets from scores.**
- **Save scores:**
  - `_AdaptiveCache._save_scores` writes `/moe/state/scores.npz` atomically (tmp file + `os.replace`) every `MOE_ADAPT_SAVE_EVERY` epochs (default 100).
  - Content: per-step EMA rates indexed by logical expert id, `logical[li, order[li]] = score[li]/ema_steps`.
  - This needs a read-write mount `-v ~/dsv41/moe/state:/moe/state` in `serve-moe.sh`.
- **Build the plan:**
  - `_plan_from_scores(rate, n_cold_total)` ports the block logic from `analyze_routing.py`.
  - It takes the globally coldest 16-expert blocks and gives each layer its coldest `16*nb[l]` experts.
  - It keeps cold blocks spread at positions `int((i+0.5)*24/nb[l])` and orders hot experts hottest first.
- **Startup:**
  - In `run()`: use `MOE_PLAN_SCORES` + `MOE_DRAM_EXPERTS` if the scores file exists, otherwise fall back to `MOE_PLACEMENT_PLAN`.
  - Before any scores exist, run `analyze_routing.py` (decode counts) with budget = `n_cold*16.875/1024` GiB.
- **Not yet:** moving blocks between layers at runtime is feasible (VMM remap, 2 MiB-aligned, addresses fixed) but not worth it until re-planning at restart has been measured.

**Validation:**
- Bit-identical `tf_logprobs` and the decode check (`correctness.py`).
- `measure3.sh`: D per forward should drop about 17-20% for A alone, about 27% with B.
- `bench.sh` on ShareGPT and zh at conc 1, 8, 32.

## Design B: decode-touch scoring, async epochs, no ping-pong, epoch dump (#3 + #6)

**B1 – counting.** This replaces the `index_add_` block in `_patch_adapt.apply`:

```python
c = getattr(self, "_moe_counts", None)
if c is not None and hidden_states.size(0) <= _MAX_TOK:   # static per graph/eager call -> capture-safe
    n_e = c.numel()
    ids = topk_ids.reshape(-1).long()
    ids = torch.where((ids >= 0) & (ids < n_e), ids, n_e)  # padded/invalid rows -> dummy slot
    hit = self._moe_hit                                     # persistent fp32 [n_e+1]
    hit.zero_()
    hit.index_fill_(0, ids, 1.0)
    c.add_(hit[:n_e])
    if self._moe_layer0:
        self._moe_steps.add_(1.0)
```

- `_MAX_TOK` comes from `MOE_ADAPT_MAX_TOKENS`, default 128 (the largest CUDA-graph size). Large prefill steps run outside graphs and are skipped.
- The Python condition is checked per captured size at capture time and per call at runtime, so it is correct under graph replay.
- It launches about as many small kernels as today. If nsys shows they cost time, fuse them into one Triton kernel.

**B2 – swap policy.**
- Score = EMA of touches. The existing z test stays valid: touches have variance no larger than their mean, so the test is conservative.
- Replace the traffic floor `tot < 6*40*64` with "fewer than 30 counted steps".
- Minimum residence time: track `moved[L,E]` by slot and skip a swap if either slot moved within `MOE_ADAPT_MIN_RES` epochs (default 5).

**B3 – async epoch (remove both `torch.cuda.synchronize()` calls).**
- At the epoch boundary in `on_step`, queue on the current stream:
  1. Non-blocking copy of `counts` and `_moe_steps` into a pinned host buffer.
  2. `counts.zero_()`.
  3. `ev.record()`; set `pending`.
- On a later step, once `ev.query()` is true: run the numpy policy, then queue batched swaps per layer on the current stream:

  ```python
  tmp = t.index_select(0, ia)
  t.index_copy_(0, ia, t.index_select(0, ib))
  t.index_copy_(0, ib, tmp)
  ```

  This also works on host-backed rows.
- **Why it is safe:** the hook runs before `orig()` queues the forward, so the swaps land between two forwards in stream order.
- **Check first:**
  - That the v0.33 model runner has no private forward stream: log `current_stream()` in the hook and in the first captured `apply`.
  - That any auxiliary MoE stream forks from the main stream after the swaps (nsys).
  - If either check fails, keep one `ev.synchronize()` before queuing the swaps.
- Expected: epoch overhead drops from about 50-60 ms per 3 s to under 10 ms.

**B4 – epoch dump.** `MOE_ADAPT_DUMP=/moe/state/epochs-<pid>.bin` appends per epoch: fp16 [L,E] touches by logical id, counted steps and a timestamp. That is 30 KB per epoch, about 36 MB per hour. With it, half-life, z, margin and residence can be tuned offline on real traffic, and staging-LRU hit rates can be estimated.

**Expected:**
- -9.7% D on the ShareGPT bench: about +4% at conc 32 and conc 8.
- Prefill-heavy traffic: -22% D at 1:1 and -35% at 4:1, about +9-16%.
- About 0 on the zh bench, whose prompts are only about 15 tokens.

## Design C (largest upside; measure before building): cross-layer prefetch into an HBM staging cache

**Gate experiment, about 1 GPU-hour:**
- At layer i, compute layer i+1's gate on layer i's input: `gate_{i+1}(x_i)` with the real bias and sqrtsoftplus scoring.
- Measure recall over cold-slot selections only, for top-6 and top-10 predictions.
- Proceed if recall is at least 50% with at most 1.5x over-fetch.
- Fate reports 97% accuracy, but on smaller models without this model's 4-stream mHC residual.

**Mechanism:**
1. **Staging slots:**
   - Extend each layer's w13/w2 range to E+S slots (S=16).
   - Even layers' staging slots map one shared physical HBM buffer, odd layers another (VMM aliasing).
   - Cost: 540 MiB for weights + 675 MiB for [E+S] scale rows ≈ 72 fewer hot experts.
2. **Predict at layer i, inside the graph:**
   - A tiny GEMM, bias, top-k'.
   - Keep cold candidates not already staged, up to S.
   - Write `remap_{i+1}` and a copy list.
3. **Copy:**
   - An SM copy kernel with 8-16 thread blocks on a side stream (fork/join with events inside capture).
   - It must be an SM kernel because `cudaMemcpyAsync` inside a graph can't use data-dependent addresses.
   - The link and LPDDR5X have more than 200 GB/s of headroom.
4. **At layer i+1:**
   - Join the copy.
   - `ids = where(remap[ids] >= 0, E + remap[ids], ids)`.
   - Call the kernel with `num_experts = E+S`, which triggers one FlashInfer autotune run.
5. **Misses** read from host exactly as today, so results stay bit-identical.

**Estimate:**
- After A and B, D is about 305. Hiding 30-70% of it saves 4-9.6 ms per step.
- Overhead is about 1.4 ms.
- Result: **+7 to +24% at conc 32**, central +15% at 50% recall.

## What this analysis can't answer yet
- **Request-level locality.** It decides whether churn helps, the best half-life, and whether a fast LRU staging tier pays: copy-on-miss doubles C2C traffic, so it needs above 50% reuse. Answering it needs per-token traces. `collect_routing.py` already receives per-request (tokens, 40, 6) arrays; save them instead of only summing them, and combine with the B4 dumps.
- **The exact per-layer cost of a DRAM touch.** Measure it in the nsys phase. Check:
  - Whether layer time behaves like max(HBM part, DRAM part) – then gains stop below about 6 DRAM experts per layer – or like a sum.
  - Whether cold tiles form a tail at the end of each GEMM (that would support #7).
  - Kernel-side fixes such as L2 prefetch of cold tiles, deeper pipelines or cold-first ordering. These lower every gain above, so re-rank after any kernel change.

## Literature

Most of these papers assume a PCIe box, where experts must be copied before use. On this box the GPU reads cold experts in place over coherent C2C, and the LPDDR5X isn't saturated. What carries over is the right scoring objective (touches), capacity matched to the workload, and overlap driven by prediction.

- Fate, cross-layer gate prefetch, about 97% accuracy: https://arxiv.org/html/2502.12224v2
- MoE-SpeQ, prefetch during speculative verification: https://arxiv.org/html/2511.14102v1
- SP-MoE, speculative-decoding-aware prefetch: https://arxiv.org/html/2510.10302v2
- MoE-Infinity, request-level activation tracing: https://arxiv.org/html/2401.14361v2
- FineMoE, fine-grained offloading with semantic hints: https://arxiv.org/pdf/2502.05370v2
- HybriMoE, score-based caching and inter-layer prefetch: https://www.arxiv.org/pdf/2504.05897
- DALI, workload-aware greedy assignment: https://arxiv.org/html/2602.03495v1
- ExpertFlow: https://arxiv.org/html/2410.17954v1
- ExpertFlow (cache-aware routing): https://arxiv.org/abs/2510.26730v1
- PreScope: https://arxiv.org/html/2509.23638v1
- SMoE, expert substitution (#10): https://arxiv.org/pdf/2508.18983
- DGX Station GB300 specs (LPDDR5X 396 GB/s, HBM3e 7.1 TB/s): https://www.storagereview.com/review/msi-xpertstation-ws300-review-748gb-of-coherent-memory-and-20-petaflops-on-a-desk
- GB300 specs overview: https://pi3g.com/nvidia-gb300-specifications-including-memory-bandwidth-and-llm-benchmarks-based-on-2026-systems/