> **Provenance:** engineering notes written during the TidalMoE tuning work (Nov 2026) on the GB300 test machine.
> Paths such as `~/dsv41/...`, `~/dsv41/tuning/...` and `~/dsv41/moe/...` refer to that original workspace;
> in this repository the production pieces live in `serve.sh`, `plugin/`, `plans/`, `bench/`, `correctness/` and `tools/`.

# DeepSeek-V4.1 hot/cold MoE expert placement (GB300, single GPU)

`serve-moe.sh` starts DeepSeek-V4.1-Flash-1017 as container `dsv41-moe` on :8000 with the `moe_placement` vLLM plugin.
Current version: 2026-11-04 (WF3). It is the 2026-11-04 persist version (groups-of-8 plan, forced MoE tactics, touch
counting, embedding on host, 1M-token KV + 32 GiB CPU KV tier, host-aware persistent MoE kernels `MOE_PERSIST=1`) plus
**DSpark adaptive verification with a cost curve measured for the persistent kernels** (`MOE_AV=1`, default; see the
next section).
- ShareGPT: ~293 / 982 / 1862 tok/s at conc 1 / 8 / 32 (WF3 verification means; +3% (noise) / +5.6% / +13.9% over the
  persist version in the same session), c16 +9.9%, sampled (T=0.7) c32 +13.1%.
- zh: ~942 / 1821 tok/s at conc 8 / 32 (+2.9% / +11.7%).
- About 2.3x / 3.5x / 4.3x stock at ShareGPT c1 / c8 / c32 (stock 124 / 282 / 432).
- `MOE_AV=0` = the previous (persist, no AV) version. Previous directories: `~/dsv41/moe.bak-wf3` (persist, no AV),
  `~/dsv41/moe.bak-wf2` (2026-11-03) and `~/dsv41/moe.bak-pre-g8tac`.

## 2026-11-04 (WF3) change: DSpark adaptive verification (`MOE_AV=1`, default)
Promoted after an independent A/B verification (WF3 verifier, 2026-11-04 10:40-13:00). Backup of the previous directory
(persist kernels, no AV): `~/dsv41/moe.bak-wf3`. Candidate, calibration and rationale: `~/dsv41/tuning/av-persist/`
(`README.md`, `serve-moe-pav2.sh`). Verifier scripts: `~/dsv41/tuning/verify3/`. Results: `~/dsv41/tuning/results/wf3-*`.

### What changed
- `serve-moe.sh`: with `MOE_AV=1` (default) the DSpark config is `[[1,32,5]]` + `"enable_adaptive_verification": true`,
  and these defaults are exported to the container (any `MOE_AV_*` / `MOE_STEP_*` in the caller's env is passed through):
  `MOE_AV_CURVE=builtin MOE_AV_DRAFT_MS=1:0.1,...,32:3.2 (0.1 ms x requests)`
  `MOE_AV_VERIFY=1:7.3,2:7.6,4:8.8,6:9.2,8:11.2,12:12.7,16:14.0,18:15.2,24:17.0,32:19.1,40:21.5,48:23.2,64:26.4,96:32.9,128:39.0,192:150,256:160,1024:170,8192:400`
  `MOE_AV_MIN_REQS=2 MOE_AV_UNIFORM_CG=1`. `SPEC='<json>'` overrides the speculative config entirely.
- `plugin/moe_placement/__init__.py`: + the AV section (`_patch_av`, `_patch_av_uniform_cg`, `_prewarm_budget`, ...),
  pure additions, byte-identical to `~/dsv41/tuning/av-persist/plugin`. `persist_*.py` unchanged. The section is not
  even imported unless an `MOE_AV_*` / `MOE_STEP_TRACE` variable is set.
- What AV does: in every step with >= 2 running requests, vLLM's adaptive verification ranks the 5 drafts per request
  by DSpark's confidence head and verifies only the budget that maximizes expected accepted tokens / (draft(r) +
  verify(n)) ms. vLLM's own startup profile of that cost omits the routed-expert MoE (padding tokens route nowhere), so
  the plugin replaces it with the curve measured on this box with the persist kernels (`av-persist/calib.sh`). With
  1 request every draft is verified on the uniform 1x6 graph, i.e. the previous c1 path. Mixed prefill+decode steps
  (> 128 tokens) are priced flat, so AV keeps every draft there.

### How to turn it off
- `docker rm -f dsv41-moe; MOE_AV=0 ~/dsv41/moe/serve-moe.sh` = exactly the previous production: identical `docker run`
  arguments (checked with a stub docker), schedule `[[1,4,5],[5,8,3],[9,16,2],[17,32,1]]`, no `MOE_AV_*` env, AV code
  not loaded. Started and load-tested that way in this session.
- Full rollback: `docker rm -f dsv41-moe; rm -rf ~/dsv41/moe; cp -a ~/dsv41/moe.bak-wf3 ~/dsv41/moe; ~/dsv41/moe/serve-moe.sh`.
- Knobs (see the plugin header and `av-persist/README.md`): `MOE_AV_MIN_REQS`, `MOE_AV_DRAFT_MS`, `MOE_AV_VERIFY`,
  `MOE_AV_MAX_K`, `MOE_AV_CAP_TOKENS`, `MOE_AV_UNIFORM_CG`, `MOE_AV_LOG=1` (per-20-s `moe_placement[avdec]` lines:
  generated tok/s, kept vs scheduled tokens), `MOE_STEP_TRACE=1` (decode step ms by token count).

### Verification results (WF3)
Harness `~/dsv41/tuning/verify3/run-v3.sh` = `av-persist/run-ab3.sh` (fresh server; ShareGPT warm-up 96 prompts conc
16; ShareGPT bench c1/8/32; zh warm-up; zh bench c8/32; vllm-bench, 1024 output tokens, temperature 0, ignore-eos)
plus two extra benches after the standard ones: ShareGPT c2/4/16, and ShareGPT sampled (temperature 0.7, top-p 0.95)
c8/32. Runs interleaved ABBA: a1 (AV), p1 (previous), p2 (previous), a2 (AV). Output tok/s:

| point | AV a1 | AV a2 | previous p1 | previous p2 | AV vs previous (means) | worst AV vs best previous |
|---|---:|---:|---:|---:|---:|---:|
| ShareGPT c1 | 300.3 | 285.9 | 277.4 | 291.9 | +3.0% (noise; same path) | -2.0% |
| ShareGPT c8 | 982.8 | 980.9 | 930.5 | 929.5 | **+5.6%** | +5.4% |
| ShareGPT c32 | 1882.3 | 1840.7 | 1636.7 | 1630.8 | **+13.9%** | +12.5% |
| zh c8 | 936.4 | 948.5 | 911.1 | 920.1 | +2.9% | +1.8% |
| zh c32 | 1818.4 | 1823.7 | 1619.4 | 1641.9 | **+11.7%** | +10.8% |
| extra: ShareGPT c2 | 430.5 | 435.5 | 424.1 | 426.9 | +1.8% | +0.8% |
| extra: ShareGPT c4 | 666.4 | 640.3 | 624.5 | 623.8 | +4.7% | +2.5% |
| extra: ShareGPT c16 | 1326.4 | 1341.4 | 1214.3 | 1213.7 | **+9.9%** | +9.2% |
| extra: sampled c8 | 997.6 | 986.0 | 945.0 | 948.7 | +4.7% | +3.9% |
| extra: sampled c32 | 1822.0 | 1838.0 | 1621.7 | 1614.8 | **+13.1%** | +12.4% |

- No point regresses: the only point where the worst AV run is below the best previous run is c1 (identical path, 4-prompt
  bench noise).
- Median TPOT at c32: 15.4-15.9 ms vs 18.7-19.1 ms (sampled 15.6 vs 19.3). Acceptance length at c32 2.48-2.50 vs
  1.76-1.77 (zh 2.24-2.27 vs 1.73). At c2-c8 the acceptance length is lower (2.35-2.63 vs 2.55-2.95) because AV drops
  low-confidence drafts; the steps are faster. Expect a changed acceptance-rate metric on dashboards.
- Median TTFT is unchanged (within +-5% at every point). p99 TTFT shows the usual first-wave outliers in both configs
  (ShareGPT c8 1.2 s AV vs 1.8 s previous; c32 0.4 / 1.8 s AV vs 0.9 s previous).
- 0 failed requests and 0 errors in all four docker logs. HBM after startup 248,706 / 248,957 MiB vs 248,839 MiB.

### Correctness (a1 vs p1, same session)
- Prefill logprobs (`tf_logprobs.py`): bit-identical, 352/352 tokens, both.
- Single-request greedy decode (`correctness.py`): diverges from ref-identity at tokens 19/25/38/5/46/34 in both (c1
  path is unchanged).
- c1 argmax re-score (`single_check.py`, 10 x 512): AV 3.03% not argmax (gap > 1: 8, max 5.0) vs 2.82% (gap > 1: 7, max
  2.25). The single 5.0 gap (prompt 6, token 95) also occurs in previous-version runs (avp-p2; "one run 5.0" above):
  the c1 path is the same, the generated text differs with the adaptive-cache state.
- 30 concurrent greedy requests with AV pruning active, teacher-forced re-score (`conc_check.py`): AV 2.77% (gap > 1: 4,
  max 1.125) vs previous 3.57% (gap > 1: 5, max 2.75). Lossless within batch noise.
- 221k-token prompt: 4.1 s in both.

### Stability
- 10-minute mixed load on the AV server (`verify2/mixload2.sh`, `results/wf3-mix-*`: ShareGPT conc 16 natural lengths +
  zh conc 16 1024 out + 210k / 528k / 210k-repeat prompts): 2528 requests completed (2080 + 448; previous version 2144
  in v2-mix), 0 failed, health 200. Long prompts 8.0 / 12.7 / 0.7 s (v2-mix: 7.7 / 10.2 / 1.8 s). Min GPU free 612 MiB
  (v2-mix 952 MiB). Node 0: MemFree >= 26.3 GiB, MemFree + file LRU >= 124.9 GiB; node-1 page cache max 470 MiB.
  Docker log: only the known warnings (FlashInfer mxfp8_gemm tuning range, 3 SimpleCPUOffloadScheduler stale-pin).
- Long-prompt TTFT under a steady zh c32 decode load (`verify3/longload.sh`, unique prompts, same session):

  | | 558k-token prompts | 221k-token prompts | server decode tok/s during the prefill |
  |---|---|---|---|
  | AV (candidate server after mixload) | 19.3 / 18.5 / 19.4 s | 7.6 / 7.8 s | 405-527 |
  | AV (promoted serve-moe.sh, fresh) | 20.8 / 20.7 / 19.0 s | 7.4 / 7.4 s | 450-501 |
  | previous (`MOE_AV=0`), run 1 | 18.9 / 27.8 / 28.8 s | 13.8 / 16.2 s | 313-1069 |
  | previous (`MOE_AV=0`), run 2 | 27.8 / 28.3 / 18.0 s | 14.4 / 16.7 s | 342-1125 |

  With AV the long prefill finishes sooner and more predictably, while the concurrent streams get fewer tokens during it.
- A 1,036,682-token prompt under zh c32 load on the promoted server: TTFT 47 s, min GPU free 1,296 MiB, node-0
  MemAvailable >= 92 GiB, health 200, no errors.

### Risks / notes
- **Less HBM headroom while long prompts mix with heavy decode.** After the same long-prompt-under-load test the AV
  server had 1,300 MiB free vs 2,594 MiB for the previous version (allocator high-water ~1.3 GiB higher); minima seen:
  612 MiB (mixed load), 1,148 MiB (longload), 1,296 MiB (1.04M prompt). No OOM anywhere, but if one ever appears,
  start with `MOE_AV=0`.
- **Calibration dependency:** the verify curve was measured for plan-e207-g8 + the persist kernels + this image. After
  a plan, kernel, `MOE_PERSIST*` or image change, re-run `~/dsv41/tuning/av-persist/calib.sh` and refit, or set `MOE_AV=0`.
  A wrong curve over- or under-prunes (the earlier TRT-calibrated curve lost 7.5% at zh c8).
- **vLLM internals:** the plugin wraps `AdaptiveVerificationManager.set_initial_cost_curves / get_num_tokens` (reads
  `_batch_budget`) and `CudaGraphManager._init_candidates / dispatch` of vLLM v0.33.0. Re-validate after any image change.
- **Decode during a huge prefill:** concurrent streams decode at ~450-550 tok/s aggregate while a 200k-500k prompt
  prefills (previous version ~800-1100 tok/s, but the prefill takes 1.4-2x longer).
- Not separately tested: structured output / guided decoding and tool-call streaming with AV (spec decode is lossless
  and the parsers are unchanged; the benches used plain chat). Sampled traffic was tested only at temperature 0.7 / top-p 0.95.

## 2026-11-04 change: host-aware persistent MoE kernels (`MOE_PERSIST=1`, default)
Promoted after an independent A/B verification. Backup of the previous directory: `~/dsv41/moe.bak-wf2`.

### What changed
- `plugin/moe_placement/persist_moe.py`, `persist_moe_cute.py`, `persist_front.py` (new), plus two hook blocks in
  `plugin/moe_placement/__init__.py`:
  - at plugin registration, `TrtLlmMxfp4ExpertsModular.apply` is wrapped (before the adaptive-cache counting wrapper, so
    counting still runs);
  - after placement, a persistent MoE is bound to all 40 placed layers, and every decode size is run once (Triton
    compiles before graph capture).
  - The plugin is byte-identical to the measured candidate `~/dsv41/tuning/persist/cand/plugin`. Source, design notes
    and standalone benches are in `~/dsv41/tuning/persist/` (`README.md`, `DESIGN.md`). `persist_moe_cute.py` is
    generated by `src/make_persist_cute.py`: edit that file, not the generated one.
- `serve-moe.sh` adds `-e MOE_PERSIST=${MOE_PERSIST:-1}` and passes every `MOE_PERSIST_*` variable through. Nothing else
  changed: same plan, adaptive cache, `MOE_TACTICS`, embedding on host, KV (1M tokens on HBM + 32 GiB CPU tier),
  DSpark schedule and capture sizes.
- Every routed-expert MoE call of <= 128 tokens (decode, spec-verify and short prefill chunks) runs four launches,
  chained with PDL and captured in the CUDA graphs:
  1. a Triton front builds the routing record (cold tiles first) and gathers the MXFP8 rows;
  2. a persistent GEMM1 (CuTe DSL, one CTA per SM, 6-stage TMA pipeline);
  3. a persistent GEMM2 (10 stages). In both GEMMs, CTAs `[0, kc)` stream the DRAM-resident (cold) experts over
     NVLink-C2C while the other CTAs stream HBM, inside the same launch;
  4. a Triton finalize, only when vLLM does not defer the top-k sum (more than 16 tokens).
- Larger calls and the DSpark draft model's MoE stay on TRT-LLM, so `MOE_TACTICS` still applies to them.
- Standalone ncu at 64 tokens with 12 cold experts: host reads 378 GB/s with HBM at 5.97 TB/s, against 265 GB/s and
  4.1 TB/s for TRT-LLM. One MoE layer with 4-18 cold experts takes 24-30% less time.

### How to turn it off
- `docker rm -f dsv41-moe; MOE_PERSIST=0 ~/dsv41/moe/serve-moe.sh` gives exactly the previous TRT-LLM path: the persist
  modules are not imported.
- Partial: `MOE_PERSIST_MAX_TOKENS=<n>` sends calls above n tokens to TRT-LLM. `MOE_PERSIST_LAYERS=0-9,20` restricts
  the kernels to some layers (for bisecting).
- Other knobs and their defaults: `MOE_PERSIST_STAGES=6,10`, `MOE_PERSIST_CTAK=256,128`, `MOE_PERSIST_KC=4,16`,
  `MOE_PERSIST_BW=385,6000,46`, `MOE_PERSIST_FRONT=2`, `MOE_PERSIST_PDL=1`, `MOE_PERSIST_WARM=1`. Their meaning is in
  `~/dsv41/tuning/persist/README.md`.
- Full rollback: `docker rm -f dsv41-moe; rm -rf ~/dsv41/moe; cp -a ~/dsv41/moe.bak-wf2 ~/dsv41/moe; ~/dsv41/moe/serve-moe.sh`.

### Verification results (2026-11-04)
Harness: `~/dsv41/tuning/verify2/run-ab2.sh`, results in `~/dsv41/tuning/results/v2-*`. Each run is a fresh server
start, then:
- ShareGPT warm-up (96 prompts, conc 16) and a ShareGPT bench at conc 1 / 8 / 32;
- zh warm-up and a zh bench at conc 8 / 32.

All benches use vllm-bench with 1024 output tokens and temperature 0. Runs were interleaved A1 B1 C1 A2 B2 C2:
- A = previous `serve-moe.sh`;
- B = this version;
- C = this version plus DSpark adaptive verification (optional, not promoted; see below).

Output tok/s:

| config | run | ShareGPT c1 | c8 | c32 | zh c8 | zh c32 |
|---|---|---:|---:|---:|---:|---:|
| A previous | 1 | 256.5 | 806.9 | 1329.8 | 754.5 | 1304.7 |
| A previous | 2 | 263.5 | 791.1 | 1309.6 | 765.3 | 1302.7 |
| **B this version** | 1 | 310.5 | 942.5 | 1577.7 | 930.4 | 1617.3 |
| **B this version** | 2 | 294.3 | 927.3 | 1620.9 | 909.5 | 1616.9 |
| C B + AV (optional) | 1 | 296.4 | 970.9 | 1820.1 | 852.9 | 1792.1 |
| C B + AV (optional) | 2 | 287.7 | 974.1 | 1783.6 | 848.7 | 1832.0 |
| **B vs A (means)** | | **+16.3%** | **+17.0%** | **+21.2%** | **+21.1%** | **+24.0%** |
| C vs A (means) | | +12.3% | +21.7% | +36.5% | +12.0% | +39.0% |
| C vs B (means) | | -3.4% | +4.0% | +12.7% | -7.5% | +12.1% |

- **Every B run beats every A run at every point.** The worst B run is still 11.7-23.9% above the best A run.
- Median TPOT at conc 32: 18.7-18.8 ms against 23.0-23.1 ms. At conc 1: 3.2-3.3 ms against 3.7-3.8 ms.
- Median TTFT is unchanged. As before, the first wave of 32 simultaneous requests sometimes waits 1-3.5 s (p99 TTFT
  at conc 32 is noisy in both configs).
- HBM after startup: 248,838 MiB against 249,443 MiB (0.6 GiB less, because the TRT-LLM decode workspaces are gone).
- 0 errors in every docker log.

### Correctness
- **Prefill logprobs:** bit-identical to the reference (352/352 tokens, `tf_logprobs.py`) in every run. Texts 0 and 1
  (112 and 99 tokens) run through the new kernels.
- **Decode is not bit-identical to the previous version.**
  - Greedy decode (`correctness.py`) diverges from the identity reference on 6/6 prompts at tokens 5-46, at near-ties
    (for example p = 0.48 vs 0.60). The previous version diverges on 2/6 prompts.
  - A static-placement control (`MOE_ADAPT=0` on both sides, `verify2/ctl.sh`) compared this version with TRT-LLM forced
    to its non-split-K tactic 16:388. It is not bit-identical either: 0/36 prompts match, the first divergence is at
    tokens 0-76, and the common-prefix |dlogprob| is up to 0.6. So "bit-identical to TRT-LLM's non-split-K tactics"
    holds in the standalone row checks and for the 99/112-token prefills, but not for the whole served decode path.
  - The new path is deterministic: a re-run on the same server gave 36/36 bit-identical outputs.
- **Quality is the same as TRT-LLM's.** Greedy outputs were re-scored teacher-forced through the TRT-LLM prefill path:

  | check | previous / TRT-LLM | this version |
  |---|---|---|
  | 36 prompts x 384 tokens, static placement (13.3k tokens): not argmax | 3.01% | 2.93% (2.91% with AV, adaptive placement) |
  | same: mean \|decode - teacher-forced logprob\| | 0.0478 | 0.0479 |
  | same: gap > 1 / gap > 2 / max gap | 35 / 5 / 2.6 | 53 / 8 / 4.25 (AV run: 48 / 5 / 2.5) |
  | 10 prompts x 512 tokens, c1 (`single_check.py`): not argmax | 2.19% (gap > 1: 8, max 2.25) | 3.00 / 3.14 / 3.02 / 2.91% (gap > 1: 7-8; max 2.25, one run 5.0) |
  | 30 concurrent greedy requests (`conc_check.py`): not argmax | 3.02 / 2.90% (max gap 3.4 / 2.75) | 2.67 / 2.50% (max gap 1.1 / 1.5) |

  - The previous version's split-K tactic 8:437 gives fewer near-tie flips at c1 (2.2%). TRT-LLM's own non-split-K
    tactic gives 3.0%, the same as this version.
  - Isolated large gaps (one token where the decode logprob is about -0.1 to -1.6 and the teacher-forced one about
    -2.3 to -4.4, while the neighbouring tokens agree) occur with TRT-LLM too. This version had slightly more of them
    in the 13.3k-token test.
- **Long greedy generations** (6 prompts up to 2048 tokens: English, Traditional Chinese, code, math, a story, FFT;
  `verify2/longgen.py`) are coherent in every run. The math answer (400) is correct, and the repetition metric is in
  the same range as before.
  - They diverge from the previous version after 3-74 tokens. Two restarts of the previous version also diverge from
    each other (after 3-543 tokens).
- **Long prompt:** the 221k-token prompt prefills in 4.4 s (previous version: 4.4-4.9 s).

### Stability on the promoted production (`verify2/mixload2.sh`)
The test ran 10 minutes of mixed load at conc 32 against the promoted `dsv41-moe`:
- ShareGPT at conc 16 with natural output lengths;
- zh at conc 16 with 1024 output tokens and ignore-eos;
- long prompts of 210k, 528k and 210k tokens (the last one repeats the first, so it hits the prefix cache / CPU tier).

Results (`results/v2-mix-*`):
- **Requests:** 2144 completed (1760 ShareGPT + 384 zh), 0 failed, health 200.
- **Long prompts:** prefilled in 7.7 s, 10.2 s and 1.8 s while under load.
- **Docker log:** no errors, out-of-memory or tracebacks. Only warnings that the previous version also logs: FlashInfer
  "No tuned config covers mxfp8_gemm" for prefill shapes, the first-request JIT-compile notices, and one
  SimpleCPUOffloadScheduler warning.
- **GPU:** at least 952 MiB free throughout. The previous version's minimum in the same test was 0.78 GiB.
- **Host node 0:**
  - MemFree stayed at 29.8 GiB or more.
  - MemFree plus reclaimable file pages (an estimate of node-0 MemAvailable) stayed at 128 GiB or more.
  - Node-1 (HBM) page cache peaked at 218 MiB, so nothing spilled onto the HBM node.
- **Bench right after the mixed load:** ShareGPT 303.2 / 952.3 / 1633.1 tok/s at conc 1 / 8 / 32, the same as runs B1 and B2. Health 200, 0 errors.

### Risks / notes
- **Decode numerics changed** (see Correctness). Outputs differ from the previous version at near-ties. The aggregate
  error against teacher-forced prefill equals TRT-LLM's, but the tail of isolated large gaps was somewhat heavier in one
  13.3k-token test (gap > 1: 53 vs 35). If exact reproduction of earlier outputs matters, run with `MOE_PERSIST=0`.
- **Image dependency:** `persist_moe_cute.py` is generated from the vLLM v0.33.0 image's TP4 routed-expert kernels and
  needs its CuTe DSL / FlashInfer 0.7.2.post1. Re-validate after any image change (`~/dsv41/tuning/persist`: `pkb.py`
  CHECK, then `tf_logprobs.py`), or start with `MOE_PERSIST=0`.
- **Startup:** about 6.5 min, unchanged. The CuTe kernels compile at startup (no cache needed).
- **JIT compiles:** the jit monitor logs one-off JIT compiles on the first requests (CuTe `wrapper`, `_host_function`),
  as the previous version does. With `MOE_PERSIST_MAX_TOKENS=192` it also logs one `_front_kernel2` compile.
- **c32 is now mostly host-bound:** about 15-18 cold experts per layer per step, at about 375-379 GB/s against a C2C
  ceiling of about 385-390 GB/s. Further gains need fewer host bytes (more HBM for hot experts, or cross-layer
  prefetch), not kernel work.

### Optional: + DSpark adaptive verification (config C above, not promoted; superseded by the WF3 section: AV with a recalibrated curve is now the default)
- Script: `~/dsv41/tuning/verify2/serve-moe-pav.sh`, with a frozen plugin in `~/dsv41/tuning/verify2/plugin-pav`. The
  plugin is this version plus the opt-in `MOE_AV_*` knobs from `~/dsv41/tuning/av-cand`.
- It adds:
  - the schedule `[17,32,5]` with `enable_adaptive_verification`;
  - `MOE_AV_CURVE=moe` with the TRT-LLM-calibrated `MOE_AV_MOE_MS`;
  - `MOE_AV_MIN_REQS=17`;
  - `MOE_PERSIST_MAX_TOKENS=192`.
- It replaces `dsv41-moe` (same name and port).
- Measured twice: +12.7% at ShareGPT c32 and +12.1% at zh c32 over this version, but **-7.5% at zh c8** (852.9 / 848.7
  vs 930.4 / 909.5, reproducible) and -3.4% at c1.
  - The likely cause: with AV on, vLLM captures variable-length decode graphs instead of uniform ones below 17
    requests.
  - Prefill logprobs are bit-identical, the concurrent argmax check is 2.54 / 2.84%, and the docker logs show 0 errors.
  - It was not stress-tested under the mixed long-prompt load.
- Use it for traffic that is mostly at 17-32 concurrent requests. The AV cost curve was calibrated for TRT-LLM, not
  for the persistent kernels.


## What it does
The model has 40 MoE layers x 384 routed experts (FP4, 16.9 MiB each, 6 per token); ~61 GiB of them do not fit in HBM.
Stock vLLM UVA offload puts whole layers (0-9) in host DRAM, so ~24% of every token's expert reads cross NVLink-C2C.
Routing is skewed: the top 20% of experts take 60-75% of decode selections, the coldest 24% only ~3.5%.
The plugin keeps the cold experts in DRAM instead, per expert:

1. After the MoE kernel's weight post-processing, each layer's experts are permuted (gate rows, routing biases,
   scales and clamp limits permuted the same way, so routing decisions are unchanged).
2. `w13_weight` / `w2_weight` are re-backed by one CUDA VMM range per tensor: HBM pages for hot experts, pinned host
   pages (NUMA 0) for cold ones. Cold slots sit in groups of 8 experts spread evenly across the layer (plan
   `plan-e207-g8.json`; the previous `plan.json` used 2 MiB-aligned blocks of 16). The TRT-LLM MoE kernel launches CTAs
   in slot order, so evenly spaced small cold groups overlap the slow host tiles with HBM tiles instead of tying up many
   SMs on one burst of host reads.
3. The kernel sees an ordinary CUDA tensor; nothing else in vLLM changes.
4. **Adaptive expert cache** (`MOE_ADAPT=1`, default in `serve-moe.sh`): the plan is only the starting point. Each
   placed layer's MoE call with <= 128 tokens adds +1 per distinct expert it touches into a GPU counter (one fused
   Triton kernel; `MOE_ADAPT_COUNT=touch`, `MOE_ADAPT_FUSED=1`; large prefill chunks are not counted); every 3 s the engine thread reads it, keeps an
   EMA of counts (half-life 10 epochs) and swaps the hottest DRAM experts with the coldest HBM experts when the
   difference is both > 25% and statistically significant (> 3 sigma, Poisson), at most 64 swaps per epoch, most
   beneficial first. A swap exchanges the two slots' rows in every per-expert tensor (weights, scales, gate rows,
   routing biases, clamp limits), so addresses never change (CUDA graphs stay valid) and routing is unchanged.
   Ideas from DALI (workload windows), TriMoE (EMA load prediction, benefit-ranked migration), HybriMoE (decayed
   scores) and migration-cost gating. `MOE_ADAPT=0` = static plan only.
5. **MoE tactics** (`MOE_TACTICS=8=8:437,16=16:388,32=8:71,64=32:119`, default in `serve-moe.sh`): forces the TRT-LLM
   MoE kernel config per token bucket (<= 64 tokens, i.e. decode / spec-verify steps). FlashInfer's autotuner picked
   3-stage pipelines (115-141 KB smem) on synthetic routing; these use 4-6 stages / K-tile 512 (179-224 KB), i.e. more
   bytes in flight, which helps both small-batch HBM latency and host reads. Every forced tactic is bit-identical to
   the autotuned one (checked with server-like inputs at 5-64 tokens, both finalize paths, and in the server). The
   indices belong to FlashInfer 0.7.2.post1 in image sha256:d825baf9ce4b: after any image change re-check them with
   `~/dsv41/tuning/kernel/exsweep.py` / `exact2.py`, or start with `MOE_TACTICS=` (empty = autotuned).
6. **Token embedding on host** (`MOE_HOST_PARAMS='embed_tokens\.weight$'`): the 1.23 GiB bf16 embedding is only
   gathered (~64 rows per step), so it moves to pinned host pages (bit-exact copy, before graph capture); never the
   LM head. `MOE_HOST_PARAMS=` (empty) = off.
7. **KV**: `--kv-cache-memory-bytes 2254857830` = 1,085,568 tokens (1M context, 1.04x) instead of sizing from
   `--gpu-memory-utilization` (which now only gates the startup free-memory check), plus a 64 GiB CPU KV tier
   (`SimpleCPUOffloadConnector`, eager). The HBM saved went to hot experts (207 instead of 214 cold groups' worth).

## Files
- `serve-moe.sh` — start script. Env overrides: `MOE_AV` (default 1 = DSpark adaptive verification; 0 = previous static schedule), `MOE_AV_*`, `MOE_STEP_*`, `SPEC`, `MOE_PERSIST` (default 1; 0 = TRT-LLM only), `MOE_PERSIST_*`, `PLAN` (default `plan-e207-g8.json`), `MOE_TACTICS`, `MOE_HOST_PARAMS`,
  `MOE_ADAPT*`, `KV_BYTES` (default 2254857830; empty = size KV from `GPU_UTIL`, only with the larger-DRAM `plan.json`),
  `CPU_KV_BYTES` (default 32 GiB; 0 = no CPU KV tier), `CPU_KV_LAZY=true` (lazy offload; default eager), `NAME`, `MOE_DIR`.
- `plugin/` — vLLM general plugin (`moe_placement`), loaded via `PYTHONPATH`. `persist_moe.py`, `persist_moe_cute.py` and
  `persist_front.py` hold the host-aware persistent MoE kernels: `MOE_PERSIST=1`, plus the `MOE_PERSIST_*` knobs listed
  in the 2026-11-04 section. `__init__.py` also holds the AV section (`_patch_av`: AV cost curves, `MOE_AV_MIN_REQS`,
  uniform 1x6 graph, caps, `MOE_AV_CTL` runtime control; see the WF3 section). Env: `MOE_PLACEMENT=off|inspect|apply`,
  `MOE_PLACEMENT_PLAN`, `MOE_ADAPT=1` (+ `MOE_ADAPT_POLICY=z|ratio`, `_EPOCH_S`, `_SWAPS`, `_HALFLIFE`, `_MARGIN`, `_Z`,
  `_COUNT=sel|touch`, `_MAX_TOKENS`, `_FUSED`, `_SAVE` / `_RESUME` warm start, off by default), `MOE_TACTICS`,
  `MOE_HOST_PARAMS`, `MOE_BW_TRACE=1` (log expert bytes read from DRAM/HBM per epoch), `MOE_PLACEMENT_IDENTITY=1`
  (control: no permutation), `MOE_SPLIT=1` (two-stream variant; slower, off). With none of the new knobs set it
  behaves exactly like the previous version.
- `plan-e207-g8.json` — current placement: 3312 cold experts (54.18 GiB host, 198.95 GiB experts in HBM), per-layer
  DRAM budgets from a convex step-time model (`plan_convex.py`, ShareGPT 0.5 / zh 0.3 / code 0.2), cold slots in
  groups of 8 (`relayout.py`).
- `plan.json` — previous placement (3424 cold experts = 56.4 GiB, blocks of 16); = `~/dsv41/tuning/plans/plan-c-v2.json`.
- `routing-v1.npz`, `collect_routing.py`, `analyze_routing.py`, `plan_convex.py`, `relayout.py` — re-profiling tools.
- Previous version of this directory: `~/dsv41/moe.bak-pre-g8tac` (start it with `~/dsv41/moe.bak-pre-g8tac/serve-moe.sh`
  after editing its `M=` line, or copy it back).

## Re-profile (optional with the adaptive cache)
Hot sets are domain specific (zh / code / ShareGPT overlap only 0.2-0.38 Jaccard); a static plan built without a domain
gives that domain almost no benefit. The adaptive cache fixes that online within a few minutes; re-profiling only gives
a better starting point after restarts:

1. Start a profiling server: `cd ~/dsv41/tuning && RUST_FE=0 MOE_PLACEMENT=off ./serve-tune.sh --enable-return-routed-experts`
   (routed-expert output needs the Python frontend).
2. Put representative prompts into `collect_routing.py` (lists `ZH`, `CODE`, ShareGPT sample) and run
   `docker exec dsv41-tune python3 /tuning/routing/collect_routing.py /tuning/routing/sharegpt.json /tuning/routing/routing-new.npz 200 384 16`
3. `docker exec dsv41-tune python3 /tuning/routing/analyze_routing.py /tuning/routing/routing-new.npz /tuning/plans/plan-new.json 56.5`
   -> `plan-new-v2.json` (interleaved blocks of 16, older style), or for the current style:
4. In the vLLM image (`docker run --rm --entrypoint python3 -v ~/dsv41/moe:/m <image> ...`):
   `python3 /m/plan_convex.py <routing.npz> /m/plan.json 207 /m/plan-e207.json sharegpt:0.5,zh:0.3,code:0.2 mix`
   then `python3 /m/relayout.py /m/plan-e207.json /m/plan-e207-g8.json 8`; start with `PLAN=<file> ./serve-moe.sh`.
The DRAM budget must match the HBM left by the KV setting: with `--kv-cache-memory-bytes 2254857830`, the embedding on
host and groups of 8, 207 cold groups-of-16-equivalents (54.18 GiB) leave ~7 GiB free on the GPU after startup.
Without `KV_BYTES` the KV size comes from `--gpu-memory-utilization 0.98` and must stay >= 2.03 GiB (1M context).

## Results of the 2026-11-03 version (2026-11-03 verification, vllm-bench, 1024 output tokens, temperature 0, ignore-eos)
Each run = fresh server start, ~2.5 min ShareGPT warm-up (96 prompts, conc 16), ShareGPT bench, ~2.5 min zh warm-up,
zh bench (`~/dsv41/tuning/verify/run-ab.sh`); A = previous `serve-moe.sh` (plan.json, autotuned tactics, KV from util),
B = this version. Output tok/s:
| traffic | conc | A run 1 | A run 2 | B run 1 | B run 2 | mean A | mean B | B vs A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ShareGPT | 1 | 216.2 | 218.9 | 258.4 | 260.2 | 217.6 | 259.3 | +19.2% |
| ShareGPT | 8 | 744.3 | 748.7 | 763.8 | 813.2 | 746.5 | 788.5 | +5.6% |
| ShareGPT | 32 | 1159.2 | 1189.5 | 1320.9 | 1304.8 | 1174.4 | 1312.9 | +11.8% |
| ShareGPT-CN (zh) | 8 | 705.9 | 695.7 | 756.9 | 765.3 | 700.8 | 761.1 | +8.6% |
| ShareGPT-CN (zh) | 32 | 1160.9 | 1194.0 | 1321.5 | 1309.7 | 1177.5 | 1315.6 | +11.7% |
Every B run beats every A run at every point. vs stock `serve.sh` (124.0 / 282.3 / 432.3, zh 286.7 / 418.6):
ShareGPT 2.09x / 2.79x / 3.04x, zh 2.65x / 3.14x. Median TTFT unchanged (c32 ~250 ms); mean/p99 TTFT at ShareGPT c32
were noisy (B 540 / 803 ms mean vs A 562 / 393 ms).

Gains, from the tuning runs: MoE tactics most of c1/c8 (decode step 4.6 -> 3.8 ms TPOT at c1); groups of 8 +3-6% at c32;
touch counting + convex budgets + embedding/KV HBM -> hot experts ~+3-5%.

Correctness: prefill logprobs bit-identical to the unpermuted model (352/352, `tf_logprobs.py`). Decode vs the identity
reference: 1 of 6 prompts diverges (prompt 4 at token 49, a near-tie p=0.50 vs 0.56) — the previous version diverges on
the same prompt plus prompt 2, from adaptive slot order (summation order). With static placement, tactics on vs off give
bit-identical decode logprobs; standalone, all forced tactics are bit-identical to the autotuned ones at 5-64 tokens.

Memory: KV 2.1 GiB (1,085,568 tokens); GPU free (nvidia-smi memory.free) ~5.7-5.9 GiB after startup, ~1.6-1.9 GiB
after warm-up (previous version ~5.4 / 1.5 GiB). 10-minute mixed load at conc 32 (ShareGPT + zh + 210k / 528k / 210k-repeat prompts): 1984 requests,
0 failed, no errors, min 0.78 GiB GPU free. A 1,041,965-token prompt under conc-16 load: OK, min 0.39 GiB GPU free
(previous version: 0.42 GiB in the same test). Host node 0: the CPU KV tier pins 64 GB, so node-0 MemFree is ~16-20 GiB
(previous version ~50-55 GiB) — see the risk note below.

## Operational notes / risks
- **Page cache on HBM**: node 0's Normal-zone low watermark is 13.7 GiB (64 KiB pages). When node-0 free falls below it,
  host processes that are not NUMA-bound get page cache on node 1 = HBM, which the GPU then lacks (node-1 FilePages went
  200 -> 450 MiB once during these tests). With MemFree at ~16-20 GiB there is little slack: before copying or
  downloading large files on this box run `python3 ~/dsv41/evict-cache.py <dirs>` (or `numactl --membind=0 <cmd>`),
  or restart with a smaller CPU KV tier (`CPU_KV_BYTES=34359738368 ./serve-moe.sh`, 32 GiB).
- **MOE_TACTICS**: one unexplained engine crash happened during tuning with an earlier (non-exact) tactic table; not
  reproduced in any of 8 later starts with tactics or under the mixed / 1M-token loads. If the engine ever dies in the MoE kernel, restart
  with `MOE_TACTICS=` (the g8 plan alone needs no tactics).
- **Long-prompt prefill**: touch counting (default) ignores MoE calls with > 128 tokens, so the cache adapts to decode
  only. A fresh server prefills a 528k-token prompt in ~11-12 s with either version; over repeated long prompts the
  previous version (selection counting, prefill included) adapted to them (11.8 -> 9.5 s, 8.1 s after more load) while
  this one stays at ~11 s. For long-prompt-heavy traffic start with `MOE_ADAPT_COUNT=sel` (measured 11.0 -> 8.8 s over
  4 prompts; decode throughput with `sel` was not re-benchmarked). KV sizing, the CPU tier, the embedding move and the
  plan were each ruled out as causes. (The test prompt is one repeated paragraph, so this adaptation is exaggerated.)
- The CPU KV connector occasionally logs `SimpleCPUOffloadScheduler: ... releasing the stale pin` (WARNING, harmless).

## Earlier results (previous version, for reference)

### Results with the adaptive cache (vllm-bench, 1024 output tokens, temperature 0, after ~2.5 min warm-up traffic)
| traffic | conc | stock | static plan | adaptive | adaptive vs stock |
|---|---:|---:|---:|---:|---:|
| ShareGPT (en) | 1 | 124.0 | 216.2 | 219.0 | 1.77x |
| ShareGPT (en) | 8 | 282.3 | 627.2 | 735.0 | 2.60x |
| ShareGPT (en) | 32 | 432.3 | 930.8 | 1170.0 | 2.71x |
| ShareGPT-CN (zh), plan built *without* zh | 1 | 125.8 | 158.8 | 220.0 | 1.75x |
| ShareGPT-CN (zh), plan built *without* zh | 8 | 286.7 | 422.6 | 699.9 | 2.44x |
| ShareGPT-CN (zh), plan built *without* zh | 32 | 418.6 | 643.5 | 1149.5 | 2.75x |
(zh with the production plan, static: 220.2 / 595.0 / 922.2.) Note: `--ignore-eos` benchmarks generate past EOS;
the adaptive cache also learns that traffic, which real requests do not produce.

### Bandwidth (measured during inference; HBM = GPM DRAM activity counter, DRAM = plugin MOE_BW_TRACE accounting)
GPU-side C2C/PCIe/NVLink counters read 0 on this GB300 driver, so DRAM traffic is counted by the plugin
(distinct DRAM-resident experts touched per forward x 16.875 MiB); C2C link = 10 x 44.7 = 447 GB/s per direction.
| layout | conc | steps/s | DRAM experts/forward | DRAM reads | HBM activity | SM active |
|---|---:|---:|---:|---:|---:|---:|
| stock layer offload (L0-8 in DRAM) | 1 | 40.8 | 218 | 155 GB/s | 13% | 68% |
| stock layer offload | 8 / 16 / 32 | 15.2 / 11.8 / 9.0 | 925 / 1240 / 1624 | 249 / 259 / 259 GB/s | 15% | 91-93% |
| hot/cold + adaptive | 1 | 67.3 | 50 | 59 GB/s | 24% | 51% |
| hot/cold + adaptive | 8 / 16 / 32 | 34.6 / 26.8 / 20.6 | 221 / 315 / 431 | 134 / 149 / 156 GB/s | 40-41% | 75-82% |
Stock is bound by host-memory reads: DRAM traffic plateaus at ~259 GB/s (the most the MoE kernel achieves over C2C,
~58% of the link) while SMs sit at 92% busy, stalled, with HBM at 15%. After the change DRAM reads drop 3.8x per step
and neither tier is saturated (C2C ~35% of link, HBM ~40% of ~8 TB/s); the limit is MoE kernel efficiency at
2-3 tokens per expert per step plus host-read latency of the remaining cold tiles.

### Results, static plan (vllm-bench, ShareGPT prompts, 1024 output tokens, temperature 0)
| conc | stock serve.sh | serve-moe.sh | speedup |
|---:|---:|---:|---:|
| 1 | 124.0 tok/s | 216.2 | 1.74x |
| 2 | 151.4 | 330.5 | 2.18x |
| 4 | 203.4 | 425.7 | 2.09x |
| 8 | 282.3 | 627.2 | 2.22x |
| 16 | 363.5 | 759.3 | 2.09x |
| 32 | 432.3 | 930.8 | 2.15x |

Correctness: prefill logprobs are bit-identical to the unpermuted model, also after >1,100 live adaptive swaps (352/352
tokens); decode differs only by top-k summation order (max |dlogprob| 0.15 on 6 test prompts), less than the run-to-run
variation from request batching (0.56).

## 2026-11-03 21:13 change: CPU KV tier default 64 GiB -> 32 GiB
`serve-moe.sh` now defaults to `CPU_KV_BYTES=34359738368` (32 GiB). The verifier measured only 16-20 GiB free on NUMA node 0 with the
64 GiB pinned tier (node-0 page cache had started spilling onto the HBM NUMA node). Throughput was unaffected by the tier size in all
benches. `CPU_KV_BYTES=68719476736 ./serve-moe.sh` restores the old size; backup of the previous script: `serve-moe.sh.bak-cpukv64`.
