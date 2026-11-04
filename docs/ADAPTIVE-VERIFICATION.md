> **Provenance:** engineering notes written during the TidalMoE tuning work (Nov 2026) on the GB300 test machine.
> Paths such as `~/dsv41/...`, `~/dsv41/tuning/...` and `~/dsv41/moe/...` refer to that original workspace;
> in this repository the production pieces live in `serve.sh`, `plugin/`, `plans/`, `bench/`, `correctness/` and `tools/`.

# DSpark adaptive verification (AV) on top of MOE_PERSIST=1 — candidate `serve-moe-pav2.sh` (2026-11-04)

**Update 2026-11-04 (WF3 verifier): promoted.** ~/dsv41/moe/serve-moe.sh now runs this config by default (MOE_AV=1; MOE_AV=0 = previous); plugin copied to ~/dsv41/moe/plugin; backup ~/dsv41/moe.bak-wf3; verification in ~/dsv41/moe/README.md (WF3 section).

Not promoted: `~/dsv41/moe` is unchanged and production (`dsv41-moe`, `~/dsv41/moe/serve-moe.sh`) was restored at the end.
`serve-moe-pav2.sh` = production `serve-moe.sh` + AV. It uses the same container name and port, so it replaces dsv41-moe.

## Result (same session, interleaved, standard harness `run-ab3.sh` = `verify2/run-ab2.sh`)
Output tok/s, vllm-bench, 1024 output tokens, after warm-up; ShareGPT c1/c8/c32, ShareGPT-zh c8/c32.

| run | c1 | c8 | c32 | zh c8 | zh c32 |
|---|---|---|---|---|---|
| production avp-p1 / p2 / p3 | 287.6 / 300.7 / 304.1 | 968.3 / 926.4 / 972.6 | 1611.6 / 1582.1 / 1644.6 | 915.7 / 892.2 / 898.7 | 1625.1 / 1641.5 / 1626.6 |
| **pav2 (final) avp-f1 / f2** | 295.8 / 301.9 | 1003.2 / 984.3 | 1866.1 / 1859.0 | 927.7 / 938.2 | 1802.6 / 1822.0 |
| **mean vs production** | **+0.5 %** | **+4.0 %** | **+15.5 %** | **+3.4 %** | **+11.1 %** |
| pav2 with AV also at 1 request (MIN_REQS=0) avp-b1 / b2 | 285.0 / 284.3 | 999.5 / 950.0 | 1887.8 / 1902.1 | 940.9 / 955.5 | 1835.1 / 1804.5 |

Run order: p1, b1, p2, b2, (low-concurrency A/B), f1, p3, f2. b1/b2 differ from the final config only for batches of one
request, so for c8 and above they are two more samples of it: over 4 runs c8 +3.0 %, c32 +16.5 %, zh c8 +4.3 %, zh c32 +11.3 %.
The verifier's config (`verify2/serve-moe-pav.sh`) measured c1 −3.4 %, c8 +4.0 %, c32 +12.7 %, zh c8 −7.5 %, zh c32 +12.1 %.

* c32: TPOT 15.4–15.7 ms vs 18.8–19.0 ms; acceptance length 2.47–2.49 vs 1.77 (zh 2.26–2.29 vs 1.73).
* Low concurrency (`lowc.sh`, the same 16–24 prompts, AV pruning switched on/off inside one pav2 server via MOE_AV_CTL,
  then production): c1 AV 300.1/303.5 vs no pruning 308.4/313.3 (−3.0 %), production 307.7;
  c2 AV 479.2/459.3 vs no pruning 451.1/451.0 (+4.1 %), production 475.5; c4 AV 698.5/717.1 vs 662.8/654.6 (+7.6 %),
  production 645.4 (+9.7 %). Hence AV prunes only from 2 running requests on (MOE_AV_MIN_REQS=2).

## Config (differences from production)
| | production | verifier pav | **pav2** |
|---|---|---|---|
| dspark schedule | [[1,4,5],[5,8,3],[9,16,2],[17,32,1]] | ...,[17,32,5]] | **[[1,32,5]]** (AV may keep up to 5 drafts at any size) |
| AV prunes at | off | >= 17 requests | **>= 2 requests** |
| cost curve | — | profiled (padding tokens, no MoE) + 40 x TRT-LLM kbench MoE ms; profiled drafter 7.5 ms at >= 24 requests | **measured with persist: draft = 0.1 ms x requests, verify = pure-decode step ms by target tokens** |
| decode graphs | per-tier uniform (44) | 15 varlen | 15 varlen + uniform 1x6 (production's c1 graph) |
| MOE_PERSIST_MAX_TOKENS | 128 | 192 | 128 |

Env (all defaults in the script): `MOE_AV_CURVE=builtin MOE_AV_DRAFT_MS=1:0.1,...,32:3.2`
`MOE_AV_VERIFY=1:7.3,2:7.6,4:8.8,6:9.2,8:11.2,12:12.7,16:14.0,18:15.2,24:17.0,32:19.1,40:21.5,48:23.2,64:26.4,96:32.9,128:39.0,192:150,256:160,1024:170,8192:400`
`MOE_AV_MIN_REQS=2 MOE_AV_UNIFORM_CG=1`. AV maximizes (expected accepted tokens) / (draft(r) + verify(n)) per step, n
padded to the CUDA-graph sizes; > 128 tokens has no graph (priced high, never chosen for pure decode).

## Findings
1. **Cost curve** (`calib.sh`, run avp-cal1; MOE_AV_CTL `force_tokens` holds the kept budget fixed; pure-decode GPU ms per step,
   `moe_placement[step]` trace). The step cost depends almost only on the target token count n; per request +0.1 ms.

   | n | 8 | 16 | 24 | 32 | 40 | 48 | 64 | 96 | 128 |
   |---|---|---|---|---|---|---|---|---|---|
   | r=8 SG / zh | 11.8 / 12.3 | 14.8 / 14.9 | 17.9 / 17.5 | 19.5 / 19.6 | | | | | |
   | r=17 SG | | | 18.6 | 20.7 | | 24.7 | 28.0 | 33.7 | |
   | r=24 SG | | | 19.8 | 21.6 | | 25.3 | 28.7 | 35.2 | 39.0 |
   | r=32 SG / zh | | | | 22.5 / 22.7 | 24.5 / 25.3 | 25.8 / 27.9 | 29.4 / 30.3 | 35.3 / 38.4 | 42.2 / 44.3 |
   | old (TRT) model r=32 | | | | 29.1 | 33.2 | 37.8 | 41.9 | 49.6 | 61.8 |
   | old model r=8 | 10.8 | 15.5 | 19.6 | 22.8 | | | | | |
   c1: 1x2 7.9, 1x4 8.4, 1x6 9.3 ms. The old curve is 30–45 % too high at r=32 but roughly proportional there (same choices).
   At r=8 it makes 12 tokens look cheap (12.0 vs 13.6 ms real) and 32 expensive (22.8 vs 19.4) → it over-prunes small
   batches (the earlier c8 losses). Its drafter cost of 7.5 ms at >= 24 requests is an artifact: dspark captures only
   10 drafter graphs (<= 20 requests x 5 query tokens <= 128), 21–32 requests run the drafter eagerly, but at runtime that
   is hidden (r=17/24/32 at equal n differ by ~0.1 ms per request).
2. **Why the verifier's zh c8 lost**: AV was not active there (8 < 17 requests; kept = scheduled tokens), acceptance
   length was the same (2.56 vs 2.57), and the decode-graph kind does not matter: uniform vs varlen graphs at 8 x 4 tokens
   19.6 vs 19.2 ms (zh), 19.3 vs 19.2 (SG); at 1 x 6 9.3 vs 9.3 (ctl `varlen_only` A/B in one server). Both pav runs instead
   had a ~2 s engine stall exactly when zh c8 started (TTFT p99 2174 / 2140 ms vs 196 / 199; the adaptive cache's 3-s epoch
   log shows a 5-s gap at 21:01:34→39 and 21:48:57→49:02 UTC) plus +6 % median TPOT. pav2 does not reproduce it (zh c8 TTFT
   p99 199–210 ms, +3.4 %); it differs from pav in MOE_PERSIST_MAX_TOKENS (128 vs 192), curve and schedule. The stall's
   cause was not pinned down (no log line; production has similar 1–3 s first-shape stalls at other points, e.g. SG c8/c32
   TTFT p99 1.2–3.0 s in p1/p2).
3. **Policy A/B inside one AV server** (`calib3.sh`, avp-cal3; schedule [[1,32,5]], policy switched every 20 s, 3 cycles;
   metric = pure-decode tokens per step / GPU ms per step, insensitive to prefill waves). PROD = production schedule
   emulated per request (`static_sched`), PAV = verifier policy, AVN = measured curve (= pav2), AVNP/AVNM = draft cost +4 / −3 ms.

   | decode tok/s | PROD | PAV | AVN | AVNP | AVNM | static 2 drafts |
   |---|---|---|---|---|---|---|
   | SG c32 | 1801 | 2168 | 2187 | 2206 | 2175 | 2006 |
   | SG c16 | 1460 | – | 1601 | 1596 | 1633 | |
   | SG c8 | 1087 | – | 1157 | 1132 | 1121 | |
   | SG c1 | 302 | – | 318 | 309 | 324 | (window noise +-12 %) |
   | zh c32 | 1671 | 1926 | 2026 | 1967 | 1965 | |
   | zh c8 | 1018 | – | 1024 | 1033 | 1039 | |
   At c32 every curve keeps 96 tokens (3 per request) in > 90 % of steps, so the curve barely matters there; the gain is
   AV choosing which drafts to verify (+9 % over a static 2 drafts). Curve offsets (AVNP/AVNM) and a 96-token cap were not
   better overall. Side finding: if AV is not adopted, production's [17,32,1] is not optimal for the persist kernels —
   a static 2 drafts per request at 32 requests decoded +11 % faster (not benchmarked with the full harness).
4. AV at zh c8 with pav2: active; it keeps 16 / 24 / 32 of the 48 scheduled tokens in 31 / 55 / 14 % of steps
   (acceptance 2.42–2.47 vs 2.55, steps 17.4 vs 19.3 ms).

## Correctness (avp-f1 vs production avp-p2 / p3, same session)
* Teacher-forced prefill logprobs (`tf_logprobs.py`, 352 tokens): bit-identical (mean/max |d| 0).
* Single-request greedy decode (`correctness.py` vs ref-identity): diverges at tokens 19/25/38/5/46/34, exactly where
  production persist (avp-p2) does: c1 uses production's 1x6 graph and no pruning, so c1 decode matches production.
* c1 argmax re-score (`single_check.py`, 10 prompts x 512): 2.84 % not argmax, max gap 2.25 (production p2: 3.05 %, max 5.0).
* 30 concurrent greedy requests, AV pruning active, re-scored teacher-forced (`conc_check.py`): 2.46 % not argmax,
  max gap 2.25 (production p2: 2.46 %, max 3.5; b1 with MIN_REQS=0: 2.27 %, 1.625). Lossless within batch noise.
* Long prompt (cache-exp/longprompt.py 210000 = 221,381 tokens): 4.2 s (production 4.3 s).

## Memory
HBM after startup 248,703 MiB (production 248,836–248,839); after the benches 252,239–252,413 MiB (production 252,287–252,802);
after the 210k prompt 253,867 MiB (production 253,743), i.e. 2.8 GiB free. Node 0 unchanged (no extra host memory).

## Files
* `serve-moe-pav2.sh` — the candidate start script (defaults = final config; NAME=..., MOE_AV_* overridable).
* `plugin/` — production plugin + AV section in `__init__.py` only (pure additions; persist_* byte-identical to
  `~/dsv41/moe/plugin`). New knobs: `MOE_AV_DRAFT_MS`, `MOE_AV_VERIFY` (curve overrides), `MOE_AV_MAX_K`, `MOE_AV_CAP_TOKENS`
  (per-batch caps), `MOE_AV_STATIC_SCHED` (per-request draft cap by batch size), `MOE_AV_CTL=<json>` (runtime overrides:
  min_reqs, caps, curves, force_tokens, static_sched, varlen_only; mounted at /avp from `run/`), `MOE_AV_LOG=1`
  (`moe_placement[avdec]` lines: AV decisions, generated tok/s, pure-decode tokens/steps and GPU ms).
* `calib.sh` (step-cost calibration), `calib3.sh` (policy A/B), `lowc.sh` (c1/c2/c4 paired A/B), `analyze.py <docker log>`
  (per-ctl-label aggregation), `run-ab3.sh` + `queue.sh` + `final2.sh` (A/B harness). Logs: `~/dsv41/tuning/results/avp-*`.

## Caveats
* The verify curve is a single pure-decode curve (SG/zh mixed, current HBM/host split); a different plan or kernel changes
  it — re-run `calib.sh` (force_tokens at r=8/17/24/32) and refit (0.1 ms x r + step ms by n).
* Mixed prefill+decode steps (> 128 tokens) are priced flat (150–400 ms), so AV keeps every draft there.
* AV prunes on the confidence head's survival estimates; at 1 request this lost ~3 % on the bench prompts, hence MIN_REQS=2.
  c2 is within noise of production (AV 469 vs 475.5 tok/s), c4 +9.7 %.
* HBM headroom after a 210k prompt is ~2.8 GiB, the same as production.
