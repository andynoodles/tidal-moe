> **Provenance:** engineering notes written during the TidalMoE tuning work (Nov 2026) on the GB300 test machine.
> Paths such as `~/dsv41/...`, `~/dsv41/tuning/...` and `~/dsv41/moe/...` refer to that original workspace;
> in this repository the production pieces live in `serve.sh`, `plugin/`, `plans/`, `bench/`, `correctness/` and `tools/`.

# DGX Station GB300: machine architecture for MoE inference optimization

Written 2026-11-03 without using the GPU (a bandwidth measurement run was in progress). Every claim carries a tag:

* **[D]**: NVIDIA documentation (URLs in §9)
* **[M]**: read or measured on this box (raw evidence in `results/inspect-20261103.txt`, `results/devquery.txt`, `../results/bw-sequence.log`)
* **[T]**: third-party measurement on GH200, not this machine. Treat it as a prior only.
* **[I]**: my inference from [D]/[M] numbers. Each one has a microbenchmark in `bench/` that confirms or refutes it.

Do not substitute generic PC or discrete-GPU intuition for these numbers. On this machine there is no PCIe in the CPU↔GPU data path, host memory is
coherent with the GPU, and the host DRAM (not the link) is the ceiling.

---------------------------------------------------------------------------------------------------------------------------

## 0. TL;DR

1. **The ceiling for anything the GPU reads from host memory is LPDDR5X, not NVLink-C2C.** The link is 10 × 44.712 = **447 GB/s per
   direction** [M]. The DRAM behind it is **396 GB/s** (DGX Station spec [D]), and the firmware HMAT table on this box reports **360 GB/s** read and write [M].
   Reads and writes share that DRAM budget. The CPU's own traffic does too. On GH200 (LPDDR5X 500 GB/s > C2C 450 GB/s) the link was the
   limit. On this SKU it is not, so "447 GB/s" is unreachable for DRAM-resident data.
2. **The MoE GEMM's host stream already runs close to that ceiling.** The earlier profile split per layer is: GEMM1 grows from 0.39 ms (all-HBM) to 0.95 ms with
   ~16 cold experts, and GEMM2 from 0.20 to 0.48 ms. In both cases the extra time is exactly the cold bytes at **≈337 GB/s**
   (188.7 MB / 0.56 ms; 94.4 MB / 0.28 ms) [I]. That is about 94% of the HMAT 360 and 85% of the 396 spec. The loss is not a slow host stream. **The host time is added to the
   HBM time instead of overlapping with it** (sum, not max) [I]. With perfect overlap, GEMM1 would take max(0.39, 0.56) = 0.56 ms.
3. **The average C2C rates you see (170 GB/s in production at conc 32, 258 GB/s on the stock layout) are step-averages** [M: C2C0 PMU 2026-11-03].
   In production at conc 32, a 51.5 ms step moves 8.8 GB from host. At ~337 GB/s that is about 26 ms of host streaming, so the link sits idle for
   about half of every step [I].
4. **The GPU does not cache CPU memory.** "GB300's access to CPU memory is not cached" [D DGX Station guide]. The driver also has `uvm_exp_gpu_cache_sysmem=0` [M].
   Every re-read of a cold weight tile crosses C2C again. Examples are several N-tiles per expert in prefill or a second kernel touching the same expert. L2
   promotion and persisting L2 do nothing for host data.
5. **Allocation type decides translation.** `cuMemCreate(HOST_NUMA)`, which is what moe_placement uses, is **EGM** and is mapped with **2 MB GPU pages** [D]. It is
   counted on the **C2C0** PMU together with ATS traffic [D kernel doc, M]. `malloc`/mmap memory goes through **ATS**: on a GPU TLB miss the SMMU walks the CPU page
   table with **64 KiB pages** [D/T]. VMM granularity is 2 MiB for both device and HOST_NUMA [M].
6. **TMA works on host-resident (EGM) memory** [M]. The production MoE kernels (trtllm-gen `bmm_*E2m1*`) load the weight operand with TMA
   (`cuTensorMapEncodeTiled`, L2 promotion 128B), and they read cold experts from EGM pages correctly (bit-identical prefill check). The CUDA docs place no memory-type restriction on the TMA global address [D].
7. **Levers, largest first** [I]: (a) reduce cold bytes per step through cache hit rate and HBM capacity allocation; (b) make the host stream **overlap** the HBM stream inside
   the MoE layer: a cold-only kernel on a few SMs with a deep pipeline (green-context SM partition), or copy-engine staging; (c) move cold
   transfers into the half of the step where the link is idle (predict + CE prefetch); (d) speeding up the host stream itself is worth at most ~7–15%.
8. **Computing cold experts on the Grace CPU (Fiddler-style) gains no bandwidth.** The CPU reads the same LPDDR5X; NVIDIA's STREAM reference for 72 threads on 480 GB
   Grace-Hopper/Blackwell is only ≥307 GB/s [D]. It would also break bit-identity. Its only benefit, freeing GPU SMs from host latency, is also available from (b) [I].
9. **Profiling permissions:** GPU hardware counters (ncu, nsys GPU metrics) are **admin-only on this box** (`RmProfilingAdminOnly: 1` [M]), so run nsys as root
   in a privileged container (`bench/nsys_system.sh`, `bench/serve_nsys.sh`). Grace uncore PMUs need `perf_event_paranoid ≤ 0`, which is currently -1 and temporary.
   **SCF `cmem_*` events exclude the GPU's C2C reads** [M]: GPU at 258 GB/s while `cmem_rd_data` shows 0.1 GB/s. LPDDR5X total = C2C0 + C2C1 + CMEM.

---------------------------------------------------------------------------------------------------------------------------

## 1. System identity and topology

| item | value | src |
|---|---|---|
| chassis | Supermicro "Super AI Station GB300" (a DGX Station GB300 platform), board GPU-NVGB300-WSHPM | [M] dmi |
| OS / kernel | Ubuntu 24.04, `7.0.0-1024-nvidia-64k` (64 KiB pages), cmdline `init_on_alloc=0 numa_balancing=disable` (both NVIDIA-recommended for GH/GB) | [M], [D os-settings] |
| driver | 595.99.02 open kernel module, CUDA driver API 13.2 (13020); host toolkit `/usr/local/cuda-13.2` incl. nvcc, nsys 2025.6.3, ncu | [M] |
| GPU mode | `Addressing Mode: ATS`, `GPU C2C Mode: Enabled`, capability `EGM: enabled`, MIG off, ECC on | [M] nvidia-smi -q |
| GPU memory mode | **NUMA mode** (legacy): HBM is onlined as Linux NUMA node 1 (`CoherentGPUMemoryMode: ""`). NVIDIA states CDMM is the default from R610 and "NUMA mode is not recommended" | [M], [D coherency] |
| module power | GB110 module (GPU + Grace) shares **1600 W** (module limit) and the GPU limit is 1300 W. nvidia-smi shows 39 s of cumulative SW power capping since boot | [M], [D power sloshing] |

```
             LPDDR5X 496 GB (node 0, 494 GiB visible)          HBM3e 252 GB (node 1, 249.5 GiB usable)
             396 GB/s spec | HMAT: 360 GB/s, 160 ns            7.16 TB/s (7168-bit @ 3996 MHz) | HMAT: 7.9 TB/s, 370 ns
                       |                                                   |
   +-------------------+------------------+                +---------------+--------------------------+
   | Grace: 72 x Neoverse V2 @3.37 GHz     |   NVLink-C2C   | B300 "GB110": 2 dies joined by NV-HBI    |
   | L1 64K+64K, L2 1 MiB/core             |<==============>| (10 TB/s) acting as one GPU; 152 SMs     |
   | L3 114 MiB distributed in SCF mesh    | 10 x 44.712    | @2.07 GHz, L2 115.5 MiB, 228 KiB smem/SM,|
   | SMMUv3 (GPU behind smmu3.0x16000000)  | = 447 GB/s/dir | 256 KB TMEM/SM, 3 async copy engines     |
   | uncore PMUs: SCF, C2C0, C2C1, PCIe    | coherent (CHI) | PCIe Gen6 x16 (not on the CPU data path)  |
   +---------------------------------------+                +------------------------------------------+
   CPU -> HBM (HMAT): 140 GB/s, 560 ns read        GPU -> LPDDR5X: not in HMAT (measure: bench chase/read)
```

---------------------------------------------------------------------------------------------------------------------------

## 2. Memory hierarchy with numbers

### 2.1 GPU (Blackwell Ultra B300, compute capability 10.3)

| property | value | src |
|---|---|---|
| SMs | **152** on this SKU. The full B300 has 160 ("available SM count and HBM capacity varies by SKU") | [M] devquery, [D BW-Ultra blog] |
| SM clock | 2070 MHz max = current in P0 | [M] |
| per SM | 2048 threads, 64K regs, 32 blocks, **228 KiB shared** (227 KiB opt-in/block), 4 5th-gen tensor cores, **256 KB TMEM** | [M], [D blog] |
| L2 | **121,110,528 B = 115.5 MiB** (two dies). Max persisting L2 72.2 MiB, access-policy window ≤128 MiB | [M] |
| HBM3e | bus **7168 bit** (7 × 1024-bit stacks enabled; B300 has 8), memory clock 3996 MHz → **7.16 TB/s** (spec "7.1 TB/s"), **252 GB** spec, CUDA totalMem 249.5 GiB | [M], [D DGX Station] |
| dies | "two reticle-limited dies connected by a 10 TB/s chip-to-chip interconnect in a unified single GPU" | [D Blackwell arch] |
| copy engines | `asyncEngineCount = 3` | [M] |
| misc | `TENSOR_MAP_ACCESS_SUPPORTED=1`, `CLUSTER_LAUNCH=1`, `MULTICAST_SUPPORTED=0`, HW decompression (`MEM_DECOMPRESS_ALGORITHM_MASK=7`), 64-bit stream mem-ops and `WAIT_VALUE_NOR` supported | [M] |
| coherence attrs | `PAGEABLE_MEMORY_ACCESS=1`, `..._USES_HOST_PAGE_TABLES=1` (ATS), `HOST_NATIVE_ATOMIC_SUPPORTED=1`, `CONCURRENT_MANAGED_ACCESS=1`, `DIRECT_MANAGED_MEM_ACCESS_FROM_HOST=1` | [M] |
| NUMA attrs | `NUMA_CONFIG=1` (GPU memory is a NUMA node), `NUMA_ID=1`, `HOST_NUMA_ID=0`, `HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=1` | [M] |
| VMM granularity | **2 MiB** minimum and recommended, for DEVICE and for HOST_NUMA | [M] |

The firmware HMAT table lists GPU→HBM at 7.9 TB/s (read latency 370 ns, write 200 ns). That 7.9 TB/s looks like the generic 8-stack figure; this SKU
has 7 stacks, so use 7.16 TB/s [M/I].

### 2.2 Grace CPU and LPDDR5X

| property | value | src |
|---|---|---|
| cores | 72 × Arm Neoverse V2 r0p0, 1 thread/core, max 3375 MHz (cpufreq 3.366 GHz, governor `performance`, boost disabled) | [M] lscpu |
| SIMD | "four 128-bit SVE2 units per core"; flags include sve2, bf16, i8mm, svebf16, svei8mm, asimddp | [D DGX overview], [M] |
| caches | L1d 64 KiB 4-way, L1i 64 KiB, L2 1 MiB 8-way per core, **L3 114 MiB** (12-way, 64-B lines) distributed over the SCF mesh, SCF bisection > 3.2 TB/s | [M], [D Grace tuning guide] |
| memory | **496 GB LPDDR5X** (SOCAMM, ECC). Node 0 = 494 GiB visible | [D], [M] |
| bandwidth | **396 GB/s** spec. HMAT (firmware) **360 GB/s read, 360 GB/s write; latency 160 ns read / 140 ns write** | [D DGX Station], [M] |
| STREAM reference | NVIDIA health-check minimum TRIAD for 72 threads on 480 GB Grace-Hopper/Blackwell: **307 GB/s** ("80–95% of theoretical") | [D health-checks] |
| C2C reference | NVIDIA health check, 480 GB LPDDR5X systems: host→device CE ≈ **350–360 GB/s**. GH200-120GB sample: 416 H2D / 295 D2H | [D health-checks] |

### 2.3 NUMA, pages, translation

* **NUMA** [M]: node 0 = CPUs 0-71 + LPDDR5X. Node 1 = HBM, 249.5 GiB, no CPUs, distance 80. Nodes 2-8 are empty MIG placeholders [D Grace tuning guide, GH200 example].
  The memory block size is 512 MiB and node 1 = 499 blocks starting at block 131072, so the HBM physical base is 64 TiB.
  *Because the box runs in NUMA mode, the kernel can place ordinary CPU pages, including page cache, in HBM.* About 104 MB of file pages were sitting on node 1 [M].
  Keep `numactl --membind=0` on every process, as serve scripts already do. CDMM (`NVreg_CoherentGPUMemoryMode=driver`) removes this, but it needs root and a reboot [D].
* **CPU pages** [M]: base page **64 KiB**. The THP PMD size is **512 MiB** (enabled=madvise, the 512M size `inherit`). 2 MiB mTHP exists but is `never`. hugetlb pools are 0.
  NVIDIA recommends 512M `never` and 2M `madvise` on 64K kernels [D os-settings]. Changing that needs root.
* **GPU pages** [D]: "NVIDIA GPUs support multiple physical page sizes, but prefer 2MiB physical pages or larger … TLB misses are generally significantly
  more expensive on the GPU". "EGM is mapped with 2MB pages. Therefore, users may encounter more TLB misses when accessing very large allocations."
  cudaMalloc and GPU-resident managed memory use 2 MB pages [T Schieffer]. System memory reached through ATS uses the CPU page table, which has 64 KiB pages here.
* **IOMMU** [M]: the GPU (0009:06:00.0) sits behind SMMUv3 `smmu3.0x16000000`, and its IOMMU group type is `identity` (DMA passthrough). The PMCG of that SMMU are:
  TCU `smmuv3_pmcg_16002` (`pcie_ats_trans_rq`, `tlb_miss`, `trans_table_walk_access`) and TBUs `smmuv3_pmcg_16042/16062` (`pcie_ats_trans_passed`).
  `bench/pmu.sh` counts these, which shows which allocation kinds cause ATS translations.

---------------------------------------------------------------------------------------------------------------------------

## 3. How the GPU reads Grace memory on this system

### 3.1 Three paths. All three appear on the C2C0 PMU here.

| allocation | GPU translation | counted on | used by |
|---|---|---|---|
| `cuMemCreate(CU_MEM_LOCATION_TYPE_HOST_NUMA, 0)` + `cuMemMap` | GPU page table, **2 MB pages** ("EGM": GPU addresses CPU memory as GPU memory) | C2C0 ("ATS translated **or EGM**") | moe_placement cold experts [M] |
| `cudaHostAlloc` / `cudaHostRegister` (pinned) | GPU page table entries made by the driver; page size not documented → bench `chase stride=64K/2M` | C2C0 [M calibration] | torch `pin_memory` (engram default), vLLM UVA offload |
| `malloc` / `mmap` (system-allocated, not registered) | **ATS**: on a GPU TLB miss the ATS-TBU sends a translation request over C2C to the Grace SMMU, which walks the **CPU page table (64 KiB pages)** [D, T] | C2C0 | nothing in production |
| (non-ATS "untranslated" GPU traffic) | | **C2C1** | reads ~0 in our workloads [M] |

* Sources: kernel doc "C2C0 … captures GPU ATS translated or EGM traffic"; "C2C1 … captures untranslated GPU traffic". nsys event help: C2C0 `rd_bytes_local` =
  "bytes read via ATS and Extended GPU Memory traffic by the Socket 0 GPU" [D]. That is why pinned host memory shows on C2C0 and not C2C1.
* **The SCF `cmem_*` counters only see the CPU's own DRAM traffic** [M]. Today's `bw-sequence.log` shows C2C0 at 258.1 GB/s while `cmem_rd_data`×32 = 0.1 GB/s.
  Total LPDDR5X load is therefore C2C0 + C2C1 + CMEM, as `measure3.sh` already computes.
* EGM: "Mapping system memory as EGM does not cause any performance issues" and "allocations must be aligned to the granularity" [D EGM].

### 3.2 Coherence and caching (differs from a PCIe GPU)

* NVLink-C2C is a coherent interconnect (AMBA CHI, 64-B coherence granularity [T]). The CPU **can** cache GPU memory. **The GPU's accesses to CPU memory are not cached**
  [D DGX Station guide, verbatim: "while the CPU can cache accesses to GB300's memory, GB300's access to CPU memory is not cached"]. Fusco et al.: the Hopper L2 caches HBM
  data, local and peer [T]. The driver default `uvm_exp_gpu_cache_sysmem=0` [M].
  *Consequences* [I]: every warp or TMA access to a host line crosses C2C and LPDDR5X again; L2 promotion, persisting L2 and access-policy windows do not apply to host data;
  any re-read multiplies host traffic. Bench `l2` confirms this.
* Transfer granularity: 64 B on the CPU side, 128 B on the GPU side [T Schieffer]. `bench/pmu.sh` reports the real bytes per C2C read request (`c2c0_B_per_rd_req`).
* Native atomics on host memory are supported (`HOST_NATIVE_ATOMIC_SUPPORTED=1`), and so are 64-bit stream memory operations and wait-value NOR [M]. A kernel or graph can
  therefore spin on, or `cuStreamWaitValue` on, a flag in host memory that the CPU writes, and the reverse, without page faults. This is the low-latency CPU↔GPU handshake
  primitive on this platform [D unified-memory, I].

### 3.3 Bandwidth ceilings (what can and cannot reach 447 GB/s)

* Link: 447 GB/s per direction (900 GB/s total) [M, D]. C2C client interface: at most 10 read requests per cycle (Grace guide utilization formula) [D].
* **Behind the link is LPDDR5X at 396 GB/s spec, 360 GB/s HMAT** [D, M]. Reads and writes share it. CPU traffic shares it. The SCF CMEM accepts 8 read and 8 write requests per cycle [D].
  So **GPU reads of DRAM-resident data cannot exceed ~360–396 GB/s.** NVIDIA's own expectation for CE host→device on 480 GB LPDDR5X Grace systems
  is 350–360 GB/s [D health-checks].
* Only data already in Grace's 114 MiB L3 could approach the link rate, for example data the CPU just wrote. Whether GPU reads are served from the SCF cache is undocumented. bench `slc` tests it.
* Writes (D2H): NVIDIA lists device→host lower than host→device on GH200 (295 vs 416 GB/s) [D]. The CUDA guide also notes "the device can generally write at a higher bandwidth than the
  host can read" (for host-side readers) [D]. Bench `write` and `ce dir=d2h` measure both.

### 3.4 Latency and concurrency (Little's law)

* NVIDIA documents no GPU→LPDDR5X latency. HMAT: CPU→LPDDR5X 160 ns; GPU→HBM 370 ns; CPU→HBM 560 ns [M]. Fusco et al. on GH200: GPU→DDR latency
  ≈ Grace→HBM latency [T]. Measure with bench `chase`, per SM as well. The two dies probably sit at different distances from the C2C PHY; test this with `chase sm=all`.
* The C2C PMU gives the **Grace-side** read latency and the average number of outstanding reads (`RD_CUM_OUTS/RD_REQ`, `RD_CUM_OUTS/CYCLES`), excluding the link [D].
  Run it during inference with `bench/pmu.sh <out> --sleep 30`.
* Required bytes in flight = BW × latency. For example, 340 GB/s × 1.5 µs (guess, loaded) = 510 KB ≈ 8k 64-B lines in flight GPU-wide [I].
  The MoE kernel keeps 3–6 stages × 16 KiB of A-tile in flight per CTA (§3.6), so about 10–30 CTAs streaming host tiles should saturate DRAM.
  Bench `read/persm/tma/mix` find the real number.

### 3.5 Copy engines (DMA) vs SM loads (LDG/TMA)

* 3 async copy engines [M]. CEs consume no SM slots. NVIDIA: "NVLink-C2C with ATS leverages the Hopper DMA copy engines" (pageable copies) [D GH blog].
  System-allocated memory copies take a slower path [T]. Expected CE H2D ≈ 350–360 GB/s here [D], so a CE copy cannot beat an SM stream on bandwidth. CE's advantage is
  that **it does not hold SMs while waiting on host latency**.
* SM loads: per-SM bandwidth = (bytes in flight per SM) / latency. A host-bound CTA keeps its SM slot for the whole slow transfer, and the trtllm-gen MoE CTAs use about 120 KiB smem,
  so 1 CTA per SM [I]. bench `mix` measures how much HBM bandwidth the remaining SMs still get.

### 3.6 What the MoE kernel actually does (read from the cubins/headers in the image) [M]

* Production kernels (torch-profiler traces): `bmm_MxE4m3_MxE2m1MxE4m3_…_t128x32x256(u2)_s3|s5_…_c1x1x1|c2x1x1_…_schPd2x1x2x3_…_tma_ldgstsSf_rgTma_…swiGlu…` (GEMM1)
  and `bmm_Bfloat16_MxE2m1MxE4m3_…_t128x32x256_s3|s5_…_schPd…|schedS…` (GEMM2). The t128x16 / t128x8 variants are used at small batch.
* Weights are the A operand (`transOut`, weight rows = M). A is loaded by **TMA** (`TmaDescriptor.h`: `cuTensorMapEncodeTiled`, `CU_TENSOR_MAP_L2_PROMOTION_L2_128B`).
  The box is tileM 128 rows × tileK 256 fp4 = **128 rows × 128 B = 16 KiB**, with consecutive rows **2560 B apart (w13) / 1152 B apart (w2)**. Stages are 3–6, so 48–96 KiB in flight per CTA.
  GEMM1 scale factors are loaded with LDGSTS (cp.async), and the scales stay in HBM (plugin).
* All 1804 MXFP4 cubins are `_rM_TN_` (MajorK). There is no BlockMajorK (contiguous-box) MXFP4 kernel in the image, although FlashInfer's `WeightLayout.BlockMajorK` exists for BF16 [M].
  Whether the strided pattern costs LPDDR5X efficiency is bench `tma rowbytes=2560` vs `rowbytes=128`.
* TMA on host memory works, because this production path reads cold experts from EGM pages via TMA [M]. TMA on ATS-only `malloc` memory is untested (bench `tma mem=malloc`).

### 3.7 Why the MoE GEMM shows "~200–260 GB/s" while the link is 447

All of the following are [I] from [M] numbers:
1. **The 447 figure does not apply.** The DRAM behind the link is 360 (HMAT) to 396 (spec) GB/s. NVIDIA's own CE reference is 350–360.
2. **Inside the MoE kernels the host stream already runs at about 337 GB/s.** From the per-layer profile: the extra time with ~16 cold experts is 188.7 MB / 0.56 ms (GEMM1)
   and 94.4 MB / 0.28 ms (GEMM2), both **337 GB/s**. The all-HBM GEMM1 (0.39 ms) matches ~243 distinct experts × 11.25 MiB at 7.16 TB/s, so it is HBM-roofline-bound.
3. **The 200–260 GB/s numbers are averages over whole steps**, which include attention, mHC, engram, dspark and HBM-only layers that do no host reads. Stock layout at conc 32 [M]:
   1616.6 host experts/forward × 17.69 MB × 9.0 steps/s ≈ 257 GB/s (the C2C0 PMU says 258.1), which means the host stream is busy about 76% of the step at ~337. Production at
   conc 32 [M]: 501 host experts/forward → 8.9 GB/step × 19.4 steps/s ≈ 170 GB/s (PMU 170.3). That is about 26 ms of host streaming in a 51.5 ms step.
4. **Host time adds to HBM time instead of overlapping** (0.39 + 0.56 ≈ 0.95). One likely mechanism: a CTA waiting on host data holds its SM (1 CTA/SM), so while cold tiles
   are in flight, fewer SMs pull HBM. Tile order (cold experts clustered in the persistent schedule) and the kernel tail also play a role. The nsys GPU-metrics timeline settles this (§6).
5. Smaller effects to quantify: DRAM row locality of the 128-B strided TMA pattern, ~0.1–0.3 GB/s of CPU traffic, and engram gathers on the same DRAM.

---------------------------------------------------------------------------------------------------------------------------

## 4. Implications for optimization

### 4.1 (a) MoE expert GEMMs that read some experts from host

* **The ceiling per layer** is max(HBM bytes / 7.16 TB/s, host bytes / ~340 GB/s), not the sum. At 64 tokens with ~16 cold experts: 0.56 + 0.28 ms instead of
  0.95 + 0.48 ms, which saves up to ~0.6 ms per layer (~40% of MoE GEMM time) if overlap were perfect [I]. Even then the host part dominates: 6.6% of active experts cost
  more time than the other 93%. **Cold bytes per step remain the first-order term.**
* **Overlap options** that keep numerics identical when the per-expert math is unchanged; still run the bit-identical prefill check because finalize order can change:
  1. *SM partition*: run a cold-only TRT-LLM call on N SMs with a deep-stage tactic, concurrently with the hot call on 152−N SMs. Use CUDA green contexts or stream-priority partitioning.
     Cubins with deeper stages exist (s6/s9 for t128x16/x32). The earlier 2-stream split failed with a shallow tactic and no SM limit. bench `mix` gives the minimal N that reaches ~340 GB/s
     while the HBM kernel keeps most of its bandwidth.
  2. *CE staging*: after routing, copy the step's cold experts (each w13 is a contiguous 11.25 MiB, so the copies are efficient) into HBM staging slots on a copy engine while the hot
     GEMM runs. Then run the cold GEMM from HBM, with routing ids remapped to staging slots. This uses zero SMs for host latency. bench `ce` and `cemix` give the rate and the interference.
  3. *Tile ordering*: keep the number of SMs that hold host tiles at any moment just at the Little's-law minimum. Verify on the nsys timeline that HBM DRAM% and
     "L2 Bandwidth to System/Peer Memory" are high **at the same time** during the MoE kernels.
* **Re-read amplification:** host lines are not cached in L2, so experts whose token count exceeds tileN (prefill, large batches) re-read their weights over C2C for every N-tile.
  For those experts, staging (copy once) is strictly better [I].
* The host stream itself (337 → 360–380 GB/s) is worth at most ~7–15%. Pursue it only if bench `tma` shows the strided A pattern losing more than ~10% versus contiguous.

### 4.2 (b) Prefetching cold experts into HBM

* Bandwidth budget [I]: production conc 32 moves 8.9 GB/step of cold weights; at ~337 GB/s that is ~26 ms of a 51.5 ms step. **About 25 ms per step of link and DRAM time is idle**
  (attention, engram, dspark, HBM-only work). Prefetch can only help by **moving** transfers into that idle time, or by caching (reuse across steps).
  It does not create bandwidth. A wrong prefetch costs as much DRAM time as a needed read.
* Use copy engines, not SM copy kernels, for prefetch: they use no SM slots, there are 3 of them, and transfers ≥ 2 MiB amortize per-copy overhead (bench `ce` gives the curve). Use pinned EGM
  sources. Avoid pageable or ATS sources, which take a slower path [T].
* Predictability decides everything. Before building a predictor, measure from the plugin's GPU counters: P(cold expert used at step t+1 | used at step t, same layer), and the accuracy of
  layer-ahead gate prediction (layer L−1 hidden state → layer L router).
* Evictions: weights are immutable, so an evicted expert never needs to be written back if a host copy is kept. Host↔HBM writes share the same LPDDR5X budget,
  and D2H is the slower direction [D]. Current swap traffic is ≤ 64 × 2 × 16.9 MiB per 3 s ≈ 0.7 GB/s, which is negligible today.
* Expert-cache tuning that follows from the ceiling: each cold expert-touch costs ≈17.7 MB / 337 GB/s ≈ **52 µs** of serialized DRAM time [I].
  That is the right unit for weighing an HBM-resident expert against KV-cache capacity.

### 4.3 (c) Grace CPU computing cold experts (Fiddler-style)

* Same DRAM: the CPU streams LPDDR5X at ≤360–396 GB/s, with an NVIDIA TRIAD reference of ≥307 at 72 threads. The GPU's in-kernel host stream is already ~337 GB/s. **There is no bandwidth gain.**
  CPU + GPU reading together cannot exceed the DRAM [D, I]. bench `cpugpu` verifies this.
* Compute is not the limit. MXFP4 decode is TBL lookup + SDOT, roughly 8–9 NEON instructions per 16 weight bytes at T=1. Across 4 pipes × 72 cores × 3.37 GHz that is about 1.8 TB/s of weights,
  far above DRAM [I]. `bench/cpu_mxfp4` measures the real rate; it emulates the exact expert shapes and its NEON kernel is verified against a scalar reference.
* Costs: not bit-identical with the GPU MXFP4 kernel (fails the prefill check); a per-layer GPU→CPU→GPU handoff ×40 layers (cheap via coherent flags, but it has to fit inside CUDA graphs:
  `cuStreamWaitValue`/write-value memops are supported [M]); and the CPU shares the 1600 W module budget with the GPU [D].
* The only real benefit is that GPU SMs stop waiting on host latency, and §4.1 (SM partition, CE staging) gets the same benefit with identical numerics. **Deprioritize.**
  Revisit only if `mix`/`cemix` show the GPU cannot overlap host and HBM streams.

### 4.4 (d) Engram lookups from host memory

* Model config [M]: engram layers {1, 14}, ~384M rows each, head_dim 256 fp8 (+ E8M0 scales) ≈ **189 GiB** pinned on host; 8 hash heads, n-gram ≤ 4.
  A decode step reads a few thousand random **256-B rows**, a few MB per step, from a prefetch side stream (`specialized_decode/engram/prefetch.py`).
  This is latency-bound, not bandwidth-bound [I].
* Allocation: the default is `torch.empty(pin_memory=True)` (cudaHostAlloc); the `use_thp` option is mmap + MADV_HUGEPAGE (512 MiB THP) + cudaHostRegister [M]. Random rows across 189 GiB
  mostly miss the GPU TLB, so page size matters a lot (2 MB EGM / huge pinned pages vs 64 KiB) [D TLB statement, I]. bench `gather` and `chase stride=64K/2M` compare
  egm / hostalloc / register_thp / malloc.
* It competes with cold-expert streaming for LPDDR5X only marginally in bytes, but random 64–256 B accesses are DRAM-inefficient. On the nsys trace, check whether the
  layer ever waits on the engram prefetch event. If it does not, leave engram alone.

---------------------------------------------------------------------------------------------------------------------------

## 5. What not to assume (generic-PC reflexes that are wrong here)

* "Host↔GPU is PCIe, so batch big DMA copies": the path is coherent C2C. Kernels can load host memory directly at DRAM speed with TMA/LDG, and CE ≈ SM-load bandwidth.
* "The link is the bottleneck (447)": LPDDR5X (360–396) is. Reads and writes share it, and the CPU shares it.
* "L2 will cache re-used host data": it will not [D].
* "Pinned = 4 KiB pages": the kernel uses 64 KiB pages and THP is 512 MiB; EGM uses 2 MB GPU pages; ATS translations go through the Grace SMMU.
* "GPU memory is invisible to Linux": in NUMA mode HBM is node 1 and Linux can allocate there unless you bind to node 0.
* "nvidia-smi / DCGM shows C2C": GPU-side C2C counters read 0 here [M]. Use the Grace uncore PMUs (C2C0/C2C1/SCF) and nsys `--cpu-socket-metrics`.

---------------------------------------------------------------------------------------------------------------------------

## 6. Monitoring with Nsight Systems (beyond nvidia-smi)

* Permissions [M]: GPU metrics need admin (`RmProfilingAdminOnly: 1`), so run nsys as root in a privileged container. The benchmark user is in the docker group, so no sudo is needed.
  Grace socket metrics need `perf_event_paranoid ≤ 0` (now -1; if it resets, `sudo sysctl kernel.perf_event_paranoid=-1`).
* GPU metric set **`gb11x-ct`** (chips GB110/GB112): *SMs Active, SM Tensor Pipe Active, DRAM Read/Write Bandwidth, **L2 Bandwidth to System/Peer Memory**
  (`syslts__t_sector_throughput`, the GPU side of C2C traffic), CTC RX/TX, PCIe, NVLink, warp stalls…* [M nsys config]. Maximum sampling is 200 kHz (5 µs).
* Grace socket metrics (`--cpu-socket-metrics`, `--event-sample=system-wide`, interval ≥ 1 ms): 1 CPU-LPDDR5X read BW, 10 util %, 14 latency, **100/103/105 C2C0 read BW / util % /
  latency**, 128/133 C2C1 [M nsys help].
* `bench/nsys_system.sh <label> 20 50000`: system-wide sampling while `bench.sh` drives the live server. No restart needed.
* `bench/serve_nsys.sh <label>`: starts dsv41-tune under nsys with `--cuda-graph-trace=node` (decode runs in CUDA graphs) and
  `--capture-range=cudaProfilerApi`, using vLLM `--profiler-config '{"profiler":"cuda"}'`. Capture with `curl -X POST :8000/start_profile` … `stop_profile`.
* `bench/nsys_metrics.py <sqlite> [kernel-regex]` prints every metric overall and **inside the MoE GEMM kernels**. The overlap diagnosis is:
  in-kernel DRAM-read % and System/Peer % both high at the same samples = overlapped; alternating = serialized (§3.7.4).
* `bench/pmu.sh <out> --sleep 30` prints per-second C2C0 GB/s, Grace-side read latency, outstanding reads, bytes per request, CPU DRAM traffic, total LPDDR5X, and SMMU ATS translation rate.

---------------------------------------------------------------------------------------------------------------------------

## 7. Microbenchmark plan (`~/dsv41/tuning/arch/bench/`, run on 224 when the GPU is idle)

Build (already compiled; `run_all.sh` rebuilds when a source is newer):
```
cd ~/dsv41/tuning/arch/bench
/usr/local/cuda-13.2/bin/nvcc -O3 -std=c++17 -arch=sm_103a -lineinfo -o c2cbench c2cbench.cu -lcuda -lpthread
gcc -O3 -mcpu=neoverse-v2 -fopenmp -o cpu_bw cpu_bw.c ; gcc -O3 -mcpu=neoverse-v2 -fopenmp -o cpu_mxfp4 cpu_mxfp4.c -lm
```
Run (stop production first: `docker stop dsv41-moe`; it refuses if a dsv41* container is up or GPU memory > 2 GiB; afterwards `~/dsv41/moe/serve-moe.sh`):
```
QUICK=1 ./run_all.sh            # ~10-15 min subset
./run_all.sh                    # full, ~1-1.5 h (64 GiB host buffers for chase/gather: needs the server stopped)
ONLY="mix cemix" ./run_all.sh   # selected sections
```
Output goes to `~/dsv41/tuning/arch/results/<ts>/`: `<name>.results` (RESULT lines), `<name>.pmu.txt` (Grace PMU-derived), and `index.txt` (everything in order).
Every section runs under `pmu.sh`, so each configuration gets C2C bytes, bytes per request, Grace-side latency, outstanding reads, and ATS translation counts.

| section | command (see run_all.sh) | question | decision it drives |
|---|---|---|---|
| read | `c2cbench read mem=<kind> vec=16 unroll=1,4,8 ctas=1..608` | max SM-load host BW per allocation kind (egm, hostalloc, register, register_thp, malloc(ATS), malloc_thp) and HBM; CTAs and bytes in flight needed | how many SMs a cold stream needs; best allocation kind |
| persm | `read ctas=1 threads=128..1024 unroll=1..8`, `tma ctas=1 stages=1..12` | per-SM in-flight capacity (= BW × latency) for LDG and TMA | pipeline depth for a cold-only kernel |
| write | `write mem=egm,hostalloc,hbm` | SM stores to host | eviction or write-back cost |
| tma | `tma rowbytes=2560,1152,128 stages=1..12 ctas=8..152` | MoE A-operand boxes (w13/w2 stride) vs contiguous 16 KiB boxes, from EGM and HBM | whether a contiguous cold-expert layout or custom kernel is worth it |
| rowseg | same pattern with LDG | separate TMA-engine effects from DRAM-pattern effects | |
| l2 | `l2 size=16M,64M passes=1,2,8` (+PMU C2C bytes) | is host data cached in GPU L2? (expect no [D]) | re-read amplification and staging policy |
| slc | `slc cpuwarm=0/1` (SCFCACHE=1 PMU) | can GPU reads hit Grace's 114 MiB L3? | CPU-produced data hand-off cost |
| chase | `chase stride=64 / 2M / 64K, foot=1M..64G`, `sm=all` | latency HBM vs host; TLB reach per kind (EGM 2 MB vs 64 KiB vs ATS vs THP); die asymmetry | engram allocation; stage count; SM affinity |
| ce | `ce dir=h2d,d2h chunk=64K..1G streams=1,2,3` | copy-engine BW vs chunk size, engines, source kind (incl. pageable/ATS) | prefetch implementation |
| mix | `mix hostctas=4..96` (1 CTA/SM via 120 KiB smem) | **host stream on N SMs + HBM stream on 152−N SMs at once** | SM-partition design (§4.1) |
| cemix | CE H2D concurrent with HBM-read kernel / host-read kernel | prefetch overlap, LPDDR5X contention | CE staging viability |
| gather | `gather foot=64G rowbytes=256 ctas=1..608` | engram-like random rows: rows/s, implied latency per kind | engram allocation |
| cpu | `cpu_bw` 1..72 threads; `cpu_mxfp4 11 {1,2,4,8} 96 40` | Grace DRAM BW; CPU MXFP4 expert throughput vs the 337 GB/s GPU equivalent | Fiddler go/no-go (§4.3) |
| cpugpu | `cpu_bw` 72 thr concurrent with GPU EGM read | do CPU + GPU share one DRAM ceiling? | confirms §0.1 |

Expected results (hypotheses to falsify):
* EGM/hostalloc read peaks at about **330–360 GB/s**, well under 447, saturating at ~16–48 CTAs with deep unroll.
* malloc (ATS) matches that for streaming but is far worse for random 64 KiB-page access.
* HBM ~6–6.8 TB/s.
* CE H2D ≈ 350 GB/s; D2H lower.
* Host passes in `l2` are no faster than pass 1.
* `tma rowbytes=2560` within ~10% of `rowbytes=128`. If not, the layout matters.
* In `mix`, ~24–32 host CTAs reach the DRAM ceiling while the HBM kernel keeps ≳80% of its bandwidth. If true, the SM partition is the main kernel-level win.

Other tools:
* `devquery` (no context, safe anytime): all `CUdevice_attribute`s plus VMM granularities.
* `nsys_system.sh`, `serve_nsys.sh`, `nsys_metrics.py` (§6).
* `pmu.sh`, `pmu_parse.py`.

---------------------------------------------------------------------------------------------------------------------------

## 8. Open questions (what the benches and nsys must answer)

1. Is the host stream serialized with the HBM stream inside the MoE kernels, and why? (nsys gb11x-ct in-kernel timeline; `mix`)
2. What is the real ceiling of GPU reads of LPDDR5X (EGM, sequential), and how much does the 128-B strided TMA box pattern lose? (`read`, `tma`)
3. What is GPU→LPDDR5X latency, idle and loaded, and does it differ between the two dies? (`chase`, `chase sm=all`, C2C PMU latency)
4. What page size do pinned (cudaHostAlloc / cudaHostRegister) mappings get, and how big is the GPU TLB reach? (`chase stride=64K/2M` footprint sweeps)
5. Can a single copy engine reach the DRAM ceiling, and how much does CE traffic slow an HBM-bound kernel? (`ce`, `cemix`)
6. Do GPU reads ever hit the Grace L3? (`slc`)

---------------------------------------------------------------------------------------------------------------------------

## 9. Sources

NVIDIA documentation:
* DGX Station product specs (252 GB HBM3e 7.1 TB/s; 496 GB LPDDR5X 396 GB/s; NVLink-C2C 900 GB/s): https://www.nvidia.com/en-us/products/workstations/dgx-station/
* DGX Station Development Guide: overview https://docs.nvidia.com/dgx/dgx-station-development-guide/overview.html ·
  mixed coherency (GPU access to CPU memory not cached; CDMM vs NUMA mode; ATS) https://docs.nvidia.com/dgx/dgx-station-development-guide/coherency.html ·
  optimization https://docs.nvidia.com/dgx/dgx-station-development-guide/optimization.html ·
  page size https://docs.nvidia.com/dgx/dgx-station-development-guide/porting/pagesize.html ·
  power sloshing (1600 W shared) https://docs.nvidia.com/dgx/dgx-station-development-guide/dynamic-power-sloshing.html
* Inside NVIDIA Blackwell Ultra (160 SMs full die, 256 KB TMEM, 288 GB/8 TB/s full B300, C2C 900 GB/s): https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era
* Blackwell architecture (two dies, 10 TB/s chip-to-chip): https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
* Grace Performance Tuning Guide: https://docs.nvidia.com/grace-perf-tuning-guide/index.html ·
  PMU events and formulas https://docs.nvidia.com/grace-perf-tuning-guide/measuring-performance.html ·
  OS settings (64K pages, THP 2M mTHP, init_on_alloc, AutoNUMA, CDMM) https://docs.nvidia.com/grace-perf-tuning-guide/os-settings.html ·
  health checks (STREAM ≥307 GB/s @72T 480GB; CE H2D ≈350–360 GB/s) https://docs.nvidia.com/grace-perf-tuning-guide/health-checks.html ·
  system (NUMA layout of GH/GB) https://docs.nvidia.com/grace-perf-tuning-guide/system.html
* Linux kernel, NVIDIA Tegra241 (Grace) uncore PMU (C2C0 = ATS/EGM, C2C1 = untranslated): https://docs.kernel.org/admin-guide/perf/nvidia-tegra241-pmu.html
* CUDA Programming Guide, unified and system memory: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html ·
  unified memory and HW coherency (cache-line coherence, 2 MiB page preference, TLB misses expensive): https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html ·
  Extended GPU Memory (cuMemCreate HOST_NUMA, 2 MB pages): https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/04-special-topics/extended-gpu-memory.html
* CUDA Driver API tensor maps (no memory-type restriction on globalAddress; L2 promotion): https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
* NVIDIA Grace Hopper Superchip Architecture In-Depth (C2C 450 GB/s/dir, ATS, EGM, CE): https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/

Third-party (GH200, priors only):
* Fusco et al., "Understanding Data Movement in Tightly Coupled Heterogeneous Systems: A Case Study with the Grace Hopper Superchip": https://arxiv.org/html/2408.11556v1
* Schieffer et al., "Harnessing Integrated CPU-GPU System Memory for HPC: a first look into Grace Hopper": https://arxiv.org/html/2407.07850v1

On-box evidence: `~/dsv41/tuning/arch/results/inspect-20261103.txt` (numactl, lscpu, caches, HMAT, pages, driver/UVM params, IOMMU, C2C links, PMUs),
`results/devquery.txt` (all device attributes + VMM granularity), `~/dsv41/tuning/results/bw-sequence.log` (C2C0 PMU vs plugin accounting, 2026-11-03),
FlashInfer cubin names / `trtllmGen_bmm_export/TmaDescriptor.h` in image `vllm/vllm-openai@sha256:d825baf9…`.
