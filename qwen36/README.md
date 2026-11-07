# TidalMoE on Qwen3.6-35B-A3B (consumer PC)

TidalMoE was built for DeepSeek-V4.1 on a GB300 superchip. This folder ports its core idea, per-expert hot/cold
placement plus the adaptive expert cache, to `nvidia/Qwen3.6-35B-A3B-NVFP4` on a Windows desktop with one
RTX 5060 Ti 16 GB. It runs on stock vLLM v0.33.0 in Docker Desktop (WSL2) and needs no vLLM rebuild.

The 21.8 GiB checkpoint does not fit in 16 GB of VRAM. Expert weights that are not in VRAM are read over PCIe 4.0 x8
(about 12 GB/s), roughly 30x slower than the GB300's NVLink-C2C. That makes the choice of which experts stay in VRAM
matter more here than it did on the GB300.

## Results

Output tokens/s from `vllm bench serve`: 256 output tokens, EOS ignored, greedy, warm server, the adaptive cache
warmed on ShareGPT. `random` uses 512-token random prompts; `sharegpt` uses ShareGPT first turns on the chat API.

| Configuration | sharegpt c1 | c4 | c8 | random c1 | c4 | c8 |
|---|---:|---:|---:|---:|---:|---:|
| Stock vLLM, per-layer UVA expert offload (`-Placement off`) | 32.0 | 41.1 | 48.4 | 30.1 | 43.1 | 51.4 |
| TidalMoE port, same 8.2 GiB of experts in VRAM | 74.0 | 98.0 | 115.4 | 41.2 | 99.2 | 131.3 |
| **TidalMoE port, 9.5 GiB of experts in VRAM (default)** | **85.7** | **126.2** | **150.0** | **52.0** | **109.9** | **167.1** |
| Port + DSpark drafter, k=8, 7 GiB | 37.0 | 57.0 | 67.2 | 26.1 | 44.2 | 46.5 |
| Port + DSpark drafter, k=2, 7 GiB | 45.0 | 67.2 | 91.0 | | | |

With the same expert VRAM budget, the port is 2.3x to 2.6x faster than stock vLLM. Most of the gain comes from where
expert reads land:

- **Stock:** about 50% of expert reads go to host memory, because whole layers are offloaded.
- **Port with 8.2 GiB:** about 12% of reads go to host memory.
- **Port with 9.5 GiB:** about 9%.

The port's 9.5 GiB configuration takes that extra memory from the KV cache, which still holds about 160k tokens
(2.4 full 64k-token requests).

For reference, the dense `nvidia/Qwen3.6-27B-NVFP4` ran at 1.8 tok/s on this PC. Every one of its tokens has to stream
7.4 GB of offloaded layers over PCIe. Those copies have been deleted.

**DSpark (`RedHatAI/Qwen3.6-35B-A3B-speculator.dspark`) works but is slower on this machine.** On ShareGPT it accepts
about 4 tokens per step at k=8 and about 2.5 at k=2, which matches its model card. The verify step costs too much,
though:

- **More cold experts per step:** verifying k+1 tokens per request touches many more distinct experts, so many more
  cold experts cross PCIe.
- **Attention metadata is rebuilt every step:** vLLM rebuilds the drafter's attention metadata on every step with the
  FlashInfer backend.

A k=2 step takes about 54 ms, against 11 to 13 ms for a plain decode step. TidalMoE's DSpark adaptive verification with
a measured cost curve (`MOE_AV_*` in `plugin/moe_placement`) has not been ported yet; it is the next thing to try.

## How it works

`plugin/tidal_qwen` is a vLLM general plugin, loaded through `PYTHONPATH` and its entry point the same way as
`plugin/moe_placement`:

1. **Placement.** After loading, when the NVFP4 Marlin repack is done and before vLLM profiles memory, each of the 40
   `Qwen3NextSparseMoeBlock` layers is rewritten:
   - Experts are permuted so the hot ones come first, and the router (`GateLinear`) rows are permuted the same way, so
     routing is unchanged.
   - Each large per-expert tensor (w13/w2 FP4 weights and their FP8 block scales) is rebuilt on one CUDA virtual-address
     range. VRAM pages back the hot head and pinned host pages back the cold tail (`cuMemCreate`/`cuMemMap`, host NUMA
     location; this also works under WSL2).
   - The Marlin kernel reads the result as an ordinary tensor.
2. **Adaptive expert cache.** A capture-safe counter in `MarlinExperts.apply` records which slots each MoE call touches.
   Every 3 seconds, cold experts used significantly more than hot ones are swapped in place across all per-expert
   tensors and router rows (same z-test as TidalMoE), so CUDA graphs stay valid.
   - The slot order and scores are saved to the `qwen36-state` volume and restored on the next start.
   - When `-HotGb` changes, the saved scores are re-ranked into a new plan.
3. **Embedding in host memory.** `plugin/host_embed` keeps the 0.95 GiB token embedding in pinned host memory and reads
   it through a UVA view. This is TidalMoE's `MOE_HOST_PARAMS` idea.
4. **vLLM 0.33 bug fix (always on).** With `--cpu-offload-gb`, the ModelOpt NVFP4 MoE repack goes through
   `replace_parameter`, which rebinds the offloaded Parameter's `.data` to the repacked tensor in VRAM. The Parameter
   keeps `_vllm_is_uva_offloaded=True`, so vLLM never moves it back to host memory. All "offloaded" experts end up in
   VRAM, and the Windows driver silently spills the overflow to system memory. The plugin re-offloads them right after
   each layer's repack. Without this fix, stock vLLM cannot start at all with 64k context on this card.

Not ported, because these parts are specific to DeepSeek or the B300: forced TRT-LLM tactics, the SM100 persistent MoE
kernels, mHC, and the kernel-aware cold groups. On SM120, vLLM 0.33 uses Marlin for NVFP4 (FP4 weights, BF16 math).

**Correctness.** Greedy outputs match stock vLLM token for token until a near-tie, where the top token's probability is
about 0.56 to 0.68. Both continuations are fluent and equivalent. Permuting experts changes the order in which Marlin
sums expert outputs, and even an unchanged configuration is not bitwise deterministic from run to run (Marlin uses
atomic adds). `qwen36/greedy_check.py` runs this check.

## Setup on this PC (done once)

1. **WSL2.** In an admin PowerShell, run `wsl --install --no-distribution`, then reboot. `%UserProfile%\.wslconfig`
   caps the VM at `memory=32GB`. The default of 48 GB let the VM's page cache starve Windows.
2. **Docker Desktop.** Pull the image with `docker pull vllm/vllm-openai:v0.33.0`.
3. **Model, drafter and dataset in Docker volumes.** Download straight into the volumes; copying from the Windows HF
   cache runs at only about 40 MB/s:

   ```powershell
   docker run --rm -v hf-cache:/root/.cache/huggingface --entrypoint hf vllm/vllm-openai:v0.33.0 download nvidia/Qwen3.6-35B-A3B-NVFP4
   docker run --rm -v hf-cache:/root/.cache/huggingface --entrypoint hf vllm/vllm-openai:v0.33.0 download RedHatAI/Qwen3.6-35B-A3B-speculator.dspark config.json model.safetensors
   docker run --rm -v qwen36-datasets:/datasets --entrypoint hf vllm/vllm-openai:v0.33.0 download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset --local-dir /datasets
   ```

## Use

```powershell
.\qwen36\serve.ps1                       # TidalMoE port on :8000, ready in ~4 min (placement takes ~60 s of that)
curl.exe -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" `
  -d '{\"model\":\"nvidia/Qwen3.6-35B-A3B-NVFP4\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'
.\qwen36\bench.ps1 -Dataset sharegpt -Conc "1,4,8" -OutputLen 256 -Label mytest
docker logs qwen36 2>&1 | Select-String "tidal_qwen\[adapt\]"   # host share of expert reads, swaps per epoch
```

| `serve.ps1` switch | Default | Meaning |
|---|---|---|
| `-Placement` | `apply` | `off` = stock vLLM layer offload; `inspect` = log the MoE tensor layout only |
| `-HotGb` | `9.5` | VRAM for routed experts (weights and scales, about 1.69 MiB per expert) |
| `-NoAdapt` | off | static placement, no swaps |
| `-Spec dspark -SpecTokens k` | off | DSpark drafter (needs about 2 GiB of VRAM: use `-HotGb 7`) |
| `-MaxModelLen`, `-MaxSeqs`, `-GpuUtil` | 65536, 8, 0.91 | as in vLLM |
| any `MOE_*` env var | | passed to the plugin (see `plugin/tidal_qwen/__init__.py`) |

## Caveats

- **Run Docker commands from PowerShell, or set `MSYS_NO_PATHCONV=1`.** Git Bash rewrites container paths such as
  `/datasets` into Windows paths.
- **Pass lists as quoted strings.** Windows PowerShell 5.1 strips quotes inside JSON arguments, so `serve.ps1` uses
  vLLM's dotted flags (`--speculative-config.method dspark`). `powershell -File` also reads `1,4,8` as the number 148,
  so `bench.ps1` takes `-Conc "1,4,8"` as a string.
- **The first start after changing `-HotGb` begins from re-ranked scores.** The cache keeps adapting for a minute or two
  of traffic; the log shows the host share of expert reads.
- **The plugin depends on vLLM 0.33 internals:** the `MoERunner.routed_experts` layout, `MarlinExperts.apply`, and
  `GPUModelRunner.execute_model`. Re-check it with `-Placement inspect` after upgrading.
