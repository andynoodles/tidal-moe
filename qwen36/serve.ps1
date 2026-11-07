<#
.SYNOPSIS
  Serve nvidia/Qwen3.6-35B-A3B-NVFP4 with stock vLLM v0.33.0 (Docker Desktop on WSL2) on one RTX 5060 Ti 16 GB.

.DESCRIPTION
  The checkpoint is 21.8 GiB (NVFP4 routed experts 16.9 GiB, 256 experts x 40 layers, 8 active per token) and the GPU
  has 16 GB. TidalMoE (this repo), ported to Qwen3.6 on a consumer PCIe card (plugin/tidal_qwen, plugin/host_embed):
    * -Placement apply (default)  per-expert hot/cold placement: the HotGb hottest experts of every layer in VRAM, the
                                  rest in pinned host RAM read zero-copy over PCIe, router rows permuted to match, plus
                                  the adaptive expert cache (online swaps; state kept in the qwen36-state volume)
    * host_embed plugin           token embedding (0.95 GiB BF16) in pinned host RAM, gathered through a UVA view
    * --language-model-only       skip the vision tower (0.83 GiB); the MTP head (1.57 GiB) is only loaded with -Spec mtp
    * load-time UVA offload       OffloadGb of experts in host RAM while loading (so loading fits); with -Placement
                                  apply the plugin then re-places every layer. -Placement off = stock vLLM's per-layer
                                  UVA offload (with the plugin's fix for vLLM 0.33 re-materializing repacked experts)
    * VLLM_WSL2_ENABLE_PIN_MEMORY vLLM turns pinned memory off under WSL2 by default; offload needs it
  The RTX 2060 (Turing) has no FP4/FP8 kernels, so only the RTX 5060 Ti is used (CUDA index 1 in PCI bus order).
  The model is read from the Docker volume $HfVolume (HF cache layout); see qwen36/README.md.

.EXAMPLE
  .\qwen36\serve.ps1                      # TidalMoE port, 9.5 GiB hot experts, container "qwen36" on :8000
  .\qwen36\serve.ps1 -Placement off       # stock vLLM layer offload (baseline)
  .\qwen36\serve.ps1 -Spec dspark -SpecTokens 2 -HotGb 7   # + RedHatAI DSpark drafter (slower on this PC, see README)
  .\qwen36\serve.ps1 -DryRun              # print the docker command only
  .\qwen36\serve.ps1 --max-num-seqs 16    # extra args go to `vllm serve`
#>
[CmdletBinding(PositionalBinding = $false)]
param(
  [string]$Name = "qwen36",
  [int]$Port = 8000,
  [string]$Image = "vllm/vllm-openai:v0.33.0",
  [string]$Model = "nvidia/Qwen3.6-35B-A3B-NVFP4",
  [string]$HfVolume = "hf-cache",
  [string]$Gpu = "1",                      # CUDA index in PCI bus order: 0 = RTX 2060, 1 = RTX 5060 Ti
  [int]$MaxModelLen = 65536,
  [int]$MaxSeqs = 8,
  [double]$GpuUtil = 0.91,                 # of 16 GB; Windows' desktop uses 0.3-1 GB of this GPU
  [double]$OffloadGb = 8.5,                # routed experts in host RAM (~0.42 GiB per layer); 0 = none
  [string]$OffloadParams = "experts",      # parameter-name segment(s) to offload, space separated
  [int]$BatchedTokens = 2048,              # prefill chunk; smaller = less activation memory
  [ValidateSet("off", "inspect", "apply")]
  [string]$Placement = "apply",            # tidal_qwen plugin: TidalMoE per-expert hot/cold placement; off = stock
  [double]$HotGb = 9.5,                    # with -Placement apply: VRAM for routed experts (weights + scales)
  [switch]$NoAdapt,                        # with -Placement apply: no online hot/cold swaps
  [ValidateSet("", "dspark", "mtp")]
  [string]$Spec = "",                      # speculative decoding: dspark = RedHatAI DSpark drafter, mtp = built-in head
  [int]$SpecTokens = 8,
  [string]$SpecModel = "RedHatAI/Qwen3.6-35B-A3B-speculator.dspark",
  [switch]$NoHostEmbed,
  [switch]$DryRun,
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Extra
)
$ErrorActionPreference = "Stop"
$PluginDir = Join-Path $PSScriptRoot "plugin"

# Dotted --x.y flags instead of JSON: Windows PowerShell 5.1 strips the quotes inside JSON arguments.
$vllmArgs = @(
  $Model,
  "--language-model-only",
  "--max-model-len", $MaxModelLen,
  "--max-num-seqs", $MaxSeqs,
  "--max-num-batched-tokens", $BatchedTokens,
  "--gpu-memory-utilization", $GpuUtil,
  "--kv-cache-dtype", "fp8_e4m3",
  "--reasoning-parser", "qwen3",
  "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
  "--host", "0.0.0.0", "--port", "8000"
)
if ($OffloadGb -gt 0) {
  $vllmArgs += @("--offload-backend", "uva", "--cpu-offload-gb", $OffloadGb, "--cpu-offload-params") + ($OffloadParams -split " ")
}
if ($Spec -eq "dspark") {
  if (-not $SpecModel.StartsWith("/")) {
    # HF_HUB_OFFLINE=1: vLLM maps the target's repo id to its cache snapshot but not the drafter's, so pass the path
    $cacheDir = "/root/.cache/huggingface/hub/models--" + ($SpecModel -replace "/", "--")
    $rev = docker run --rm -v "${HfVolume}:/root/.cache/huggingface" --entrypoint cat $Image "$cacheDir/refs/main"
    if (-not $rev) { throw "$SpecModel is not in the $HfVolume volume (download it first, see qwen36/README.md)" }
    $SpecModel = "$cacheDir/snapshots/$($rev.Trim())"
  }
  $vllmArgs += @("--speculative-config.model", $SpecModel, "--speculative-config.method", "dspark",
                 "--speculative-config.num_speculative_tokens", $SpecTokens)
} elseif ($Spec -eq "mtp") {
  $vllmArgs += @("--speculative-config.method", "mtp", "--speculative-config.num_speculative_tokens", $SpecTokens)
}
if ($Extra) { $vllmArgs += $Extra }

$hostEmbed = if ($NoHostEmbed) { "0" } else { "1" }
# tidal_qwen: MOE_* knobs (see plugin/tidal_qwen/__init__.py); any MOE_* variable set in this shell is passed through.
$moeEnv = @{ MOE_PLACEMENT = $Placement }
if ($Placement -eq "apply") {
  $moeEnv.MOE_HOT_GB = "$HotGb"
  if (-not $NoAdapt) {
    $moeEnv.MOE_ADAPT = "1"
    $moeEnv.MOE_ADAPT_SAVE = "/moe/state/adapt-hot$HotGb.json"
    $moeEnv.MOE_ADAPT_RESUME = "/moe/state/adapt-hot$HotGb.json"
  }
}
Get-ChildItem env: | Where-Object { $_.Name -like "MOE_*" } | ForEach-Object { $moeEnv[$_.Name] = $_.Value }
$envArgs = @(); foreach ($k in ($moeEnv.Keys | Sort-Object)) { $envArgs += @("-e", "$k=$($moeEnv[$k])") }
$dockerArgs = @(
  "run", "-d", "--name", $Name, "--gpus", "all", "--ipc=host", "-p", "${Port}:8000",
  "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "-e", "CUDA_VISIBLE_DEVICES=$Gpu",
  "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_WSL2_ENABLE_PIN_MEMORY=1",
  "-e", "PYTHONPATH=/plugins", "-e", "HOST_EMBED=$hostEmbed",
  "-v", "${HfVolume}:/root/.cache/huggingface",
  "-v", "qwen36-vllm-cache:/root/.cache/vllm", "-v", "qwen36-flashinfer:/root/.cache/flashinfer",
  "-v", "qwen36-triton:/root/.triton", "-v", "qwen36-state:/moe/state", "-v", "qwen36-datasets:/datasets:ro",
  "-v", "${PluginDir}:/plugins:ro"
) + $envArgs + @($Image) + $vllmArgs

if ($DryRun) { Write-Output ("docker " + ($dockerArgs -join " ")); exit 0 }

$clash = docker ps --format "{{.Names}} {{.Ports}}" | Where-Object { $_ -notmatch "^$Name " -and $_ -match ":$Port->" }
if ($clash) { throw "another container already publishes port ${Port}: $clash" }
if (docker ps -aq --filter "name=^${Name}$") { docker rm -f $Name | Out-Null }
& docker @dockerArgs | Out-Null
if ($LASTEXITCODE -ne 0) { throw "docker run failed" }
Write-Output "started $Name on :$Port ($Model, load-time expert offload $OffloadGb GiB, host embed $hostEmbed, placement $Placement$(if ($Placement -eq 'apply') { " hot $HotGb GiB adapt $(-not $NoAdapt)" }), spec '$Spec')"
Write-Output "logs: docker logs -f $Name    ready: curl.exe -s localhost:$Port/health    bench: .\qwen36\bench.ps1"
