<#
.SYNOPSIS
  Simple output tok/s benchmark against the running qwen36 server, using `vllm bench serve` inside the container.

.DESCRIPTION
  One run per concurrency with PromptsPerConc x concurrency prompts (at least 3), OutputLen tokens out (EOS ignored),
  greedy. Dataset:
    random    random-token prompts of InputLen tokens on the completions API
    sharegpt  ShareGPT first turns on the chat API (the README workload of the TidalMoE repo); needs the
              qwen36-datasets volume (see qwen36/README.md), mounted by serve.ps1 at /datasets
  Results are copied to results\<Label>\ and summarised as output tok/s, mean TPOT and mean TTFT.

.EXAMPLE
  .\qwen36\bench.ps1                                   # random, concurrency 1, 4, 8
  .\qwen36\bench.ps1 -Dataset sharegpt -Conc "1,8" -OutputLen 256 -Label dspark
#>
[CmdletBinding(PositionalBinding = $false)]
param(
  [string]$Name = "qwen36",
  [string]$Model = "nvidia/Qwen3.6-35B-A3B-NVFP4",
  [ValidateSet("random", "sharegpt")]
  [string]$Dataset = "random",
  [string]$Conc = "1,4,8",                 # comma-separated (a string: -File would parse 1,4,8 as 148)
  [int]$InputLen = 512,
  [int]$OutputLen = 128,
  [int]$PromptsPerConc = 2,
  [int]$Warmups = 1,
  [string]$Label = "qwen36"
)
$ErrorActionPreference = "Stop"
$out = Join-Path (Split-Path $PSScriptRoot -Parent) "results\$Label"
New-Item -ItemType Directory -Force $out | Out-Null
if ($Dataset -eq "random") {
  $data = @("--backend", "openai", "--endpoint", "/v1/completions", "--dataset-name", "random",
            "--random-input-len", $InputLen, "--random-output-len", $OutputLen)
} else {
  $data = @("--backend", "openai-chat", "--endpoint", "/v1/chat/completions", "--dataset-name", "sharegpt",
            "--dataset-path", "/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", "--sharegpt-output-len", $OutputLen)
}

$rows = @()
foreach ($c in ($Conc -split "," | ForEach-Object { [int]$_.Trim() })) {
  $n = [Math]::Max(3, $PromptsPerConc * $c)
  Write-Output "== $Dataset, concurrency $c, $n prompts, $OutputLen out"
  docker exec $Name vllm bench serve --base-url http://127.0.0.1:8000 --model $Model @data --ignore-eos `
    --temperature 0 --seed 0 --num-prompts $n --max-concurrency $c --num-warmups $Warmups `
    --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /tmp/bench --result-filename "$Dataset-c$c.json"
  if ($LASTEXITCODE -ne 0) { throw "vllm bench serve failed at concurrency $c" }
  docker cp "${Name}:/tmp/bench/$Dataset-c$c.json" (Join-Path $out "$Dataset-c$c.json") | Out-Null
  $r = Get-Content (Join-Path $out "$Dataset-c$c.json") -Raw | ConvertFrom-Json
  $rows += [pscustomobject]@{
    dataset         = $Dataset
    conc            = $c
    'output tok/s'  = [Math]::Round($r.output_throughput, 2)
    'tok/s per req' = [Math]::Round(1000 / $r.mean_tpot_ms, 2)
    'TPOT ms'       = [Math]::Round($r.mean_tpot_ms, 1)
    'TTFT ms'       = [Math]::Round($r.mean_ttft_ms, 0)
  }
}
$rows | Format-Table -AutoSize | Out-String | Tee-Object (Join-Path $out "summary-$Dataset.txt")
