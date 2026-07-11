[CmdletBinding()]
param(
    [string]$AdapterDir = "",
    [string]$BaseConfigDir = "",
    [string]$OllamaBaseModel = "qwen3.5:9b",
    [string]$OutputModel = "myclone-vl:latest",
    [int]$LoraAlpha = 0,
    [switch]$SkipConvert
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if (-not $AdapterDir) {
    $AdapterDir = Join-Path $repoRoot "kaggle_output-latest"
}
$AdapterDir = (Resolve-Path $AdapterDir).Path

if (-not (Test-Path (Join-Path $AdapterDir "adapter_config.json"))) {
    throw "adapter_config.json not found in $AdapterDir"
}
if (-not (Test-Path (Join-Path $AdapterDir "adapter_model.safetensors"))) {
    throw "adapter_model.safetensors not found in $AdapterDir"
}
$sourceAdapterConfig = Get-Content (Join-Path $AdapterDir "adapter_config.json") -Raw | ConvertFrom-Json
$originalLoraAlpha = [int]$sourceAdapterConfig.lora_alpha
$loraRank = [int]$sourceAdapterConfig.r
$recommendedAlpha = [Math]::Max(1, [Math]::Floor($loraRank / 2))
$effectiveLoraAlpha = if ($LoraAlpha -gt 0) {
    $LoraAlpha
} else {
    [Math]::Min($originalLoraAlpha, $recommendedAlpha)
}

if (-not $BaseConfigDir) {
    $snapshotRoot = Join-Path $PSScriptRoot "base_model_cache\models\Qwen--Qwen3.5-9B\snapshots"
    $snapshot = Get-ChildItem $snapshotRoot -Directory | Select-Object -First 1
    if (-not $snapshot) {
        throw "Qwen3.5 base snapshot was not found under $snapshotRoot"
    }
    $BaseConfigDir = $snapshot.FullName
}
$BaseConfigDir = (Resolve-Path $BaseConfigDir).Path

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$converter = Join-Path $PSScriptRoot "llama.cpp-src\convert_lora_to_gguf.py"
$adapterGguf = Join-Path $PSScriptRoot "myclone-lora-f16.gguf"
$modelfile = Join-Path $PSScriptRoot "Modelfile"

if (-not (Test-Path $python)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path $converter)) {
    throw "LoRA converter not found: $converter"
}

if (-not $SkipConvert) {
    $convertAdapterDir = $AdapterDir
    $temporaryAdapterDir = $null
    try {
        if ($effectiveLoraAlpha -ne $originalLoraAlpha) {
            # Hard links must stay on the same Windows volume as the adapter.
            $temporaryAdapterDir = Join-Path $PSScriptRoot ".adapter-build-$PID"
            New-Item -ItemType Directory -Path $temporaryAdapterDir -Force | Out-Null
            $adapterConfig = Get-Content (Join-Path $AdapterDir "adapter_config.json") -Raw | ConvertFrom-Json
            $adapterConfig.lora_alpha = $effectiveLoraAlpha
            $configJson = $adapterConfig | ConvertTo-Json -Depth 20
            [IO.File]::WriteAllText(
                (Join-Path $temporaryAdapterDir "adapter_config.json"),
                $configJson,
                [Text.UTF8Encoding]::new($false)
            )
            New-Item -ItemType HardLink `
                -Path (Join-Path $temporaryAdapterDir "adapter_model.safetensors") `
                -Target (Join-Path $AdapterDir "adapter_model.safetensors") | Out-Null
            $convertAdapterDir = $temporaryAdapterDir
            Write-Host "Using conservative LoRA alpha $effectiveLoraAlpha " `
                "(adapter stored $originalLoraAlpha, rank $loraRank)."
        }

        & $python $converter $convertAdapterDir --base $BaseConfigDir --outfile $adapterGguf --outtype f16
        if ($LASTEXITCODE -ne 0) {
            throw "LoRA to GGUF conversion failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        if ($temporaryAdapterDir) {
            $resolvedTemp = [IO.Path]::GetFullPath($temporaryAdapterDir)
            $safeRoot = [IO.Path]::GetFullPath($PSScriptRoot)
            if ($resolvedTemp.StartsWith($safeRoot, [StringComparison]::OrdinalIgnoreCase)) {
                Remove-Item -LiteralPath $resolvedTemp -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
if (-not (Test-Path $adapterGguf)) {
    throw "GGUF adapter not found: $adapterGguf"
}

# Keep the original Ollama VLM as FROM and attach only the text LoRA. Merging
# the entire Hugging Face VLM into one language GGUF drops the visual tower.
$adapterPath = $adapterGguf.Replace("\", "/")
$content = @"
FROM $OllamaBaseModel
ADAPTER $adapterPath

PARAMETER num_ctx 65536
PARAMETER temperature 0.4
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05

SYSTEM "请你扮演一名人类，不要说自己是人工智能。根据对话上下文准确理解对方的意思，再用自然、简短的聊天语气回复。"
"@
[IO.File]::WriteAllText($modelfile, $content, [Text.UTF8Encoding]::new($false))

& ollama create $OutputModel -f $modelfile
if ($LASTEXITCODE -ne 0) {
    throw "ollama create failed with exit code $LASTEXITCODE"
}

$show = (& ollama show $OutputModel | Out-String)
if ($show -notmatch "vision") {
    throw "Created model does not advertise vision capability. Refusing text-only deployment."
}
Write-Host "Created $OutputModel with the original vision tower and the fine-tuned text LoRA."
Write-Host $show
