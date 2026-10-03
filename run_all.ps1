# Full from-scratch reproduction of the paper's experiments: about 1 h 30 min on one RTX 4090 Laptop GPU,
# plus the UCI HAR download.
# Usage (Windows PowerShell or pwsh, from any directory):
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1 [-Python .venv\Scripts\python.exe] [-Workers 4]
# (-ExecutionPolicy Bypass applies to this call only; it is needed because the default policy blocks scripts.)
param([string]$Python = "python", [int]$Workers = 4)
$ErrorActionPreference = "Stop"
if ($Python -match '[\\/]') { $Python = (Resolve-Path -LiteralPath $Python).Path }  # absolute before Set-Location
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Step([string]$Title, [string[]]$Commands) {
    Write-Host "`n=== $Title  [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ==="
    foreach ($c in $Commands) {
        $argv = $c -split ' '
        & $Python @argv
        if ($LASTEXITCODE -ne 0) { throw "failed: $Python $c" }
    }
}

Invoke-Step "0/6 data: verify shipped PBMC file, download + verify UCI HAR" @("data/download.py --all")
Invoke-Step "1/6 unit tests" @("-m pytest -q tests")
Invoke-Step "2/6 exact Gaussian fixed-target study (~40 min)" @(
    "experiments/exact_gaussian/run.py main",
    "experiments/exact_gaussian/run.py ablation",
    "experiments/exact_gaussian/run.py tune",
    "experiments/exact_gaussian/run.py audit",
    "experiments/exact_gaussian/make_tables.py --check",
    "experiments/exact_gaussian/make_figure.py --check")
Invoke-Step "3/6 PBMC 3k matched protocol (~11 min)" @(
    "experiments/pbmc/run.py --units all",
    "experiments/pbmc/stability.py --strict")
Invoke-Step "4/6 UCI HAR matched protocol (~31 min) and five-seed evaluator (~5 min)" @(
    "experiments/uci_har/run.py --units all",
    "experiments/uci_har/evaluate.py --workers $Workers")
Invoke-Step "5/6 paired statistics (Table A3, 70 contrasts)" @(
    "experiments/paired/paired_stats.py",
    "experiments/paired/make_table.py")
Invoke-Step "6/6 bitwise verification against reference/ and the paper" @("verify.py --require-all")
Write-Host "`n=== done  [$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ==="
