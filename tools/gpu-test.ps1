#Requires -Version 5.1
<#
.SYNOPSIS
    Reproducible runner for NSL's GPU (CUDA) test suites — the documented
    manual gate that CI cannot provide (CI has no GPU).

.DESCRIPTION
    The `--features cuda` integration tests are `#[ignore]`d so a default
    `cargo test` skips them. This script is the *documented manual loop* that
    Phase 0.1 of the "make pretraining actually work" plan calls for:

      1. Preflights the CUDA toolchain (GPU present, CUDA_PATH + cuda.lib,
         NSL_SKIP_CUDA_TESTS not set) and fails loud with remediation.
      2. Stamps machine/toolchain provenance (GPU, driver, nvcc, rustc, git)
         so a green run is auditable and reproducible.
      3. Runs the ignored CUDA tests on real silicon with a clear PASS/FAIL
         summary and a non-zero exit on any failure.

    A key guardrail: a CUDA test that early-returns because no GPU is present
    still counts as "passed" to libtest. The preflight refuses to run without
    a healthy GPU, and (for -Canary / exact -Filter) the summary treats a run
    that executed 0 tests as a FAILURE — so a skip can never masquerade as a
    green gate.

.PARAMETER Canary
    Run only the curated known-green smoke tests in tools/gpu-canary.txt.
    This is the acceptance gate: if the canary is green, the harness works.

.PARAMETER Filter
    Run a specific test by exact fn name (substring if -Loose) in -Package.

.PARAMETER All
    Run every ignored CUDA test in the given -Package(s).

.PARAMETER Package
    Cargo package(s) to target. Default: nsl-runtime. The other GPU-heavy
    crate is nsl-codegen (pass -Package nsl-codegen).

.PARAMETER TestBinary
    Restrict to a single integration-test binary (cargo --test <name>).

.PARAMETER Loose
    With -Filter, treat the value as a substring instead of an exact fn name.

.PARAMETER TestThreads
    libtest --test-threads. Default 1 (GPU tests must serialize the device).

.PARAMETER Release
    Build/run in release mode.

.PARAMETER ListCanary
    Print the canary manifest and exit (no build, no preflight).

.PARAMETER NoPreflight
    Skip the preflight checks (use only when you have already verified the
    toolchain this session).

.EXAMPLE
    pwsh tools/gpu-test.ps1 -Canary

.EXAMPLE
    pwsh tools/gpu-test.ps1 -Filter x_prepass_matches_cpu_reference -TestBinary tier_b1_prepass_gpu

.EXAMPLE
    pwsh tools/gpu-test.ps1 -All -Package nsl-codegen -TestBinary csha_cuda_backward
#>
[CmdletBinding()]
param(
    [switch]   $Canary,
    [string]   $Filter,
    [switch]   $All,
    [string[]] $Package = @('nsl-runtime'),
    [string]   $TestBinary = '',
    [switch]   $Loose,
    [int]      $TestThreads = 1,
    [switch]   $Release,
    [switch]   $ListCanary,
    [switch]   $NoPreflight
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir
$manifest  = Join-Path $scriptDir 'gpu-canary.txt'
$isWin     = ($env:OS -eq 'Windows_NT')

function Write-Section($title) {
    Write-Host ''
    Write-Host ('=' * 72) -ForegroundColor DarkCyan
    Write-Host $title       -ForegroundColor Cyan
    Write-Host ('=' * 72) -ForegroundColor DarkCyan
}

function Stop-Preflight($msg, $remedy) {
    Write-Host "PREFLIGHT FAILED: $msg" -ForegroundColor Red
    if ($remedy) { Write-Host "  -> $remedy" -ForegroundColor Yellow }
    exit 2
}

function Invoke-Preflight {
    Write-Section 'GPU test harness - preflight & provenance'

    # NSL_SKIP_CUDA_TESTS makes every CUDA test early-return as a "pass".
    # That silently defeats the gate, so refuse rather than false-green.
    if ($env:NSL_SKIP_CUDA_TESTS) {
        Stop-Preflight "NSL_SKIP_CUDA_TESTS is set ('$($env:NSL_SKIP_CUDA_TESTS)')" `
            'Unset it ($env:NSL_SKIP_CUDA_TESTS=$null); it makes CUDA tests skip and falsely report green.'
    }

    # GPU present?
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Stop-Preflight 'nvidia-smi not found on PATH' 'Install the NVIDIA driver and ensure nvidia-smi is on PATH.'
    }
    $gpuInfo = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $gpuInfo) {
        Stop-Preflight 'nvidia-smi ran but reported no usable GPU' 'Check the GPU is present and the driver is healthy.'
    }

    # CUDA toolkit link requirement (cudarc dynamic-linking needs cuda.lib).
    if (-not $env:CUDA_PATH) {
        Stop-Preflight 'CUDA_PATH is not set' 'Install the CUDA Toolkit and set CUDA_PATH (e.g. C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.Y).'
    }
    if ($isWin) {
        $cudaLib = Join-Path $env:CUDA_PATH 'lib\x64\cuda.lib'
        if (-not (Test-Path $cudaLib)) {
            Stop-Preflight "cuda.lib not found at $cudaLib" 'The CUDA Toolkit lib\x64\cuda.lib is required to link cudarc.'
        }
    }

    # Provenance.
    $nvcc = '(nvcc not found)'
    if (Get-Command nvcc -ErrorAction SilentlyContinue) {
        $rel = (& nvcc --version 2>$null | Select-String 'release')
        if ($rel) { $nvcc = $rel.ToString().Trim() }
    }
    $rustc = (& rustc --version 2>$null)
    $sha = '(no git)'; $branch = ''
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $sha    = (& git -C $repoRoot rev-parse --short HEAD 2>$null)
        $branch = (& git -C $repoRoot rev-parse --abbrev-ref HEAD 2>$null)
    }
    Write-Host ("  timestamp : {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'))
    Write-Host ("  gpu       : {0}" -f ($gpuInfo -join '; '))
    Write-Host ("  cuda      : {0}" -f $env:CUDA_PATH)
    Write-Host ("  nvcc      : {0}" -f $nvcc)
    Write-Host ("  rustc     : {0}" -f $rustc)
    Write-Host ("  git       : {0} @ {1}" -f $branch, $sha)
    Write-Host ("  repo root : {0}" -f $repoRoot)
    Write-Host '  preflight : OK' -ForegroundColor Green
}

function Get-CanaryEntries {
    if (-not (Test-Path $manifest)) {
        Stop-Preflight "canary manifest not found: $manifest" 'Restore tools/gpu-canary.txt.'
    }
    $entries = @()
    foreach ($line in Get-Content $manifest) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith('#')) { continue }
        $parts = $t.Split('|')
        if ($parts.Count -lt 3) {
            Write-Host "  skipping malformed manifest line: $t" -ForegroundColor Yellow
            continue
        }
        $note = if ($parts.Count -ge 4) { $parts[3].Trim() } else { '' }
        $entries += [pscustomobject]@{
            Package = $parts[0].Trim()
            Binary  = $parts[1].Trim()
            Test    = $parts[2].Trim()
            Note    = $note
        }
    }
    return $entries
}

function Invoke-CargoTest($pkg, $bin, $extraArgs, $label, $strict) {
    $cargoArgs = @('test', '-p', $pkg, '--features', 'cuda')
    if ($Release) { $cargoArgs += '--release' }
    if ($bin)     { $cargoArgs += @('--test', $bin) }
    $cargoArgs += '--'
    $cargoArgs += '--ignored'
    if ($extraArgs) { $cargoArgs += $extraArgs }
    $cargoArgs += @('--nocapture', "--test-threads=$TestThreads")

    Write-Host ''
    Write-Host ("RUN [{0}]: cargo {1}" -f $label, ($cargoArgs -join ' ')) -ForegroundColor Magenta

    $logDir = Join-Path $repoRoot 'target\gpu-test-logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stamp     = Get-Date -Format 'yyyyMMdd-HHmmss'
    $safeLabel = ($label -replace '[^A-Za-z0-9._-]', '_')
    $logFile   = Join-Path $logDir "gpu-$stamp-$safeLabel.log"

    # Run under cmd redirection so PowerShell never wraps cargo's stderr as
    # terminating NativeCommandError records (a PS 5.1 trap with `2>&1` under
    # $ErrorActionPreference='Stop'). This keeps $LASTEXITCODE honest and
    # captures the full combined stream to the log for parsing + display.
    $cargoLine = 'cargo ' + ($cargoArgs -join ' ')
    Push-Location $repoRoot
    try {
        cmd /c "$cargoLine > `"$logFile`" 2>&1"
        $code = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    $text = if (Test-Path $logFile) { Get-Content $logFile -Raw } else { '' }
    if ($text) { Write-Host $text }

    # A run "passed" only if cargo exited 0. In strict mode we additionally
    # require that at least one test actually executed and passed — so a
    # 0-tests-ran (skipped/filtered) result cannot masquerade as green.
    $ranOk = ($code -eq 0)
    if ($strict) {
        # Sum "passed" across ALL per-binary "test result: ok" blocks, not just
        # the first — with -Filter and no -TestBinary, cargo runs every test
        # binary in the package and each prints its own summary. A filtered test
        # that passes in a later binary must still count, and a 0-tests-ran
        # result must never masquerade as green.
        $totalPassed = 0
        foreach ($mm in [regex]::Matches($text, 'test result:\s*ok\.\s*(\d+)\s+passed;')) {
            $totalPassed += [int]$mm.Groups[1].Value
        }
        $ranOk = ($code -eq 0 -and $totalPassed -ge 1)
        if ($code -eq 0 -and -not $ranOk) {
            Write-Host '  WARN: exit 0 but no test executed+passed (skipped/filtered?). Treating as FAIL.' -ForegroundColor Yellow
        }
    }

    return [pscustomobject]@{
        Label    = $label
        Package  = $pkg
        Binary   = $bin
        ExitCode = $code
        Passed   = $ranOk
        Log      = $logFile
    }
}

# --- Dispatch ---------------------------------------------------------------

if ($ListCanary) {
    Write-Section 'Canary manifest (tools/gpu-canary.txt)'
    Get-CanaryEntries | Format-Table Package, Binary, Test, Note -AutoSize
    exit 0
}

if (-not ($Canary -or $Filter -or $All)) {
    Write-Host 'NSL GPU test harness'
    Write-Host 'Usage: tools/gpu-test.ps1 [-Canary | -Filter <name> | -All] [-Package <p>] [-TestBinary <b>] [-ListCanary]'
    Write-Host ''
    Write-Host 'Examples:'
    Write-Host '  tools/gpu-test.ps1 -Canary'
    Write-Host '  tools/gpu-test.ps1 -ListCanary'
    Write-Host '  tools/gpu-test.ps1 -Filter x_prepass_matches_cpu_reference -TestBinary tier_b1_prepass_gpu'
    Write-Host '  tools/gpu-test.ps1 -All -Package nsl-codegen'
    exit 0
}

if (-not $NoPreflight) { Invoke-Preflight }

$results = @()
if ($Canary) {
    Write-Section 'Canary suite (curated known-green GPU tests)'
    foreach ($e in Get-CanaryEntries) {
        $results += Invoke-CargoTest $e.Package $e.Binary @('--exact', $e.Test) ("{0}::{1}" -f $e.Binary, $e.Test) $true
    }
}
elseif ($Filter) {
    Write-Section "Filter run: '$Filter'"
    if (-not $TestBinary) {
        Write-Host '  NOTE: no -TestBinary given; every ignored CUDA test binary in the package will be built and run (slow). Pass -TestBinary to scope it.' -ForegroundColor Yellow
    }
    $exactArgs = if ($Loose) { @($Filter) } else { @('--exact', $Filter) }
    foreach ($pkg in $Package) {
        $results += Invoke-CargoTest $pkg $TestBinary $exactArgs ("{0}/{1}" -f $pkg, $Filter) $true
    }
}
elseif ($All) {
    Write-Section 'All ignored CUDA tests'
    foreach ($pkg in $Package) {
        $results += Invoke-CargoTest $pkg $TestBinary @() ("{0}/all" -f $pkg) $false
    }
}

if ($results.Count -eq 0) {
    Write-Host ''
    Write-Host 'FAIL: no test runs were executed (empty canary/filter, or every manifest line was malformed).' -ForegroundColor Red
    Write-Host '      A gate that runs nothing is not a pass.' -ForegroundColor Yellow
    exit 1
}

Write-Section 'Summary'
$results | Format-Table Label, Package, ExitCode, Passed, Log -AutoSize
$failed = @($results | Where-Object { -not $_.Passed })
if ($failed.Count -gt 0) {
    Write-Host ("FAIL: {0}/{1} run(s) did not pass on GPU." -f $failed.Count, $results.Count) -ForegroundColor Red
    exit 1
}
Write-Host ("PASS: all {0} run(s) green on GPU." -f $results.Count) -ForegroundColor Green
exit 0
