param([string]$DataRoot = 'data', [int]$MaxToolCalls = 50, [int]$MaxTimeMinutes = 30)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Py = (Resolve-Path '.venv\Scripts\python.exe').Path

if (-not (Test-Path $Py)) { throw 'Virtual environment missing. Run .\setup_rtx5090.ps1 first.' }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw 'nvidia-smi not found.' }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if (Get-Command docker -ErrorAction SilentlyContinue) { docker info | Out-Null } else { throw 'Docker is required. Start Docker Desktop and retry.' }

& $Py scripts\smoke_test.py --variant agent_2_integrated
& $Py scripts\build_submission.py --variant agent_2_integrated
if (-not (Get-Command swegemma -ErrorAction SilentlyContinue)) { throw 'swegemma is not installed. Put harness wheels under data\wheels and rerun setup.' }

$Tasks = Join-Path $DataRoot 'tasks.jsonl'
$Snapshots = Join-Path $DataRoot 'snapshots'
$Reference = Join-Path $DataRoot 'sample_submission'
$Submission = 'build\agent_2_integrated'
$Results = 'results\denisha_rtx5090'
foreach ($Path in @($Tasks, $Snapshots, $Reference, $Submission)) { if (-not (Test-Path $Path)) { throw "Missing required path: $Path" } }

& $Py scripts\run_local_eval.py `
    --tasks $Tasks `
    --snapshots-dir $Snapshots `
    --submission-dir $Submission `
    --results-dir $Results `
    --sandbox docker `
    --max-tool-calls $MaxToolCalls `
    --max-time-minutes $MaxTimeMinutes `
    --concurrency 1
