$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw 'Python 3.10+ was not found on PATH.' }
if (-not (Test-Path '.venv\Scripts\python.exe')) { python -m venv .venv }
$Py = (Resolve-Path '.venv\Scripts\python.exe').Path
& $Py -m pip install --upgrade pip
& $Py -m pip install -r requirements.txt

$WheelDir = Join-Path $Root 'data\wheels'
if (Test-Path $WheelDir) {
    $Wheels = Get-ChildItem $WheelDir -Filter '*.whl' -File
    if ($Wheels.Count -gt 0) { & $Py -m pip install --upgrade ($Wheels | ForEach-Object { $_.FullName }) }
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { throw 'nvidia-smi was not found. Install the NVIDIA driver first.' }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
Write-Host 'Setup complete. Run .\run_rtx5090.ps1 next.'
