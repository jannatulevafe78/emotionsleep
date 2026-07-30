# ============================================================
#  ECSMP pipeline -- Windows / PowerShell setup and run
#
#  Usage:
#     .\run.ps1 -Root "C:\path\to\ECSMP"
#     .\run.ps1 -Root "C:\path\to\ECSMP" -Stage inspect
#     .\run.ps1 -Root "C:\path\to\ECSMP" -Cpu
#
#  First run installs a virtual environment (~5 min). Later runs reuse it.
# ============================================================

param(
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$Stage = "all",
    [switch]$Cpu,
    [switch]$SkipInstall,
    [switch]$Baselines
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Say($m) { Write-Host "`n>>> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "!!! $m" -ForegroundColor Yellow }

# ---------- checks ----------
if (-not (Test-Path $Root)) { throw "Dataset folder not found: $Root" }

$py = $null
foreach ($c in @("python", "py", "python3")) {
    try { & $c --version *> $null; if ($LASTEXITCODE -eq 0) { $py = $c; break } } catch {}
}
if (-not $py) { throw "Python not found on PATH. Install Python 3.10+ and retry." }
Say "Using $py ($(& $py --version 2>&1))"

# ---------- venv ----------
if (-not $SkipInstall) {
    if (-not (Test-Path ".venv")) {
        Say "Creating virtual environment"
        & $py -m venv .venv
    }
    $pip = ".\.venv\Scripts\python.exe"
    Say "Installing dependencies (first run takes a few minutes)"
    & $pip -m pip install --upgrade pip --quiet

    if ($Cpu) {
        & $pip -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
    } else {
        Say "Installing CUDA build of torch (use -Cpu if you have no NVIDIA GPU)"
        & $pip -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet
    }
    & $pip -m pip install -r requirements.txt --quiet
    & $pip -m pip install openpyxl h5py --quiet   # .xlsx scales, MATLAB v7.3 files
    Say "Dependencies installed"
}

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = $py }

$devArg = if ($Cpu) { @("--device", "cpu") } else { @() }
$blArg  = if ($Baselines) { @("--baselines") } else { @() }

# ---------- run ----------
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null

if ($Stage -eq "all") {
    Say "STEP 1/5  inspect  (verify the pipeline can see your files)"
    & $python run_ecsmp.py inspect --root "$Root"

    Write-Host ""
    Warn "Check the counts above before continuing."
    Warn "Expect ~89 subjects, ~6 emotion files each, ~1 sleep file each."
    $ans = Read-Host "Do the counts look right? (y/n)"
    if ($ans -ne "y") {
        Warn "Stopping. Open outputs\ecsmp_inventory.json, look at 'unclassified',"
        Warn "and send it back so the filename patterns can be corrected."
        exit 1
    }

    Say "STEP 2/5  prep  (epochs + sleep HRV + questionnaires)"
    & $python run_ecsmp.py prep --root "$Root"

    Say "STEP 3/5  relate  (THE RESEARCH QUESTION)"
    & $python run_ecsmp.py relate

    Say "STEP 4/5  classify  (6-class emotion recognition)"
    & $python run_ecsmp.py classify @devArg @blArg

    Say "STEP 5/5  report"
    & $python run_ecsmp.py report
} else {
    Say "Running stage: $Stage"
    & $python run_ecsmp.py $Stage --root "$Root" @devArg @blArg
}

# ---------- summary ----------
Say "Finished"
if (Test-Path "outputs\REPORT.md") {
    Write-Host "Report : $(Resolve-Path 'outputs\REPORT.md')" -ForegroundColor Green
    Write-Host "Figures: $(Resolve-Path 'outputs')" -ForegroundColor Green
    Write-Host ""
    Write-Host "--- REPORT PREVIEW ---" -ForegroundColor Gray
    Get-Content "outputs\REPORT.md" -TotalCount 45
} else {
    Warn "No report produced. Check the errors above."
}
Write-Host "`nSend back: outputs\REPORT.md, outputs\ecsmp_summary.json, outputs\ecsmp_inventory.json" -ForegroundColor Cyan
