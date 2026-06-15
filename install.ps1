# install.ps1 — one-command installer for DEAP-bridge + optional extras.
#
# Usage:
#   .\install.ps1                  core only (no RL, no UI)
#   .\install.ps1 service          backend (FastAPI + RL trainers)
#   .\install.ps1 ui               frontend (Streamlit, no ML deps)
#   .\install.ps1 ui-full          everything (backend + frontend)
#   .\install.ps1 dev              development tools
#   .\install.ps1 all              every optional extra
#

param([string]$Extra = "")

$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

# -- AgileRL pre-install (skip if already present) --------------------------
$agilerlInstalled = & $Python -c "import agilerl" 2>$null
if (-not $?) {
    Write-Host "[deap-bridge] Installing AgileRL (without its pettingzoo pin) ..."
    & $Python -m pip install --no-deps "agilerl>=2.6,<3"
    if ($LASTEXITCODE -ne 0) { throw "AgileRL install failed" }
}

# -- Main install ----------------------------------------------------------
$extraArg = if ($Extra) { ".[$Extra]" } else { "." }
Write-Host "[deap-bridge] Installing deap-pettingzoo$extraArg ..."
& $Python -m pip install -e $extraArg
if ($LASTEXITCODE -ne 0) { throw "deap-pettingzoo install failed" }

Write-Host "[deap-bridge] Done."
