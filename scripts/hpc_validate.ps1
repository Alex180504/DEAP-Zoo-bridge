<#
.SYNOPSIS
    Builds the shared HPC base image and runs the Docker-based HPC rig tests.

.DESCRIPTION
    Rebuilds deap-hpc-base:latest from docker/hpc-base/Dockerfile, then runs
    every test under tests/hpc

.PARAMETER Keep
    Leave the rig(s) running after the tests finish (sets DEAP_HPC_KEEP=1),
    e.g. to poke at the Dask dashboard on http://localhost:8787.

.EXAMPLE
    .\scripts\hpc_validate.ps1
    .\scripts\hpc_validate.ps1 -Keep
#>
param(
    [switch]$Keep
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Building deap-hpc-base:latest..."
docker build -t deap-hpc-base:latest -f "$RepoRoot/docker/hpc-base/Dockerfile" $RepoRoot
if ($LASTEXITCODE -ne 0) { throw "Base image build failed." }

$env:DEAP_HPC_TESTS = "1"
if ($Keep) {
    $env:DEAP_HPC_KEEP = "1"
}

try {
    python -m pytest -q --no-cov "$RepoRoot/tests/hpc"
    $testExit = $LASTEXITCODE
}
finally {
    Remove-Item Env:\DEAP_HPC_TESTS -ErrorAction SilentlyContinue
    Remove-Item Env:\DEAP_HPC_KEEP -ErrorAction SilentlyContinue
}

exit $testExit
