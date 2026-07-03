#!/usr/bin/env bash
# Builds the shared HPC base image and runs the Docker-based HPC rig tests.
#
# Usage:
#   ./scripts/hpc_validate.sh          # tear the rig(s) down afterwards
#   ./scripts/hpc_validate.sh --keep   # leave them running (DEAP_HPC_KEEP=1)
set -euo pipefail

KEEP=0
if [[ "${1:-}" == "--keep" ]]; then
    KEEP=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building deap-hpc-base:latest..."
docker build -t deap-hpc-base:latest -f "$REPO_ROOT/docker/hpc-base/Dockerfile" "$REPO_ROOT"

export DEAP_HPC_TESTS=1
if [[ "$KEEP" == "1" ]]; then
    export DEAP_HPC_KEEP=1
fi

python -m pytest -q --no-cov "$REPO_ROOT/tests/hpc"
