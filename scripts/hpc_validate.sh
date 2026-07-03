#!/usr/bin/env bash
# Builds the shared HPC base image and runs the Docker-based HPC rig tests.
#
# Usage:
#   ./scripts/hpc_validate.sh          # tear the rig(s) down afterwards
#   ./scripts/hpc_validate.sh --keep   # leave them running (DEAP_HPC_KEEP=1)
#   ./scripts/hpc_validate.sh --gpu    # also run the Slurm rig's GPU node +
#                                      # AgileRL GPU test (DEAP_HPC_GPU=1;
#                                      # needs an NVIDIA GPU in Docker)
set -euo pipefail

KEEP=0
GPU=0
for arg in "$@"; do
    case "$arg" in
        --keep) KEEP=1 ;;
        --gpu) GPU=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Building deap-hpc-base:latest..."
docker build -t deap-hpc-base:latest -f "$REPO_ROOT/docker/hpc-base/Dockerfile" "$REPO_ROOT"

export DEAP_HPC_TESTS=1
if [[ "$KEEP" == "1" ]]; then
    export DEAP_HPC_KEEP=1
fi
if [[ "$GPU" == "1" ]]; then
    export DEAP_HPC_GPU=1
fi

python -m pytest -q --no-cov "$REPO_ROOT/tests/hpc"
