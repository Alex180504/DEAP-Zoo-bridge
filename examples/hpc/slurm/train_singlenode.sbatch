#!/bin/bash
# Single-node baseline: no Dask cluster to stand up, just the "process"
# backend confined to this node's allocated CPUs. Useful to sanity-check
# the driver script and environment (module loads, venv, library install)
# before running the multi-node train_dask.sbatch variant.
#
#SBATCH --job-name=deap-hpc-singlenode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>

set -euo pipefail

# module load python/3.11
# source /path/to/venv/bin/activate

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SHARED_DIR="${DEAP_HPC_SHARED_DIR:-$REPO_ROOT}"
RUN_DIR="$SHARED_DIR/runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

python3 "$REPO_ROOT/examples/hpc/train_ea_dask.py" \
    --backend process \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --output-dir "$RUN_DIR" \
    "$@"
