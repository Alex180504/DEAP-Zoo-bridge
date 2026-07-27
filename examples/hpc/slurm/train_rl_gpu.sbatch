#!/bin/bash
# Single-node GPU training for the AgileRL baselines (MADDPG smoke run):
# no Dask cluster — the RL learners are single-process torch training that
# scale up on one GPU, not out across nodes.
#
# This is the exact template exercised by tests/hpc/test_slurm_rig.py
# against the Docker Slurm simulation rig's GPU node (compose profile
# "gpu") — real-cluster deltas should only be the partition/gres names,
# the commented-out account/module/venv lines, and the training budget.
#
#SBATCH --job-name=deap-hpc-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# #SBATCH --account=<account>

set -euo pipefail

# module load python/3.11 cuda/12.4
# source /path/to/venv/bin/activate

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SHARED_DIR="${DEAP_HPC_SHARED_DIR:-$REPO_ROOT}"
RUN_DIR="$SHARED_DIR/runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# --require-cuda: inside a GPU allocation, a silent CPU fallback (the
# library's default behaviour when CUDA is missing) would waste the
# reservation — fail loudly instead.
python3 "$REPO_ROOT/examples/hpc/train_maddpg_gpu.py" \
    --require-cuda \
    --output-dir "$RUN_DIR" \
    "$@"
