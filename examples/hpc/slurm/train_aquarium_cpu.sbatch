#!/bin/bash
# Aquarium pursuit-evasion suite: CPU cells (EA / CMA-ES / PSO / MAP-Elites /
# HyperNEAT vs-APF and sweeps, plus every coevolution pairing).
#
#
#SBATCH --job-name=aquarium-suite
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# One BLAS/OpenMP thread per process: evaluation workers parallelize at
# the process level, and os.cpu_count() sees the whole node while the
# cgroup only grants --cpus-per-task (native thread pools sized to the
# node oversubscribe the allocation and degrades the performance).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

module load apps/anaconda3/2024.10/bin
source activate ~/envs/deap

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SHARED_DIR="${DEAP_HPC_SHARED_DIR:-$REPO_ROOT}"
RUN_DIR="$SHARED_DIR/runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

python3 "$REPO_ROOT/examples/hpc/train_aquarium_suite.py" \
    --device cpu \
    --num-workers "${SLURM_CPUS_PER_TASK:-32}" \
    --output-dir "$RUN_DIR" \
    "$@"
