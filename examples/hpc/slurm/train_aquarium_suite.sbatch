#!/bin/bash
# Aquarium pursuit-evasion suite on a single GPU node.
#
# One node fits every experiment family: MATD3 (and the coevolution RL
# sides) train on the GPU while EA / CMA-ES / PSO / MAP-Elites / HyperNEAT
# candidate evaluation fans out over the node's CPU cores via process
# workers (--cpus-per-task below is the evaluation parallelism).
#
#SBATCH --job-name=aquarium-suite
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# #SBATCH --account=<account>

set -euo pipefail

# One BLAS/OpenMP thread per process: evaluation workers parallelize at
# the process level, and os.cpu_count() sees the whole node while the
# cgroup only grants --cpus-per-task (native thread pools sized to the
# node oversubscribe the allocation and degrades the performance).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# module load python/3.11 cuda/12.4
# source /path/to/venv/bin/activate

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
SHARED_DIR="${DEAP_HPC_SHARED_DIR:-$REPO_ROOT}"
RUN_DIR="$SHARED_DIR/runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

python3 "$REPO_ROOT/examples/hpc/train_aquarium_suite.py" \
    --require-cuda \
    --num-workers "${SLURM_CPUS_PER_TASK:-16}" \
    --output-dir "$RUN_DIR" \
    "$@"

# ---------------------------------------------------------------------------
# Submission recipes.  Campaign algorithms: ea, cmaes, pso, map_elites,
# hyperneat, matd3.  "Proper budget" defaults ship in experiment_config.py:
# 2.5M env steps per vs-APF run, 5M per coevolution pairing, 1M per sweep
# config; add --budget to override.  Submit one job per (algorithm/pairing,
# seed, mode) cell so jobs stay short and pack well — see
# docs/enucc_campaign.md for per-cell resource requests.
#
# 1. vs-APF comparisons — both roles, both action modes, 10 seeds,
#    one cell per job:
#
#      for EXP in pursuer evader; do
#        for MODE in discrete continuous; do
#          for ALGO in ea cmaes pso map_elites hyperneat matd3; do
#            for SEED in $(seq 0 9); do
#              sbatch examples/hpc/slurm/train_aquarium_suite.sbatch \
#                --experiment "$EXP" --algos "$ALGO" \
#                --action-mode "$MODE" --seed "$SEED" --seeds 1
#            done
#          done
#        done
#      done
#
# 2. Coevolution suite — full round-robin: every ordered pair of distinct
#    algorithms (each algorithm meets every other as both predators and
#    prey; 30 pairings), both action modes, 5 seeds, one (pairing, seed)
#    per job:
#
#      ALGOS=(ea cma pso qdax hyperneat matd3)
#      for MODE in discrete continuous; do
#        for PRED in "${ALGOS[@]}"; do
#          for PREY in "${ALGOS[@]}"; do
#            [ "$PRED" = "$PREY" ] && continue
#            for SEED in $(seq 0 4); do
#              sbatch examples/hpc/slurm/train_aquarium_suite.sbatch \
#                --experiment coevolution --pairing "${PRED}_vs_${PREY}" \
#                --action-mode "$MODE" --seed "$SEED" --seeds 1
#            done
#          done
#        done
#      done
#
# 3. Hyperparameter sweeps (validate the shipped per-mode values), one
#    algorithm grid per job:
#
#      for ROLE in pursuer evader; do
#        for MODE in discrete continuous; do
#          for ALGO in ea cmaes pso map_elites hyperneat matd3; do
#            sbatch examples/hpc/slurm/train_aquarium_suite.sbatch \
#              --experiment sweep --role "$ROLE" --action-mode "$MODE" \
#              --algos "$ALGO"
#          done
#        done
#      done
# ---------------------------------------------------------------------------
