#!/bin/bash
# Bidirectional EA-vs-MADDPG co-evolution across the whole cluster: the
# driver (and the MADDPG evader's CUDA training) on one GPU node, the EA
# side's generation evaluation fanned out to Dask workers on CPU nodes.
# The live RL opponent reaches the workers as a per-generation frozen CPU
# actor snapshot, so the CPU nodes only need CPU torch + AgileRL installed
# — RL learning never leaves the GPU node.
#
# The split allocation is a Slurm *heterogeneous job*: component 0 is the
# GPU node (batch script, driver, dask scheduler), component 1 the CPU
# worker nodes. This is the exact template exercised by
# tests/hpc/test_slurm_rig.py against the Docker Slurm simulation rig's
# gpu profile — real-cluster deltas should only be the partition/gres
# names, the commented-out account/module/venv lines, node counts, and the
# training budget.
#
#SBATCH --job-name=deap-hpc-coev
#SBATCH --output=%x-%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# #SBATCH --account=<account>
#SBATCH hetjob
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
# #SBATCH --account=<account>

set -euo pipefail

# module load python/3.11 cuda/12.4
# source /path/to/venv/bin/activate

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
# Must be a filesystem shared by every allocated node (the scheduler-file
# handshake and the driver's results both live here).
SHARED_DIR="${DEAP_HPC_SHARED_DIR:-$REPO_ROOT}"
RUN_DIR="$SHARED_DIR/runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

SCHED_FILE="$RUN_DIR/scheduler.json"
rm -f "$SCHED_FILE"

SCHEDULER_PID=""
WORKER_PID=""
cleanup() {
    [[ -n "$SCHEDULER_PID" ]] && kill "$SCHEDULER_PID" 2>/dev/null || true
    [[ -n "$WORKER_PID" ]] && kill "$WORKER_PID" 2>/dev/null || true
    rm -f "$SCHED_FILE"
}
trap cleanup EXIT TERM INT

# --- Scheduler -----------------------------------------------------------
# Plain background process on the GPU node (the batch script runs on het
# component 0's node); it dies with the batch step, no srun needed.
dask scheduler --scheduler-file "$SCHED_FILE" --no-dashboard &
SCHEDULER_PID=$!

echo "Waiting for scheduler file: $SCHED_FILE"
for _ in $(seq 1 60); do
    [[ -s "$SCHED_FILE" ]] && break
    sleep 1
done
if [[ ! -s "$SCHED_FILE" ]]; then
    echo "ERROR: scheduler did not write $SCHED_FILE within 60s" >&2
    exit 1
fi

# --- Workers ---------------------------------------------------------------
# One worker task per CPU node (het component 1), each spawning
# --cpus-per-task single-threaded worker processes: the worker-side
# evaluation caches are not thread-safe.
CPU_NODES="${SLURM_JOB_NUM_NODES_HET_GROUP_1:-2}"
WORKERS_PER_NODE="${DEAP_HPC_WORKERS_PER_NODE:-2}"
srun --het-group=1 --ntasks="$CPU_NODES" --overlap \
    dask worker --scheduler-file "$SCHED_FILE" \
    --nthreads 1 --nworkers "$WORKERS_PER_NODE" --memory-limit 0 \
    --no-dashboard &
WORKER_PID=$!

# --- Driver ------------------------------------------------------------
# Foreground on the GPU node. --require-cuda: inside a GPU allocation a
# silent CPU fallback would waste the reservation — fail loudly instead.
export DEAP_DASK_SCHEDULER="$SCHED_FILE"
python3 "$REPO_ROOT/examples/hpc/train_coevolution_ea_vs_rl.py" \
    --require-cuda \
    --backend dask \
    --num-workers "$((CPU_NODES * WORKERS_PER_NODE))" \
    --output-dir "$RUN_DIR" \
    "$@"
