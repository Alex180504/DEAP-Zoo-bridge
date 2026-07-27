#!/bin/bash
# Multi-node Dask training on Slurm: one allocation, a scheduler + workers
# launched with srun --overlap inside it, the driver running in the
# foreground on the batch step.
#
# This is the exact template exercised by tests/hpc/test_slurm_rig.py
# against the Docker Slurm simulation rig -- real-cluster deltas should only
# be the commented-out lines (partition/account/module/venv) and node count.
#
#SBATCH --job-name=deap-hpc-dask
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>

set -euo pipefail

# --- Environment -------------------------------------------------------
# Adapt to your site: load a module and/or activate a venv with
# deap-pettingzoo[dask] installed *identically* on every node (mismatched
# versions across driver/workers are only warned about, not blocked --
# see docs/hpc_guide.md prerequisites).
# module load python/3.11
# source /path/to/venv/bin/activate

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
# Must be a filesystem shared by every allocated node (the scheduler-file
# handshake and the driver's results both live here).
SHARED_DIR="${DEAP_HPC_SHARED_DIR:-$REPO_ROOT}"
RUN_DIR="$SHARED_DIR/runs/${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# Scheduler-file pattern: the canonical dask-on-Slurm handshake over a
# shared FS -- workers/driver discover the scheduler without needing to
# know its hostname/port ahead of time.
SCHED_FILE="$RUN_DIR/scheduler.json"
rm -f "$SCHED_FILE"

SCHEDULER_PID=""
WORKER_PID=""
cleanup() {
    # Runs on normal exit AND on scancel (SIGTERM). The srun --overlap
    # steps below die with the job step regardless, but this avoids
    # orphaned background processes if the driver is killed mid-run.
    [[ -n "$SCHEDULER_PID" ]] && kill "$SCHEDULER_PID" 2>/dev/null || true
    [[ -n "$WORKER_PID" ]] && kill "$WORKER_PID" 2>/dev/null || true
    rm -f "$SCHED_FILE"
}
trap cleanup EXIT TERM INT

# --- Scheduler -----------------------------------------------------------
srun --nodes=1 --ntasks=1 --overlap \
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
# One worker task per node, each spawning $SLURM_CPUS_PER_TASK worker
# processes with one thread apiece: the project's worker-side evaluation
# caches are not thread-safe (one task at a time per process, like
# ProcessPoolExecutor).
srun --ntasks="$SLURM_JOB_NUM_NODES" --overlap \
    dask worker --scheduler-file "$SCHED_FILE" \
    --nthreads 1 --nworkers "$SLURM_CPUS_PER_TASK" --memory-limit 0 \
    --no-dashboard &
WORKER_PID=$!

# --- Driver ------------------------------------------------------------
# Foreground on the batch step; writes results under $RUN_DIR on the
# shared FS (matches ExperimentLogger's single-writer design -- only the
# driver ever writes, workers only evaluate).
export DEAP_DASK_SCHEDULER="$SCHED_FILE"
python3 "$REPO_ROOT/examples/hpc/train_ea_dask.py" \
    --backend dask \
    --num-workers "$((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))" \
    --output-dir "$RUN_DIR" \
    "$@"
