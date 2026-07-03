# Base image shared by every HPC validation rig (Dask now, Slurm later) so
# they run byte-identical Python/numpy/BLAS stacks -- required for the
# serial-vs-distributed reproducibility guarantee the rigs assert on.
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE.txt ./
COPY deap_pettingzoo ./deap_pettingzoo
COPY examples ./examples
# tests/hpc ships module-level custom-env factories the rig registers at
# runtime; they must be importable inside every container (driver AND
# workers) to prove genuine cross-container registry-snapshot replay, so
# they're baked in like the library itself.
COPY tests/hpc ./tests/hpc
RUN touch ./tests/__init__.py

# No torch: EA / CMA-ES / PSO evaluation is pure numpy. bokeh powers the
# Dask dashboard (--dashboard-address).
RUN pip install --no-cache-dir ".[dask,logging]" bokeh

ENV SDL_VIDEODRIVER=dummy \
    SDL_AUDIODRIVER=dummy \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

CMD ["python3"]
