"""Standalone SCOOP launcher script for the custom-env propagation test.

Run via ``python -m scoop -n 2 tests/_scoop_custom_env_script.py`` from the
repo root. A genuine SCOOP run (unlike an in-process ``scoop.futures.map``)
spawns separate worker processes, so a custom env that is registered only at
runtime in this main process is **not** present in the workers' registries —
unless the registry snapshot shipped on each ``_EvalWorkerSpec`` is replayed
there. This script exercises exactly that path and exits non-zero on failure
so the calling test can assert success via the return code.
"""

from __future__ import annotations

import sys

import numpy as np

from deap import creator

from deap_pettingzoo.environments.registry import get_registry_snapshot
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.integration.deap_integration import (
    _EvalWorkerSpec,
    evaluate_population_parallel,
    setup_deap_creators,
)
from deap_pettingzoo.utils.parallel import make_executor

from tests.multidiscrete_env import (
    MD_ENV_NAME,
    MD_NVEC,
    MD_OBS_DIM,
    register_md_env,
)


def main() -> int:
    # Register the custom env in the main process only; workers must obtain it
    # from the snapshot shipped on the worker spec.
    register_md_env()
    snapshot = get_registry_snapshot()

    setup_deap_creators()
    out_size = int(sum(MD_NVEC))
    inds = []
    for k in range(4):
        genome = FixedTopologyGenome(
            layer_sizes=[MD_OBS_DIM, 8, out_size],
            activation="tanh",
            weight_range=(-1.0, 1.0),
            random_seed=k,
        )
        ind = creator.Individual()
        ind.genome = genome
        inds.append(ind)

    spec = _EvalWorkerSpec(
        env_name=MD_ENV_NAME,
        env_config=None,
        max_steps=10,
        aggregation_method="sum",
        team_mapping=None,
        registry_snapshot=snapshot,
    )

    with make_executor(2, backend="scoop") as executor:
        results = evaluate_population_parallel(inds, executor=executor, spec=spec)

    ok = len(results) == 4 and all(np.isfinite(r[0]) for r in results)
    print("SCOOP_CUSTOM_ENV_OK" if ok else "SCOOP_CUSTOM_ENV_FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
