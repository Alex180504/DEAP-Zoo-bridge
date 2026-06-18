"""Convert between ExperimentSpec (JSON) and Setup-page form state.

The Setup page stores its form state in ``UIState.setup_config`` as a JSON
blob with keys like ``env``, ``algos``, ``env_values``, ``algo_values``,
``budget``, etc.  The canonical :class:`ExperimentSpec` has a different
structure (``environment``, ``algorithms``, ``hyperparameters``, etc.).

This module bridges the two so the re-run button on the Browse page can
pre-populate the Setup form.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def spec_to_form_defaults(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a canonical ExperimentSpec dict to Setup-page form state.

    The returned dict is suitable for merging into ``UIState.setup_config``::

        state.setup_config = json.dumps({
            **spec_to_form_defaults(spec),
            "_loaded_spec": spec,  # mark as loaded from re-run
        })

    Args:
        spec: An ExperimentSpec dict with keys like ``environment``,
            ``algorithms``, ``hyperparameters``, ``budget``, etc.

    Returns:
        A dict compatible with the Setup page's internal form state.
    """
    # Map top-level keys.
    form: Dict[str, Any] = {
        "env": spec.get("environment", ""),
        "algos": spec.get("algorithms", []),
    }

    # env_config → env_values
    env_config = spec.get("env_config") or {}
    if isinstance(env_config, dict):
        form["env_values"] = dict(env_config)

    # hyperparameters → algo_values
    hyperparams = spec.get("hyperparameters") or {}
    if isinstance(hyperparams, dict):
        form["algo_values"] = {
            algo: dict(params) for algo, params in hyperparams.items()
        }

    # budget
    budget = spec.get("budget") or {}
    if isinstance(budget, dict):
        form["budget"] = {
            "budget_mode": budget.get("mode", "env_steps"),
            "budget_value": budget.get("value", 50000),
            "num_seeds": budget.get("num_seeds", 1),
        }

    # fitness
    fitness = spec.get("fitness") or {}
    if isinstance(fitness, dict):
        form["fitness"] = {
            "aggregation_method": fitness.get("aggregation_method", "sum"),
            "team_mapping": fitness.get("team_mapping"),
        }

    # recording
    recording = spec.get("recording") or {}
    if isinstance(recording, dict):
        form["recording"] = {
            "enabled": recording.get("enabled", False),
            "fps": recording.get("fps", 20),
            "fmt": recording.get("fmt", "mp4"),
            "mode": recording.get("mode", "count"),
            "every_n_steps": recording.get("every_n_steps"),
            "max_recordings": recording.get("max_recordings"),
            "max_dim": recording.get("max_dim"),
        }

    # parallelism
    parallelism = spec.get("parallelism") or {}
    if isinstance(parallelism, dict):
        form["parallelism"] = {
            "num_workers": parallelism.get("num_workers", 1),
            "parallel_backend": parallelism.get("parallel_backend", "process"),
        }

    # coevolution
    coev = spec.get("coevolution") or {}
    if isinstance(coev, dict):
        form["coevolution"] = {
            "enabled": coev.get("enabled", False),
            "mode": coev.get("mode", "competitive_2pop"),
            "schedule": coev.get("schedule", "synchronous"),
            "k_opponents": coev.get("k_opponents", 3),
            "aggregation": coev.get("aggregation", "mean"),
            "alternating_generations": coev.get("alternating_generations", 1),
            "reporting_side": coev.get("reporting_side"),
            "sides": coev.get("sides", []),
        }

    # raw_config → raw_text (the free-form YAML/JSON overlay)
    raw_config = spec.get("raw_config") or {}
    if isinstance(raw_config, dict) and raw_config:
        import json
        form["raw_text"] = json.dumps(raw_config, indent=2)
    else:
        form["raw_text"] = ""

    return form


def extract_loaded_spec(setup_config_json: Optional[str]) -> Optional[Dict[str, Any]]:
    """Check whether ``setup_config_json`` contains a loaded spec from a re-run.

    Returns the ExperimentSpec dict if present, ``None`` otherwise.
    """
    if not setup_config_json:
        return None
    try:
        import json
        data = json.loads(setup_config_json)
    except (json.JSONDecodeError, TypeError):
        return None
    spec = data.get("_loaded_spec")
    if isinstance(spec, dict) and spec.get("environment"):
        return spec
    return None
