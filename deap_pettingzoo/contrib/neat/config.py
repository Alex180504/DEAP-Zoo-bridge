"""Programmatic NEAT configuration builder.

Builds a :class:`neat.Config` from a few user-facing parameters without
requiring the caller to write a NEAT ``.ini`` config file.

The template is written to a **temporary file** because ``neat.Config``
requires a filesystem path.  The file is deleted immediately after the
:class:`~neat.Config` object is built; the resulting ``Config`` object
is fully self-contained (the file is never read again).

All mutation-rate, structural, and stagnation defaults are sensible for
the small-to-medium neuroevolution problems this library targets, but
they can be overridden via the ``overrides`` dict.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

import neat

# Sensible defaults for small-to-medium neuroevolution problems
# 
_DEFAULT_NEAT_CONFIG_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "NEAT": {
        "fitness_criterion": "max",
        "fitness_threshold": 1e9,  # effectively no early stop
        "pop_size": 50,
        "reset_on_extinction": False,
        "no_fitness_termination": True,
    },
    "DefaultGenome": {
        "num_hidden": 0,
        "feed_forward": True,
        "activation_default": "tanh",
        "activation_mutate_rate": 0.05,
        "activation_options": "tanh sigmoid relu identity",
        "aggregation_default": "sum",
        "aggregation_mutate_rate": 0.0,
        "aggregation_options": "sum",
        "bias_init_mean": 0.0,
        "bias_init_stdev": 0.2,
        "bias_init_type": "gaussian",
        "bias_replace_rate": 0.1,
        "bias_mutate_rate": 0.7,
        "bias_mutate_power": 0.5,
        "bias_max_value": 30.0,
        "bias_min_value": -30.0,
        "response_init_mean": 1.0,
        "response_init_stdev": 0.0,
        "response_init_type": "gaussian",
        "response_replace_rate": 0.0,
        "response_mutate_rate": 0.0,
        "response_mutate_power": 0.0,
        "response_max_value": 30.0,
        "response_min_value": -30.0,
        "time_constant_init_mean": 1.0,
        "time_constant_init_stdev": 0.0,
        "time_constant_init_type": "gaussian",
        "time_constant_replace_rate": 0.0,
        "time_constant_mutate_rate": 0.0,
        "time_constant_mutate_power": 0.0,
        "time_constant_max_value": 30.0,
        "time_constant_min_value": -30.0,
        "weight_init_mean": 0.0,
        "weight_init_stdev": 1.0,
        "weight_init_type": "gaussian",
        "weight_replace_rate": 0.1,
        "weight_mutate_rate": 0.8,
        "weight_mutate_power": 0.5,
        "weight_max_value": 30.0,
        "weight_min_value": -30.0,
        "enabled_default": True,
        "enabled_mutate_rate": 0.1,
        "enabled_rate_to_true_add": 0.0,
        "enabled_rate_to_false_add": 0.0,
        "compatibility_disjoint_coefficient": 1.0,
        "compatibility_weight_coefficient": 0.5,
        "conn_add_prob": 0.5,
        "conn_delete_prob": 0.2,
        "node_add_prob": 0.2,
        "node_delete_prob": 0.1,
        "single_structural_mutation": False,
        "structural_mutation_surer": "default",
        "initial_connection": "full_direct",
    },
    "DefaultSpeciesSet": {
        "compatibility_threshold": 3.0,
    },
    "DefaultStagnation": {
        "species_fitness_func": "max",
        "max_stagnation": 15,
        "species_elitism": 2,
    },
    "DefaultReproduction": {
        "elitism": 2,
        "survival_threshold": 0.2,
        "min_species_size": 2,
    },
}


def _dict_to_neat_ini(d: Dict[str, Dict[str, Any]]) -> str:
    """Convert a nested dict to a NEAT-compatible ``.ini``-style string."""
    lines: list[str] = []
    for section, items in d.items():
        lines.append(f"[{section}]")
        for key, value in items.items():
            if isinstance(value, bool):
                value = str(value).lower()
            lines.append(f"{key} = {value}")
        lines.append("")
    return "\n".join(lines)


def build_neat_config(
    input_size: int,
    output_size: int,
    *,
    pop_size: int = 50,
    seed: Optional[int] = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> neat.Config:
    """Build a :class:`neat.Config` programmatically.

    Args:
        input_size: Dimension of the observation vector (num inputs).
        output_size: Dimension of the action vector / logits (num outputs).
        pop_size: Population size for the NEAT run.
        seed: Optional random seed placed into ``[NEAT] seed = ...``.
        overrides: Optional nested dict of section → key → value to
            override any default parameter.

    Returns:
        A fully-initialised :class:`neat.Config` suitable for use with
        :class:`neat.Population`.
    """
    # Merge defaults with overrides.
    cfg: Dict[str, Dict[str, Any]] = {}
    for section, items in _DEFAULT_NEAT_CONFIG_OVERRIDES.items():
        cfg[section] = dict(items)

    # Mandatory fields from caller (set before overrides so overrides win).
    cfg["NEAT"]["pop_size"] = pop_size
    cfg["DefaultGenome"]["num_inputs"] = input_size
    cfg["DefaultGenome"]["num_outputs"] = output_size
    if seed is not None:
        cfg["NEAT"]["seed"] = int(seed)

    # Apply caller overrides (top priority).
    overrides = overrides or {}
    for section, items in overrides.items():
        cfg.setdefault(section, {}).update(items)

    # Write ini to temp file, build Config, delete file.
    ini_text = _dict_to_neat_ini(cfg)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cfg", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(ini_text)
        tmp.close()
        return neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            tmp.name,
        )
    finally:
        os.unlink(tmp.name)


# CPPN-specific config

#: CPPN-compatible activation functions — includes periodic (sin),
#: Gaussian, absolute-value, and clamped variants that enable the CPPN
#: to produce regular patterns.
CPPN_ACTIVATION_OPTIONS = "sin sigmoid gauss tanh relu identity abs clamped square"

#: CPPN defaults — wider weight ranges and more activation variety
#: than the standard neuroevolution config.
_CPPN_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "DefaultGenome": {
        "activation_default": "sin",
        "activation_mutate_rate": 0.1,
        "activation_options": CPPN_ACTIVATION_OPTIONS,
        "aggregation_default": "sum",
        "aggregation_options": "sum product",
        "aggregation_mutate_rate": 0.0,
        "weight_init_mean": 0.0,
        "weight_init_stdev": 1.0,
        "weight_init_type": "gaussian",
        "weight_mutate_rate": 0.4,
        "weight_mutate_power": 0.2,
        "weight_max_value": 8.0,
        "weight_min_value": -8.0,
    },
}


def build_cppn_config(
    input_size: int,
    output_size: int,
    *,
    pop_size: int = 50,
    seed: Optional[int] = None,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> neat.Config:
    """Build a NEAT :class:`~neat.Config` suitable for **CPPN** evolution.

    This config uses periodic / Gaussian activation functions and
    different weight mutation rates than :func:`build_neat_config`,
    making it appropriate for Compositional Pattern Producing Networks.

    The resulting config can be used directly with
    :class:`~deap_pettingzoo.contrib.neat.NEATTrainer` — the CPPN is
    just a regular NEAT genome with different activation defaults.

    Parameters
    ----------
    input_size:
        Number of CPPN inputs (typically 5: x1, y1, x2, y2, bias).
    output_size:
        Number of CPPN outputs (typically 1 for weight queries).
    pop_size:
        Population size.
    seed:
        Optional RNG seed.
    overrides:
        Optional section→key→value overrides (merged on top of
        defaults).

    Returns
    -------
    neat.Config
    """
    # Start from the base defaults, then layer CPPN-specific overrides.
    base_overrides: Dict[str, Dict[str, Any]] = {}
    for section in _CPPN_DEFAULTS:
        base_overrides[section] = dict(_CPPN_DEFAULTS[section])

    # Merge caller overrides on top.
    if overrides:
        for section, items in overrides.items():
            base_overrides.setdefault(section, {}).update(items)

    return build_neat_config(
        input_size=input_size,
        output_size=output_size,
        pop_size=pop_size,
        seed=seed,
        overrides=base_overrides if base_overrides else None,
    )
