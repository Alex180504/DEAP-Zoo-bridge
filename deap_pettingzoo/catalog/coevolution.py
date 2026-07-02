"""Co-evolution option catalog.

Exposes the co-evolution configuration vocabulary — modes, schedules, opponent
samplers / pools, aggregation methods and which algorithm optimizers may sit on
a side — as plain JSON-able data the setup UI can render into dropdowns.

The mode / schedule / aggregation constants mirror the authoritative tuples in
:mod:`deap_pettingzoo.baselines.coevolution.trainer` and
:mod:`...coevolution.evaluator`. They are duplicated here (rather than imported)
so the catalog stays import-light; a Phase 0 drift test asserts the two stay in
sync, giving single-source-of-truth-by-test without importing the heavy
co-evolution stack to build a dropdown.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Mirror of ``coevolution.trainer.VALID_MODES`` (drift-guarded by tests).
COEV_MODES = ("competitive_2pop", "self_play_hof", "cooperative")
# Mirror of ``coevolution.trainer.VALID_SCHEDULES``.
COEV_SCHEDULES = ("synchronous", "alternating")
# Mirror of ``coevolution.evaluator.VALID_AGGREGATIONS``.
COEV_AGGREGATIONS = ("mean", "min", "max", "sum")
# Opponent samplers exposed by ``coevolution.samplers``.
COEV_SAMPLERS = ("uniform", "latest", "fitness_proportional", "tournament")

_MODE_LABELS = {
    "competitive_2pop": "Competitive (2 populations)",
    "self_play_hof": "Self-play (hall of fame)",
    "cooperative": "Cooperative",
}
_MODE_HELP = {
    "competitive_2pop": (
        "Two (or more) sides with opposing rewards co-adapt against the "
        "current generation of their opponents."
    ),
    "self_play_hof": (
        "Each side samples opponents from its own archive of past champions "
        "(hall of fame) rather than the live generation."
    ),
    "cooperative": ("All sides share the same reward sign and are jointly optimised."),
}
_SCHEDULE_HELP = {
    "synchronous": "All sides advance one generation together.",
    "alternating": "Sides take turns advancing for N inner generations.",
}

#: Tokens accepted by ``Side.optimizer`` that correspond to the standard,
#: UI-exposable algorithms (plus the special ``frozen`` opponent). QDax / NEAT /
#: HyperNEAT optimizers are Python-API-only per the UI spec and excluded here.
_FROZEN_OPTIMIZER = "frozen"


def eligible_side_optimizers() -> List[Dict[str, Any]]:
    """Return the optimizer tokens selectable for a co-evolution side.

    Each entry maps a co-evolution ``Side.optimizer`` token to the catalog
    algorithm(s) it represents, so the UI can present algorithm names while
    submitting the optimizer token. ``frozen`` is always available (a fixed,
    non-learning opponent).
    """
    from deap_pettingzoo.catalog import algorithms

    by_optimizer: Dict[str, List[str]] = {}
    for spec in algorithms.list_algorithms():
        token = spec.coevolution_optimizer
        if token is None:
            continue
        by_optimizer.setdefault(token, []).append(spec.name)

    entries = [
        {"optimizer": token, "algorithms": names}
        for token, names in by_optimizer.items()
    ]
    entries.append({"optimizer": _FROZEN_OPTIMIZER, "algorithms": []})
    return entries


def get_coevolution_options() -> Dict[str, Any]:
    """Return the full co-evolution option set as JSON-able data."""
    return {
        "modes": [
            {
                "name": mode,
                "label": _MODE_LABELS.get(mode, mode),
                "help": _MODE_HELP.get(mode, ""),
            }
            for mode in COEV_MODES
        ],
        "schedules": [
            {
                "name": sched,
                "label": sched.title(),
                "help": _SCHEDULE_HELP.get(sched, ""),
            }
            for sched in COEV_SCHEDULES
        ],
        "aggregations": list(COEV_AGGREGATIONS),
        "samplers": list(COEV_SAMPLERS),
        "side_optimizers": eligible_side_optimizers(),
    }
