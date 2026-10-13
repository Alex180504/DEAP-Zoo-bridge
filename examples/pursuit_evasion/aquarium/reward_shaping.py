"""Potential-based progress shaping shared by the aquarium pursuit-evasion envs.

Every shaping site in the suite (the coevolution both-sides adapter and the
two vs-scripted-APF wrappers) rewards the same quantity: movement of the
nearest-opponent distance in the direction the role wants.  Expressing it as
a *potential* rather than a raw per-step delta,

.. math::

    \\Phi(s) = \\mp\\, d(\\text{nearest opponent}) / 200,
    \\qquad F(s, s') = \\Phi(s') - \\Phi(s)

(the sign is negative for pursuers, positive for evaders), makes the shaping
policy-invariant in the Ng-Harada-Russell sense: an episode's shaping total
telescopes to ``Phi(s_T) - Phi(s_0)`` regardless of the path taken, so it is
bounded by the arena diagonal (~10.6 on the 1500x1500 config) and cannot be
farmed by circling a prey at constant distance.  The previous formulation
summed unrelated per-step deltas, which had neither property.

Two rules make the telescoping actually hold in this environment:

* **No FOV gate.**  A potential must be defined over every opponent; gating
  on the view cone leaves ``Phi`` undefined on most steps (the shipped config
  has a 400-unit view distance on a 1500x1500 arena) and the sum no longer
  telescopes.
* **Identity keying.**  With ``keep_prey_count_constant=False`` caught prey
  are removed from the raw env's lists, so positional indexing silently
  re-pairs surviving agents with a different entity.  Look entities up by
  their own ``id()``.

The residual discontinuity — the nearest opponent being caught, which moves
``Phi`` by one target-switch rather than one step of motion — is bounded by an
optional symmetric ``cap`` on the raw distance delta, applied *before* the
scale divisor.  One step of plausible relative motion is well inside the cap,
so it never fires on ordinary movement.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

# Distance divisor: a shaping unit is 200 arena units of closing/opening.
POTENTIAL_SCALE: float = 200.0


def find_raw_env(env: Any) -> Optional[Any]:
    """Walk a PettingZoo wrapper chain down to the raw aquarium env.

    Args:
        env: Any env or wrapper in the chain.

    Returns:
        The first object exposing both ``predators`` and ``prey``, or ``None``
        when the chain bottoms out without one.
    """
    inner = env
    for _ in range(12):
        if hasattr(inner, "predators") and hasattr(inner, "prey"):
            return inner
        for attr in ("aec_env", "env"):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
                break
        else:
            break
    return None


def entities_by_id(entities: Iterable[Any]) -> Dict[str, Any]:
    """Map ``agent_id -> live entity`` keyed by the entity's own identity.

    Args:
        entities: Raw marl-aquarium entities (``raw.predators`` / ``raw.prey``).

    Returns:
        Dict from ``entity.id()`` to the entity.  Entities removed from the
        raw env (caught prey) are simply absent, which is what lets callers
        distinguish "dead" from "at distance 0".
    """
    return {entity.id(): entity for entity in entities if hasattr(entity, "id")}


def nearest_opponent_distances(
    agent_ids: Iterable[str],
    observers: Mapping[str, Any],
    targets: Iterable[Any],
    torus: Any,
) -> Dict[str, float]:
    """Torus distance from each live observer to its nearest live opponent.

    No FOV gating: the potential is defined over *all* surviving opponents
    (see the module docstring).

    Args:
        agent_ids: Fixed agent roster to iterate, in a stable order.
        observers: ``agent_id -> entity`` for the observing side, as built by
            :func:`entities_by_id`.  Agents missing from it are dead and are
            omitted from the result.
        targets: Surviving entities of the opposing side.
        torus: The raw env's torus helper (provides ``get_distance_in_torus``).

    Returns:
        ``agent_id -> distance`` for every live observer.  Empty when no
        opponent survives or the torus is unavailable — the potential is
        undefined there, and callers must skip shaping rather than substitute
        a value.
    """
    target_list = list(targets)
    if not target_list or torus is None:
        return {}
    distances: Dict[str, float] = {}
    for agent_id in agent_ids:
        entity = observers.get(agent_id)
        if entity is None:
            continue
        distances[agent_id] = min(
            float(torus.get_distance_in_torus(entity.position, target.position))
            for target in target_list
        )
    return distances


def potential_shaping(
    current: Mapping[str, float],
    previous: Mapping[str, float],
    *,
    alpha: float,
    closing: bool,
    cap: Optional[float] = None,
    scale: float = POTENTIAL_SCALE,
) -> Dict[str, float]:
    """Per-agent shaping term ``F = Phi(s') - Phi(s)`` for one side.

    Args:
        current: Nearest-opponent distances after the step.
        previous: The same quantity before the step.
        alpha: Shaping weight.
        closing: ``True`` for pursuers (reward shrinking distance), ``False``
            for evaders (reward growing distance).
        cap: Optional symmetric clip on the raw distance delta, applied before
            the scale divisor.  Bounds the target-switch discontinuity.
        scale: Distance divisor (:data:`POTENTIAL_SCALE`).

    Returns:
        ``agent_id -> shaping reward``, containing only agents whose potential
        is defined at *both* endpoints.  Agents that died, respawned, or had no
        opponent in range of definition are absent, so callers never credit a
        name that no longer maps to a live entity.
    """
    shaped: Dict[str, float] = {}
    for agent_id, current_distance in current.items():
        previous_distance = previous.get(agent_id)
        if previous_distance is None:
            continue
        delta = (
            previous_distance - current_distance
            if closing
            else current_distance - previous_distance
        )
        if cap is not None:
            delta = max(-cap, min(cap, delta))
        shaped[agent_id] = alpha * (delta / scale)
    return shaped
