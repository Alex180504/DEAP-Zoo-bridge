"""
AgileRL <-> PettingZoo adapter utilities.

Provides helpers to:
- Build a raw PettingZoo parallel environment compatible with AgileRL's
  multi-agent training APIs (reset -> (obs, info), step -> 5-tuple).
- Extract per-agent observation/action spaces as the list-aligned structure
  AgileRL constructors expect.
- Translate a :class:`BudgetMode` to an environment-step budget.
"""

from typing import Any, Dict, List, Optional, Tuple

from mpe2 import (
    simple_adversary_v3,
    simple_spread_v3,
    simple_tag_v3,
)

# Optional: marl-aquarium (pip install marl-aquarium). Imported lazily so
# users without the extra can still use MPE2 environments.
try:  # pragma: no cover - presence depends on optional extra
    from marl_aquarium import aquarium_v0 as _aquarium_v0
except ImportError:  # pragma: no cover
    _aquarium_v0 = None

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig


# Environments supported by both our EA code and AgileRL's multi-agent algos.
SUPPORTED_PARALLEL_ENVS = {
    "simple_tag_v3": simple_tag_v3,
    "simple_adversary_v3": simple_adversary_v3,
    "simple_spread_v3": simple_spread_v3,
}
if _aquarium_v0 is not None:
    SUPPORTED_PARALLEL_ENVS["aquarium_v0"] = _aquarium_v0


# Per-env default episode length, used when translating
# BudgetMode.EVAL_EPISODES into env steps. MPE family defaults to
# ``max_cycles=25``; marl-aquarium defaults to ``max_time_steps=3000``.
DEFAULT_EPISODE_LEN = {
    "simple_tag_v3": 25,
    "simple_adversary_v3": 25,
    "simple_spread_v3": 25,
    "aquarium_v0": 3000,
}


def build_parallel_env(
    env_name: str,
    env_config: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
):
    """Build a raw PettingZoo parallel environment.

    Args:
        env_name: PettingZoo environment name (e.g. ``'simple_spread_v3'``).
        env_config: Optional kwargs forwarded to ``parallel_env``.
        render_mode: Optional render mode ('human', 'rgb_array', None).

    Returns:
        A PettingZoo parallel environment.

    Raises:
        ValueError: If ``env_name`` is not in :data:`SUPPORTED_PARALLEL_ENVS`.
    """
    if env_name not in SUPPORTED_PARALLEL_ENVS:
        raise ValueError(
            f"Unsupported environment '{env_name}'. "
            f"Supported: {list(SUPPORTED_PARALLEL_ENVS.keys())}"
        )
    module = SUPPORTED_PARALLEL_ENVS[env_name]
    return module.parallel_env(render_mode=render_mode, **(env_config or {}))


def extract_space_metadata(env) -> Tuple[List[Any], List[Any], List[str]]:
    """Extract per-agent spaces in the order expected by AgileRL algorithms.

    Args:
        env: A PettingZoo parallel environment.

    Returns:
        Tuple ``(observation_spaces, action_spaces, agent_ids)`` where each
        list is indexed by ``possible_agents``.
    """
    agent_ids: List[str] = list(env.possible_agents)
    observation_spaces = [env.observation_space(a) for a in agent_ids]
    action_spaces = [env.action_space(a) for a in agent_ids]
    return observation_spaces, action_spaces, agent_ids


def translate_budget_to_env_steps(
    config: TrainingConfig, default_episode_len: int = 25
) -> int:
    """Translate a :class:`TrainingConfig` budget to an env-step count.

    Supports :class:`BudgetMode.ENV_STEPS` directly and
    :class:`BudgetMode.EVAL_EPISODES` by multiplying with an env-specific
    default episode length (see :data:`DEFAULT_EPISODE_LEN`), falling
    back to ``default_episode_len`` for unknown envs.

    Args:
        config: Training configuration.
        default_episode_len: Fallback episode length for environments not
            listed in :data:`DEFAULT_EPISODE_LEN` (MPE default is 25).

    Returns:
        Integer number of environment steps.

    Raises:
        ValueError: If ``config.budget_mode`` is ``WALL_CLOCK`` (not
            supported here; trainer should short-circuit on a timer).
    """
    mode = config.budget_mode
    if mode == BudgetMode.ENV_STEPS:
        return int(config.budget_value)
    if mode == BudgetMode.EVAL_EPISODES:
        ep_len = DEFAULT_EPISODE_LEN.get(config.env_name, default_episode_len)
        return int(config.budget_value) * int(ep_len)
    raise ValueError(
        f"Cannot translate budget mode {mode} to env steps; "
        "wall-clock training must use a timer-based stop condition."
    )
