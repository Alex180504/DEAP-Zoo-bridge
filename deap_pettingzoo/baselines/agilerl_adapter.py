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

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig


# Environments supported by both our EA code and AgileRL's multi-agent algos.
SUPPORTED_PARALLEL_ENVS = {
    "simple_tag_v3": simple_tag_v3,
    "simple_adversary_v3": simple_adversary_v3,
    "simple_spread_v3": simple_spread_v3,
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
    :class:`BudgetMode.EVAL_EPISODES` by multiplying with
    ``default_episode_len``.

    Args:
        config: Training configuration.
        default_episode_len: Assumed episode length for the
            ``EVAL_EPISODES`` mode (MPE default is 25).

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
        return int(config.budget_value) * default_episode_len
    raise ValueError(
        f"Cannot translate budget mode {mode} to env steps; "
        "wall-clock training must use a timer-based stop condition."
    )
