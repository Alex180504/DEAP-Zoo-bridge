"""
AgileRL <-> PettingZoo adapter utilities.

Provides helpers to:
- Build a raw PettingZoo parallel environment compatible with AgileRL's
  multi-agent training APIs (reset -> (obs, info), step -> 5-tuple).
- Extract per-agent observation/action spaces as the list-aligned structure
  AgileRL constructors expect.
- Translate a :class:`BudgetMode` to an environment-step budget.

Environment resolution goes through
:mod:`deap_pettingzoo.environments.registry`; any registered env is
accepted.  No direct ``mpe2`` import happens here.
"""

from typing import Any, Dict, List, Optional, Tuple

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.environments.registry import (
    DEFAULT_EPISODE_LEN_FALLBACK,
    build_env,
    get_env_spec,
    is_registered,
)


def build_parallel_env(
    env_name: str,
    env_config: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
):
    """Build a raw PettingZoo parallel environment.

    Args:
        env_name: Registered PettingZoo environment name.
        env_config: Optional kwargs forwarded to ``parallel_env``.
        render_mode: Optional render mode ('human', 'rgb_array', None).

    Returns:
        A PettingZoo parallel environment.

    Raises:
        ValueError: If ``env_name`` is not registered.  See
            :func:`deap_pettingzoo.environments.registry.register_env`.
        TypeError: If the factory/api declaration is inconsistent
            (propagated from
            :func:`~deap_pettingzoo.environments.registry.build_env`).
    """
    return build_env(env_name, env_config=env_config, render_mode=render_mode)


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
    config: TrainingConfig, default_episode_len: int = DEFAULT_EPISODE_LEN_FALLBACK
) -> int:
    """Translate a :class:`TrainingConfig` budget to an env-step count.

    Supports :class:`BudgetMode.ENV_STEPS` directly and
    :class:`BudgetMode.EVAL_EPISODES` by multiplying with the env's
    registered :attr:`EnvSpec.default_episode_len`, falling back to
    ``default_episode_len`` for unregistered envs.

    Args:
        config: Training configuration.
        default_episode_len: Fallback episode length for environments not
            present in the registry (MPE default is 25).

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
        if is_registered(config.env_name):
            ep_len = get_env_spec(config.env_name).default_episode_len
        else:
            ep_len = default_episode_len
        return int(config.budget_value) * int(ep_len)
    raise ValueError(
        f"Cannot translate budget mode {mode} to env steps; "
        "wall-clock training must use a timer-based stop condition."
    )

__all__ = [
    "build_parallel_env",
    "extract_space_metadata",
    "translate_budget_to_env_steps",
]
