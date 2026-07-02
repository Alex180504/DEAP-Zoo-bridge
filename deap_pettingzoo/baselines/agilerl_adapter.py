"""
AgileRL <-> PettingZoo adapter utilities.

Provides helpers to:
- Build a raw PettingZoo parallel environment compatible with AgileRL's
  multi-agent training APIs (reset -> (obs, info), step -> 5-tuple).
- Build a vectorized PettingZoo environment via AgileRL's
    :class:`AsyncPettingZooVecEnv` for AgileRL trainers that opt into
    ``num_envs > 1``.
- Extract per-agent observation/action spaces as the list-aligned structure
  AgileRL constructors expect (works for both single and vec envs).
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
    build_env_from_spec,
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


def build_vec_parallel_env(
    env_name: str,
    env_config: Optional[Dict[str, Any]] = None,
    num_envs: int = 1,
    render_mode: Optional[str] = None,
):
    """Build an :class:`AsyncPettingZooVecEnv` of ``num_envs`` workers.

    Each worker is a fresh independent ``build_env(...)`` instance.
    Workers run in subprocesses (``multiprocessing``); per-env auto-reset
    on episode end is handled inside the worker. Observations, rewards
    and termination/truncation flags returned from ``reset``/``step``
    have a leading ``num_envs`` axis (i.e. ``obs[a].shape ==
    (num_envs, *single_obs_shape)``).

    Args:
        env_name: Registered PettingZoo environment name.
        env_config: Optional kwargs forwarded to ``parallel_env``
            (same kwargs each worker receives).
        num_envs: Number of parallel sub-environments. Must be ``>= 1``.
        render_mode: Render mode forwarded to each worker.

    Returns:
        ``AsyncPettingZooVecEnv`` instance (subclass of
        ``PettingZooVecEnv``). Exposes ``num_envs`` attribute,
        ``possible_agents`` list, ``single_observation_space(agent)`` /
        ``single_action_space(agent)`` callables, and ``reset`` / ``step``
        with batched dict outputs.

    Raises:
        ImportError: If AgileRL's vector module is not importable.
        ValueError: If ``num_envs < 1`` or ``env_name`` is not registered
            (propagated from :func:`build_env`).
    """
    if int(num_envs) < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs!r}")
    try:
        from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "AsyncPettingZooVecEnv requires agilerl >= 2.6. "
            "Install with: pip install 'deap-pettingzoo[rl-baselines]'"
        ) from exc

    cfg = dict(env_config) if env_config else None
    # Resolve the EnvSpec in the main process so each subprocess worker can call build_env_from_spec(spec, ...) directly
    spec = get_env_spec(env_name)

    def _factory():
        return build_env_from_spec(spec, env_config=cfg, render_mode=render_mode)

    return AsyncPettingZooVecEnv([_factory for _ in range(int(num_envs))])


def extract_space_metadata(env) -> Tuple[List[Any], List[Any], List[str]]:
    """Extract per-agent spaces in the order expected by AgileRL algorithms.

    Works for both raw PettingZoo parallel envs and AgileRL's
    :class:`AsyncPettingZooVecEnv`. For vec envs, returns the
    *single-env* spaces (not the batched ones), since AgileRL
    algorithm constructors expect single-env shapes.

    Args:
        env: A PettingZoo parallel environment OR an
            ``AsyncPettingZooVecEnv``.

    Returns:
        Tuple ``(observation_spaces, action_spaces, agent_ids)`` where each
        list is indexed by ``possible_agents``.
    """
    agent_ids: List[str] = list(env.possible_agents)
    # Vec envs expose ``single_observation_space``/``single_action_space``
    # callables that return per-env spaces; raw parallel envs expose
    # ``observation_space``/``action_space`` directly.
    if hasattr(env, "num_envs") and hasattr(env, "single_observation_space"):
        observation_spaces = [env.single_observation_space(a) for a in agent_ids]
        action_spaces = [env.single_action_space(a) for a in agent_ids]
    else:
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
        ValueError: If ``config.budget_mode`` is neither ``ENV_STEPS`` nor
            ``EVAL_EPISODES`` — notably ``GENERATIONS``, which is a
            population-trainer-only unit with no meaning for the step-based
            AgileRL learners.
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
        f"AgileRL trainers cannot translate budget mode {mode} to env steps. "
        "Use BudgetMode.ENV_STEPS or BudgetMode.EVAL_EPISODES; "
        "BudgetMode.GENERATIONS is supported only by the population trainers "
        "(EA / CMA-ES / PSO / co-evolution / QDax)."
    )


__all__ = [
    "build_parallel_env",
    "build_vec_parallel_env",
    "extract_space_metadata",
    "translate_budget_to_env_steps",
]
