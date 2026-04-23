"""Smoke tests for :mod:`deap_pettingzoo.contrib.openspiel`.

Skipped unless both :mod:`shimmy` and :mod:`pyspiel` are installed
(optional extra: ``pip install 'deap-pettingzoo[shimmy]'``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("shimmy")
pytest.importorskip("pyspiel")

from deap_pettingzoo.contrib.openspiel import register_openspiel_game
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.environments.registry import (
    get_env_spec,
    is_registered,
    unregister_env,
)


ENV_NAME = "test_openspiel_tictactoe"


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    if is_registered(ENV_NAME):
        unregister_env(ENV_NAME)


def test_register_openspiel_game_smoke():
    """``register_openspiel_game`` registers a turn-based spec."""
    spec = register_openspiel_game(
        ENV_NAME,
        game_name="tic_tac_toe",
        default_episode_len=9,
        overwrite=True,
    )
    assert spec.api == "aec_masked"
    assert "openspiel" in spec.tags
    assert "shimmy" in spec.tags
    assert "turn_based" in spec.tags

    # Spec must round-trip through the registry.
    assert is_registered(ENV_NAME)
    assert get_env_spec(ENV_NAME).api == "aec_masked"


def test_register_openspiel_game_rejects_parallel_api():
    with pytest.raises(ValueError, match="turn-based"):
        register_openspiel_game(
            ENV_NAME,
            game_name="tic_tac_toe",
            api="parallel",
            overwrite=True,
        )


def test_openspiel_selfplay_single_episode():
    """Build the env, run one :class:`SelfPlayEvaluator` episode end-to-end."""
    import numpy as np

    from deap_pettingzoo.evaluators import SelfPlayEvaluator
    from deap_pettingzoo.genomes import FixedTopologyGenome

    register_openspiel_game(
        ENV_NAME,
        game_name="tic_tac_toe",
        default_episode_len=9,
        overwrite=True,
    )

    wrapper = PettingZooWrapper(env_name=ENV_NAME)
    try:
        obs = wrapper.reset(seed=0)
        agent = next(iter(obs))
        obs_dim = int(np.asarray(obs[agent]).size)
        action_space = wrapper.action_space(agent)
        action_dim = action_space.n

        class _Ind:
            pass

        ind = _Ind()
        ind.genome = FixedTopologyGenome(
            layer_sizes=[obs_dim, 16, action_dim],
            activation="tanh",
            random_seed=0,
        )

        evaluator = SelfPlayEvaluator(
            wrapper,
            max_steps=18,
            aggregation="first_agent",
            masked=True,
            flatten_obs=True,
        )
        fitness = evaluator.evaluate(ind, seed=0)
        assert isinstance(fitness, tuple) and len(fitness) == 1
        assert np.isfinite(fitness[0])
    finally:
        wrapper.close()
