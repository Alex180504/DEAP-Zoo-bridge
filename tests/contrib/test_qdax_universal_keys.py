"""Regression tests threading TrainingConfig's universal keys (seed,
max_steps_per_episode) into the QDax trainers' algorithm_params dict.

See ``deap_pettingzoo.contrib.qdax.trainer._apply_universal_keys``.
"""

import pytest

pytest.importorskip("qdax")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.contrib.qdax.trainer import _apply_universal_keys


def _config(**overrides):
    base = dict(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.GENERATIONS,
        budget_value=1,
    )
    base.update(overrides)
    return TrainingConfig(**base)


class TestApplyUniversalKeys:
    def test_config_seed_overrides_default(self):
        params = {"seed": 0, "episode_length": 25}
        _apply_universal_keys(_config(seed=7), params)
        assert params["seed"] == 7

    def test_config_seed_overrides_explicit_algorithm_params_seed(self):
        """config.seed always wins, mirroring EA/CMA/PSO's strict switch."""
        params = {"seed": 99, "episode_length": 25}
        _apply_universal_keys(
            _config(seed=7, algorithm_params={"seed": 99}), params
        )
        assert params["seed"] == 7

    def test_no_config_seed_leaves_params_untouched(self):
        params = {"seed": 42, "episode_length": 25}
        _apply_universal_keys(_config(), params)
        assert params["seed"] == 42

    def test_max_steps_per_episode_fills_episode_length_default(self):
        params = {"seed": 0, "episode_length": 25}
        _apply_universal_keys(_config(max_steps_per_episode=64), params)
        assert params["episode_length"] == 64

    def test_explicit_episode_length_wins_over_max_steps_per_episode(self):
        """An explicit algorithm_params['episode_length'] survives -- it also
        fixes the JAX-scan trajectory shape, so it must not be silently
        overridden by the generic first-class field."""
        params = {"seed": 0, "episode_length": 30}
        _apply_universal_keys(
            _config(
                max_steps_per_episode=64,
                algorithm_params={"episode_length": 30},
            ),
            params,
        )
        assert params["episode_length"] == 30
