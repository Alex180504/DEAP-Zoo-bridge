"""PSO coevolution smoke test (Phase 1.4)."""
import pytest

pytest.importorskip("deap")
pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    Side,
)


def _pso_cfg(seed: int) -> dict:
    return {"population_size": 4, "hidden_size": 8, "seed": seed}


def _base_tcfg(cfg: CoEvolutionConfig, budget: int = 40) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_tag_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=budget,
        coevolution_config=cfg,
    )


class TestPSOCoevolution:
    def test_pso_vs_ea(self):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pso_side", ("adversary",), "pso", _pso_cfg(0)),
                Side("ea_side", ("agent",), "ea",
                     {"population_size": 4, "hidden_size": 8, "seed": 1}),
            ],
            k_opponents=1,
            max_steps_per_episode=15,
            seed=42,
        )
        result = CoEvolutionTrainer().train(_base_tcfg(cfg))
        assert result.metrics is not None
