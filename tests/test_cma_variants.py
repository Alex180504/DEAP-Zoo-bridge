"""Tests for the (1+λ)-CMA-ES variant plumbing in :class:`CMAESTrainer`."""

import numpy as np
import pytest

from deap_pettingzoo.baselines import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer
from deap_pettingzoo.integration import (
    setup_cma_one_plus_lambda_strategy,
    setup_cma_strategy,
)


def _tiny_config(seed: int = 0) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_tag_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=12,
        num_seeds=1,
        algorithm_params={"seed": seed, "max_steps_per_episode": 10},
    )


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="Unknown CMA-ES variant"):
        CMAESTrainer(variant="1+1")


def test_setup_one_plus_lambda_helper_returns_correct_dim():
    genome_config = {"layer_sizes": [4, 4, 2], "activation": "tanh",
                     "weight_range": (-1.0, 1.0)}
    strategy, dim = setup_cma_one_plus_lambda_strategy(
        genome_config=genome_config,
        sigma0=0.3,
        population_size=4,
    )
    # Sanity: dimension matches a freshly-built strategy from the
    # standard helper.
    _, dim_ref = setup_cma_strategy(genome_config, sigma0=0.3)
    assert dim == dim_ref
    assert hasattr(strategy, "generate")
    assert hasattr(strategy, "update")


def test_setup_one_plus_lambda_validates_parent_shape():
    with pytest.raises(ValueError, match="parent has shape"):
        setup_cma_one_plus_lambda_strategy(
            genome_config={"layer_sizes": [4, 4, 2]},
            parent=np.zeros(3),
        )


@pytest.mark.parametrize("variant", ["mu_mu_lambda", "one_plus_lambda"])
def test_cma_variant_smoke_train(variant):
    trainer = CMAESTrainer(
        hidden_size=8,
        population_size=6,
        sigma0=0.3,
        variant=variant,
    )
    result = trainer.train(_tiny_config(seed=0))
    assert result.algorithm == "cma-es"
    assert result.metadata["cma_config"]["variant"] == variant
    assert len(result.metrics["fitness_history"]) >= 1
