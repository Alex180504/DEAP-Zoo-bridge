"""Tests for new EA strategy kwargs on :class:`EATrainer`."""

import pytest

from deap_pettingzoo.baselines import BudgetMode, EATrainer, TrainingConfig


def _tiny_config(seed: int = 0) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=12,  # pop_size=6 -> 2 generations
        num_seeds=1,
        seed=seed, max_steps_per_episode=10,
    )


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError, match="Unknown EA strategy"):
        EATrainer(strategy="nsga2")


def test_comma_lambda_requires_lambda_ge_mu():
    with pytest.raises(ValueError, match="lambda_ >= mu"):
        EATrainer(
            strategy="mu_comma_lambda",
            population_size=10,
            mu=10,
            lambda_=5,
        )


def test_defaults_mu_lambda_to_population_size():
    trainer = EATrainer(strategy="mu_plus_lambda", population_size=7)
    assert trainer.mu == 7
    assert trainer.lambda_ == 7


@pytest.mark.parametrize(
    "strategy", ["simple", "mu_plus_lambda", "mu_comma_lambda"],
)
def test_strategy_smoke_train(strategy):
    """
    Each strategy produces a working policy and populated history.
   
    Notes:
    the best fitness observed across all recorded generations (`best_ever`) must be >= gen 0's maximum fitness.
    Equality is acceptable for tiny runs where performance may plateau.
    A strict decrease indicates a regression bug.
    """
    trainer = EATrainer(
        hidden_size=8,
        population_size=6,
        strategy=strategy,
        # Keep mu/lambda identical to pop for mu_comma_lambda validity.
        mu=6,
        lambda_=6,
    )
    result = trainer.train(_tiny_config(seed=0))
    assert result.algorithm == "ea"
    assert result.policy is not None
    history = result.metrics["fitness_history"]
    assert len(history) >= 2
    assert result.metadata["ea_config"]["strategy"] == strategy

    best_ever = max(rec["max"] for rec in history)
    assert best_ever >= history[0]["max"], (
        f"best-ever fitness regressed for strategy={strategy!r}: "
        f"gen0_max={history[0]['max']}, best_ever={best_ever}"
    )


def test_method_kwargs_propagate_to_metadata():
    """Verify that explicitly selected EA operator methods are preserved in training metadata."""
    trainer = EATrainer(
        hidden_size=8,
        population_size=4,
        strategy="simple",
        selection_method="tournament",
        crossover_method="sbx",
        mutation_method="polynomial_bounded",
        crossover_params={"eta": 10.0},
        mutation_params={"eta": 10.0, "indpb": 0.5},
    )
    result = trainer.train(_tiny_config(seed=1))
    meta = result.metadata["ea_config"]
    assert meta["selection_method"] == "tournament"
    assert meta["crossover_method"] == "sbx"
    assert meta["mutation_method"] == "polynomial_bounded"
