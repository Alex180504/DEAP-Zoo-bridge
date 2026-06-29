"""Regression tests for budget fairness.

These pin the behaviours that make an "equal budget" EA-vs-RL comparison
actually fair:

* Generation back-derivation rounds to nearest and *warns* when a budget is
  too small to fund more than one generation.
* The initial population evaluation (generation 0) is charged on the env-step
  axis instead of reporting zero.
* Population evaluators expose a real ``env_steps_consumed`` counter that is
  below the nominal ``population * max_steps`` whenever episodes end early.
* ``BudgetMode.GENERATIONS`` is honored by the population trainers and clearly
  rejected by the step-based AgileRL trainers (``WALL_CLOCK`` is gone).
"""

import logging

import pytest

from deap_pettingzoo.baselines import (
    BudgetMode,
    CMAESTrainer,
    EATrainer,
    PSOTrainer,
    TrainingConfig,
)
from deap_pettingzoo.baselines.agilerl_adapter import translate_budget_to_env_steps
from deap_pettingzoo.baselines.base import derive_generations
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.genomes import FixedTopologyGenome


class _Individual:
    """Minimal DEAP-style individual carrying a genome."""

    def __init__(self, genome: FixedTopologyGenome) -> None:
        self.genome = genome




class TestGenerationDerivation:
    def test_underderivation_warns_and_clamps(self):
        """A budget below one generation's cost clamps to 1 and warns."""
        trainer = EATrainer(population_size=20)
        # 20 * 200 = 4000 env steps / generation; 1000 funds only 0.25 gens.
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=1000,
            algorithm_params={"max_steps_per_episode": 200},
        )
        with pytest.warns(RuntimeWarning, match="clamped to 1 generation"):
            params = trainer._translate_budget(config)
        assert params["num_generations"] == 1

    def test_helper_rounds_to_nearest(self):
        """The shared helper rounds rather than floors (no silent truncation)."""
        # 2.6 generations should round up to 3, not floor to 2.
        assert derive_generations(2600, 10, 100) == 3
        # A comfortable budget does not warn.
        assert derive_generations(10000, 10, 100) == 10

    @pytest.mark.parametrize(
        "budget,expected",
        [
            (1900, 2),  # 1.9 gens -> 2 (floor would give 1)
            (2100, 2),  # 2.1 gens -> 2
            (3500, 4),  # 3.5 gens -> banker's rounding to 4 (even)
            (2500, 2),  # 2.5 gens -> banker's rounding to 2 (even)
        ],
    )
    def test_helper_rounding_boundaries(self, budget, expected):
        """Round-to-nearest (incl. banker's rounding at .5) over floor."""
        assert derive_generations(budget, 10, 100) == expected

    def test_eval_episodes_clamp_warns(self):
        """The clamp warning also fires under an EVAL_EPISODES budget."""
        trainer = EATrainer(population_size=20)
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=5,  # 5 episodes < one population (20) -> clamp
        )
        with pytest.warns(RuntimeWarning, match="clamped to 1 generation"):
            params = trainer._translate_budget(config)
        assert params["num_generations"] == 1

    def test_helper_warns_only_on_clamp(self, recwarn):
        """A budget that funds many generations emits no warning."""
        derive_generations(10000, 10, 100)
        assert not recwarn.list

    def test_logs_warning_to_module_logger(self, caplog):
        """The clamp warning is also emitted to the logger (service.log)."""
        with caplog.at_level(logging.WARNING):
            with pytest.warns(RuntimeWarning):
                derive_generations(100, 10, 100, label="EA")
        assert any("clamped" in rec.message for rec in caplog.records)




class TestGenZeroCharge:
    def test_gen0_progress_event_reports_zero_env_steps(self):
        """The gen-0 progress event reports env_steps == 0 so the comparison
        chart x-axis aligns with RL algorithms that start from step 0."""
        pop = 4
        max_steps = 25
        trainer = EATrainer(population_size=pop, strategy="simple")
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=1,
            env_config={"max_cycles": max_steps},
            algorithm_params={
                "max_steps_per_episode": max_steps,
                "seed": 0,
            },
        )
        events = []
        trainer.train(config, progress_callback=events.append)

        assert events, "expected at least one progress event"
        gen0 = events[0]
        assert gen0.current == 0
        assert gen0.env_steps == 0



class TestEnvStepsConsumed:
    def test_counter_below_nominal_on_early_termination(self):
        """Real consumed steps are below the nominal max_steps when the env
        truncates early."""
        wrapper = PettingZooWrapper(
            env_name="simple_spread_v3",
            env_config={"max_cycles": 25},
        )
        try:
            obs = wrapper.reset(seed=0)
            agent = next(iter(obs))
            genome = FixedTopologyGenome(
                layer_sizes=[len(obs[agent]), 16, wrapper.action_space(agent).n],
                random_seed=0,
            )
            nominal_max_steps = 200
            evaluator = SingleEpisodeEvaluator(
                env_wrapper=wrapper, max_steps=nominal_max_steps
            )
            assert evaluator.env_steps_consumed == 0
            evaluator.evaluate(_Individual(genome), seed=0)
            # The env truncates at 25 cycles, well below the 200-step nominal.
            assert 0 < evaluator.env_steps_consumed < nominal_max_steps
        finally:
            wrapper.close()

    def test_counter_accumulates_across_calls(self):
        """The counter is cumulative across evaluate() calls."""
        wrapper = PettingZooWrapper(
            env_name="simple_spread_v3",
            env_config={"max_cycles": 25},
        )
        try:
            obs = wrapper.reset(seed=0)
            agent = next(iter(obs))
            genome = FixedTopologyGenome(
                layer_sizes=[len(obs[agent]), 16, wrapper.action_space(agent).n],
                random_seed=0,
            )
            evaluator = SingleEpisodeEvaluator(env_wrapper=wrapper, max_steps=200)
            evaluator.evaluate(_Individual(genome), seed=0)
            after_one = evaluator.env_steps_consumed
            evaluator.evaluate(_Individual(genome), seed=1)
            assert evaluator.env_steps_consumed > after_one
        finally:
            wrapper.close()



class TestGenerationsBudgetMode:
    def test_wall_clock_member_removed(self):
        """``WALL_CLOCK`` is no longer a constructible enum member."""
        assert not hasattr(BudgetMode, "WALL_CLOCK")
        assert {m.value for m in BudgetMode} == {
            "env_steps",
            "eval_episodes",
            "generations",
        }

    @pytest.mark.parametrize("factory", [EATrainer, CMAESTrainer, PSOTrainer])
    def test_population_trainers_accept_generations(self, factory):
        """EA / CMA / PSO honor a GENERATIONS budget exactly."""
        trainer = factory(population_size=8)
        assert trainer.supports_budget_mode(BudgetMode.GENERATIONS) is True
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=7,
        )
        params = trainer._translate_budget(config)
        assert params["num_generations"] == 7

    def test_agilerl_rejects_generations(self):
        """A GENERATIONS budget is clearly rejected for the step-based path."""
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=50,
        )
        with pytest.raises(ValueError, match="GENERATIONS"):
            translate_budget_to_env_steps(config)

    def test_coevolution_supports_generations(self):
        """Co-evolution honors a GENERATIONS budget independent of side setup."""
        from types import SimpleNamespace

        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        trainer = CoEvolutionTrainer()
        assert trainer.supports_budget_mode(BudgetMode.GENERATIONS) is True
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=9,
        )
        # The GENERATIONS branch returns early, before per-side bookkeeping,
        # so a bare stand-in coev config is sufficient here.
        coev_cfg = SimpleNamespace(sides=[], k_opponents=1, max_steps_per_episode=25)
        assert trainer._translate_budget(config, coev_cfg) == 9

    def test_qdax_generations_and_init_deduction(self):
        """QDax honors GENERATIONS and deducts the init repertoire from ENV_STEPS."""
        from deap_pettingzoo.contrib.qdax.trainer import _budget_to_iterations

        params = {
            "batch_size": 10,
            "episode_length": 20,
            "eval_episodes": 1,
            "init_batch_size": 50,
        }
        gen_cfg = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=12,
        )
        assert _budget_to_iterations(gen_cfg, params) == 12

        # init costs 50*20 = 1000 steps; per iter = 10*20 = 200.
        # A naive (no-deduction) derivation of 2000 steps would give 10 iters;
        # deducting the init leaves 1000 steps -> 5 iters.
        env_cfg = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=2000,
        )
        assert _budget_to_iterations(env_cfg, params) == 5


class TestRealCounterReporting:
    """EA surfaces the real consumed-step counter (serial path)."""

    def test_ea_reports_real_env_steps_below_nominal(self):
        """A serial EA run reports a real env-step count below the nominal one
        on an early-truncating env."""
        pop = 4
        max_steps = 200  # nominal cap well above the env's 25-cycle truncation
        trainer = EATrainer(population_size=pop, strategy="simple")
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=2,
            env_config={"max_cycles": 25},
            algorithm_params={"max_steps_per_episode": max_steps, "seed": 0},
            num_workers=1,
        )
        result = trainer.train(config)
        nominal = result.metrics["env_steps_nominal"]
        real = result.metrics["env_steps_consumed"]
        assert real is not None
        assert 0 < real < nominal


def _early_trunc_config(factory_kwargs=None, **overrides):
    """Build a TrainingConfig on an early-truncating env (25-cycle cap, 200
    nominal max_steps) so real consumed steps fall well below nominal."""
    base = dict(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.GENERATIONS,
        budget_value=2,
        env_config={"max_cycles": 25},
        algorithm_params={"max_steps_per_episode": 200, "seed": 0},
    )
    base.update(overrides)
    return TrainingConfig(**base)


class TestParallelCounterAggregation:
    """Real ``env_steps_consumed`` is aggregated across ``num_workers > 1``."""

    @pytest.mark.parametrize("factory", [EATrainer, CMAESTrainer, PSOTrainer])
    def test_parallel_reports_real_int_below_nominal(self, factory):
        trainer = factory(population_size=4)
        config = _early_trunc_config(num_workers=2)
        result = trainer.train(config)
        nominal = result.metrics["env_steps_nominal"]
        consumed = result.metrics["env_steps_consumed"]
        assert isinstance(consumed, int)
        assert 0 < consumed < nominal


class TestCMAPSOReporting:
    """CMA-ES / PSO surface both env-step keys on the serial path."""

    @pytest.mark.parametrize("factory", [CMAESTrainer, PSOTrainer])
    def test_serial_reports_both_keys_below_nominal(self, factory):
        trainer = factory(population_size=4)
        config = _early_trunc_config(num_workers=1)
        result = trainer.train(config)
        assert "env_steps_nominal" in result.metrics
        assert "env_steps_consumed" in result.metrics
        nominal = result.metrics["env_steps_nominal"]
        consumed = result.metrics["env_steps_consumed"]
        assert isinstance(consumed, int)
        assert 0 < consumed < nominal


class TestConsumedStepBudgeting:
    """Opt-in ``budget_on_consumed`` stops on the real env-step budget."""

    @pytest.mark.parametrize("factory", [EATrainer, CMAESTrainer, PSOTrainer])
    def test_stops_at_consumed_budget(self, factory):
        pop = 4
        budget = 3000
        max_steps = 200
        trainer = factory(population_size=pop, **(
            {"strategy": "simple"} if factory is EATrainer else {}
        ))
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=budget,
            env_config={"max_cycles": 25},
            algorithm_params={"max_steps_per_episode": max_steps, "seed": 0},
            budget_on_consumed=True,
        )
        result = trainer.train(config)
        consumed = result.metrics["env_steps_consumed"]
        # The run stops once the real budget is exhausted and overshoots by at most one generation's budget.
        assert consumed >= budget
        assert consumed < budget + pop * max_steps
        # The realized generation count differs from the nominal derivation,
        # because episodes truncate at 25 cycles (real per-gen << nominal).
        nominal_gens = derive_generations(budget, pop, max_steps)
        assert result.metrics["num_generations"] != nominal_gens

    def test_no_effect_under_generations_mode(self):
        """The flag is inert (no error, nominal behaviour) under GENERATIONS."""
        trainer = EATrainer(population_size=4, strategy="simple")
        config = _early_trunc_config(budget_on_consumed=True)
        result = trainer.train(config)
        # GENERATIONS=2 honored exactly; the flag did nothing.
        assert result.metrics["num_generations"] == 2


class TestConsumedStepBudgetingRejection:
    """Trainers / strategies that can't honor ``budget_on_consumed`` reject it."""

    def _env_steps_config(self):
        return TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=3000,
            env_config={"max_cycles": 25},
            algorithm_params={"max_steps_per_episode": 200, "seed": 0},
            budget_on_consumed=True,
        )

    def test_ea_mu_plus_lambda_rejects(self):
        trainer = EATrainer(population_size=4, strategy="mu_plus_lambda")
        with pytest.raises(ValueError, match="budget_on_consumed"):
            trainer.train(self._env_steps_config())

    def test_coevolution_rejects(self):
        from deap_pettingzoo.baselines.coevolution.sides import Side
        from deap_pettingzoo.baselines.coevolution.trainer import (
            CoEvolutionConfig,
            CoEvolutionTrainer,
        )

        coev_cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side(name="a", role_prefixes=("agent_0",),
                     optimizer="ea", optimizer_config={"population_size": 4}),
                Side(name="b", role_prefixes=("agent_1", "agent_2"),
                     optimizer="ea", optimizer_config={"population_size": 4}),
            ],
            k_opponents=1,
            max_steps_per_episode=200,
        )
        config = self._env_steps_config()
        config.algorithm_params["coevolution"] = coev_cfg
        with pytest.raises(ValueError, match="budget_on_consumed"):
            CoEvolutionTrainer().train(config)

    def test_neat_rejects(self):
        pytest.importorskip("neat")
        from deap_pettingzoo.contrib.neat.trainer import NEATTrainer

        with pytest.raises(ValueError, match="budget_on_consumed"):
            NEATTrainer().train(self._env_steps_config())

    def test_qdax_rejects(self):
        pytest.importorskip("qdax")
        from deap_pettingzoo.contrib.qdax.trainer import MAPElitesTrainer

        with pytest.raises(ValueError, match="budget_on_consumed"):
            MAPElitesTrainer().train(self._env_steps_config())


class TestCoevolutionRealSteps:
    """Co-evolution surfaces real consumed steps below the nominal ceiling."""

    def test_ea_vs_ea_reports_consumed_below_nominal(self):
        from deap_pettingzoo.baselines.coevolution.sides import Side
        from deap_pettingzoo.baselines.coevolution.trainer import (
            CoEvolutionConfig,
            CoEvolutionTrainer,
        )

        coev_cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side(name="a", role_prefixes=("agent_0",),
                     optimizer="ea", optimizer_config={"population_size": 4}),
                Side(name="b", role_prefixes=("agent_1", "agent_2"),
                     optimizer="ea", optimizer_config={"population_size": 4}),
            ],
            k_opponents=1,
            max_steps_per_episode=200,
            seed=0,
        )
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=2,
            env_config={"max_cycles": 25},
            algorithm_params={"coevolution": coev_cfg},
        )
        result = CoEvolutionTrainer().train(config)
        nominal = result.metrics["env_steps_nominal"]
        consumed = result.metrics["env_steps_consumed"]
        assert isinstance(consumed, int)
        assert 0 < consumed < nominal


class TestNeatRealSteps:
    """NEAT surfaces real consumed steps below the nominal ceiling."""

    @pytest.mark.parametrize("num_workers", [1, 2])
    def test_neat_reports_consumed_below_nominal(self, num_workers):
        pytest.importorskip("neat")
        from deap_pettingzoo.contrib.neat.trainer import NEATTrainer

        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=20_000,
            env_config={"max_cycles": 25},
            algorithm_params={
                "pop_size": 6,
                "n_generations": 2,
                "eval_episodes_per_genome": 1,
                "seed": 0,
            },
        )
        result = NEATTrainer(num_workers=num_workers).train(config)
        nominal = result.metrics["env_steps_nominal"]
        consumed = result.metrics["env_steps_consumed"]
        assert isinstance(consumed, int)
        assert 0 < consumed < nominal


class TestSpecFrontDoor:
    """GENERATIONS is reachable through the spec/catalog front door
    for population algorithms and rejected for the step-based ones."""

    def test_generations_validates_for_ea_rejects_for_maddpg(self):
        from deap_pettingzoo.baselines.experiment_spec import (
            BudgetSpec,
            ExperimentSpec,
            validate_spec,
        )

        def _spec(algorithm):
            return ExperimentSpec(
                environment="simple_spread_v3",
                algorithms=[algorithm],
                budget=BudgetSpec(mode="generations", value=10),
            )

        # "generations" is a valid mode and supported by EA -> no budget error.
        ea_report = validate_spec(_spec("ea"))
        assert not any(
            i.code in ("invalid_budget_mode", "budget_mode_unsupported")
            for i in ea_report.errors
        ), [i.code for i in ea_report.errors]

        # MADDPG (step-based) must reject the generation budget.
        maddpg_report = validate_spec(_spec("maddpg"))
        assert any(
            i.code == "budget_mode_unsupported" for i in maddpg_report.errors
        )
