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
            max_steps_per_episode=200,
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
            max_steps_per_episode=max_steps, seed=0,
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
        max_steps_per_episode=200, seed=0,
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
            max_steps_per_episode=max_steps, seed=0,
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


class TestConsumedStepBudgetingSupport:
    """Per-trainer ``budget_on_consumed`` contract: co-evolution, NEAT, and
    QDax (numpy scoring path) honor it; the DEAP built-in EA strategies
    (fixed-``ngen`` loops) reject it."""

    def _env_steps_config(self):
        return TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=3000,
            env_config={"max_cycles": 25},
            max_steps_per_episode=200, seed=0,
            budget_on_consumed=True,
        )

    def test_ea_mu_plus_lambda_rejects(self):
        trainer = EATrainer(population_size=4, strategy="mu_plus_lambda")
        with pytest.raises(ValueError, match="budget_on_consumed"):
            trainer.train(self._env_steps_config())

    def test_coevolution_consumed_budget_synchronous(self):
        """EA-vs-EA co-evolution with budget_on_consumed=True now succeeds."""
        from deap_pettingzoo.baselines.coevolution.sides import Side
        from deap_pettingzoo.baselines.coevolution.trainer import (
            CoEvolutionConfig,
            CoEvolutionTrainer,
        )

        budget = 500
        coev_cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side(name="a", role_prefixes=("agent_0",),
                     optimizer="ea", optimizer_config={"population_size": 4}),
                Side(name="b", role_prefixes=("agent_1", "agent_2"),
                     optimizer="ea", optimizer_config={"population_size": 4}),
            ],
            k_opponents=1,
            max_steps_per_episode=25,
            seed=42,
        )
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=budget,
            env_config={"max_cycles": 25},
            coevolution_config=coev_cfg,
            budget_on_consumed=True,
        )
        result = CoEvolutionTrainer().train(config)
        consumed = result.metrics["env_steps_consumed"]
        assert consumed >= budget, f"expected consumed >= {budget}, got {consumed}"
        # Overshoot <= one generation's pooled cost (2 sides × pop=4 × k=1 × max_steps=25)
        assert consumed <= budget + 200
        assert isinstance(result.metrics["num_generations"], int)
        assert result.metrics["num_generations"] >= 1

    def test_coevolution_consumed_budget_alternating(self):
        """Alternating schedule also honors budget_on_consumed."""
        from deap_pettingzoo.baselines.coevolution.sides import Side
        from deap_pettingzoo.baselines.coevolution.trainer import (
            CoEvolutionConfig,
            CoEvolutionTrainer,
        )

        budget = 300
        coev_cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            schedule="alternating",
            sides=[
                Side(name="a", role_prefixes=("agent_0",),
                     optimizer="ea", optimizer_config={"population_size": 4}),
                Side(name="b", role_prefixes=("agent_1", "agent_2"),
                     optimizer="ea", optimizer_config={"population_size": 4}),
            ],
            k_opponents=1,
            max_steps_per_episode=25,
            seed=7,
        )
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=budget,
            env_config={"max_cycles": 25},
            coevolution_config=coev_cfg,
            budget_on_consumed=True,
        )
        result = CoEvolutionTrainer().train(config)
        consumed = result.metrics["env_steps_consumed"]
        assert consumed >= budget
        # Overshoot <= one side-step cost (pop=4 × k=1 × max_steps=25 = 100)
        assert consumed <= budget + 100
        assert result.metrics["num_generations"] >= 1

    def test_coevolution_qdax_side_consumed_budget(self):
        """A QDax side participates in the shared consumed-step pool.

        The overshoot bound is the same finish-generation-then-stop bound
        as every other side type: one generation's pooled cost, which for
        the QDax side includes one full batch of episodes.
        """
        pytest.importorskip("qdax")
        import jax.numpy as jnp

        from deap_pettingzoo.baselines.coevolution.sides import Side
        from deap_pettingzoo.baselines.coevolution.trainer import (
            CoEvolutionConfig,
            CoEvolutionTrainer,
        )

        def _descriptor(trajectory):
            return jnp.mean(trajectory["positions"], axis=(0, 1))

        _descriptor.min_bd = (-1.0, -1.0)
        _descriptor.max_bd = (1.0, 1.0)
        _descriptor.output_shape = (2,)

        batch = 4
        pop = 4
        max_steps = 25
        budget = 1_000
        coev_cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side(
                    name="pursuers",
                    role_prefixes=("adversary",),
                    optimizer="qdax_map_elites",
                    optimizer_config={
                        "batch_size": batch,
                        "grid_shape": (2, 2),
                        "episode_length": max_steps,
                        "descriptor_fn": _descriptor,
                    },
                ),
                Side(
                    name="evaders",
                    role_prefixes=("agent",),
                    optimizer="ea",
                    optimizer_config={"population_size": pop},
                ),
            ],
            k_opponents=1,
            max_steps_per_episode=max_steps,
            seed=11,
        )
        config = TrainingConfig(
            env_name="simple_tag_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=budget,
            env_config={"max_cycles": max_steps},
            coevolution_config=coev_cfg,
            budget_on_consumed=True,
        )
        result = CoEvolutionTrainer().train(config)
        consumed = result.metrics["env_steps_consumed"]
        assert consumed >= budget
        # Overshoot <= one generation's pooled cost: the QDax batch
        # (batch x k x max_steps) plus the EA side (pop x k x max_steps).
        assert consumed <= budget + (batch + pop) * max_steps
        assert result.metrics["num_generations"] >= 1

    def test_neat_honors_consumed_budget(self):
        """NEAT steps n=1 generations until the real budget is exhausted."""
        pytest.importorskip("neat")
        from deap_pettingzoo.contrib.neat.trainer import NEATTrainer

        pop = 6
        budget = 2_000
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=budget,
            env_config={"max_cycles": 25},
            max_steps_per_episode=200, seed=0,
            budget_on_consumed=True,
            algorithm_params={"pop_size": pop, "eval_episodes_per_genome": 1},
        )
        result = NEATTrainer().train(config)
        consumed = result.metrics["env_steps_consumed"]
        assert consumed >= budget
        # Overshoot at most one generation: pop genomes x 25 real cycles.
        assert consumed <= budget + pop * 25
        # Realized generations exceed the nominal derivation (2000 / (6*200)
        # rounds to 2) because episodes truncate at 25 of 200 nominal steps.
        nominal_gens = derive_generations(budget, pop, 200)
        assert result.metrics["final_generation"] > nominal_gens

    def test_qdax_honors_consumed_budget_numpy_path(self):
        """MAP-Elites iterates until the scoring counter hits the budget."""
        pytest.importorskip("qdax")
        import jax.numpy as jnp

        from deap_pettingzoo.contrib.qdax.trainer import MAPElitesTrainer

        def _descriptor(trajectory):
            return jnp.mean(trajectory["positions"], axis=(0, 1))

        _descriptor.min_bd = (-1.0, -1.0)
        _descriptor.max_bd = (1.0, 1.0)
        _descriptor.output_shape = (2,)

        batch = 8
        budget = 3_000
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=budget,
            env_config={"max_cycles": 25},
            max_steps_per_episode=200,  # nominal; env truncates at 25
            seed=0,
            budget_on_consumed=True,
            algorithm_params={
                "descriptor_fn": _descriptor,
                "batch_size": batch,
                "init_batch_size": batch,
                "grid_shape": (4, 4),
                "save_repertoire": False,
                "verbose": False,
                "num_workers": 1,
            },
        )
        result = MAPElitesTrainer().train(config)
        consumed = result.metrics["env_steps_consumed"]
        assert isinstance(consumed, int)
        assert consumed >= budget
        # Overshoot at most one iteration: batch genotypes x 25 real cycles.
        assert consumed <= budget + batch * 25
        # Realized iterations exceed the nominal derivation (which prices
        # each episode at the 200-step nominal cap).
        assert result.metrics["num_iterations"] > 1


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
            coevolution_config=coev_cfg,
        )
        result = CoEvolutionTrainer().train(config)
        nominal = result.metrics["env_steps_nominal"]
        consumed = result.metrics["env_steps_consumed"]
        assert isinstance(consumed, int)
        assert 0 < consumed < nominal


class TestCoevolutionConsumedBudgetMixed:
    """Co-evolution shared-pool consumed budget with heterogeneous side types."""

    def test_ea_vs_cma_stops_at_budget(self):
        """EA-vs-CMA co-evolution stops when the shared pool reaches budget."""
        from deap_pettingzoo.baselines.coevolution.sides import Side
        from deap_pettingzoo.baselines.coevolution.trainer import (
            CoEvolutionConfig,
            CoEvolutionTrainer,
        )

        budget = 400
        coev_cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side(name="a", role_prefixes=("agent_0",),
                     optimizer="ea", optimizer_config={"population_size": 4}),
                Side(name="b", role_prefixes=("agent_1", "agent_2"),
                     optimizer="cma", optimizer_config={"population_size": 4}),
            ],
            k_opponents=1,
            max_steps_per_episode=25,
            seed=0,
        )
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=budget,
            env_config={"max_cycles": 25},
            coevolution_config=coev_cfg,
            budget_on_consumed=True,
        )
        result = CoEvolutionTrainer().train(config)
        consumed = result.metrics["env_steps_consumed"]
        assert consumed >= budget
        assert result.metrics["num_generations"] >= 1


class TestRLHPOReporting:
    """rl_hpo side reports real per-generation env steps (not triangular)."""

    def test_env_steps_is_real_not_triangular(self):
        pytest.importorskip("agilerl")
        from deap_pettingzoo.baselines.coevolution.sides import Side
        from deap_pettingzoo.baselines.coevolution.trainer import (
            CoEvolutionConfig,
            CoEvolutionTrainer,
        )

        rl_hpo_env_steps: list = []

        def _callback(gen, states):
            st = states.get("b")
            if st is not None and st.history:
                rl_hpo_env_steps.append(int(st.history[-1].env_steps))

        evo_every = 30
        coev_cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side(name="a", role_prefixes=("agent_0",),
                     optimizer="ea", optimizer_config={"population_size": 4}),
                Side(name="b", role_prefixes=("agent_1", "agent_2"),
                     optimizer="rl_hpo",
                     optimizer_config={
                         "algorithm": "maddpg_hpo",
                         "population_size": 2,
                         "evo_every_n_steps": evo_every,
                     }),
            ],
            k_opponents=1,
            max_steps_per_episode=25,
            seed=0,
            generation_callback=_callback,
        )
        config = TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.GENERATIONS,
            budget_value=2,
            env_config={"max_cycles": 25},
            coevolution_config=coev_cfg,
        )
        CoEvolutionTrainer().train(config)

        assert len(rl_hpo_env_steps) >= 2, "callback must capture at least 2 gens"
        g1, g2 = rl_hpo_env_steps[0], rl_hpo_env_steps[1]
        # Each generation should reflect real ticks (not triangular: gen2 ≠ 2*gen1).
        assert g2 != 2 * g1, (
            f"env_steps appears triangular: gen1={g1}, gen2={g2} — "
            "bug fix may have regressed"
        )
        # Real steps must be greater than evo_every alone (includes eval steps).
        assert g1 > evo_every, f"gen1 env_steps={g1} should exceed evo_every={evo_every}"


class TestPerSideBudgetSplit:
    """``CoEvolutionConfig.budget_split='per_side'`` — equal consumed-step
    slices per evolving side, with ``on_side_exhausted`` deciding whether an
    exhausted side freezes (run continues) or cuts the whole run off."""

    @staticmethod
    def _coev_cfg(schedule="synchronous", **kwargs):
        from deap_pettingzoo.baselines.coevolution.sides import Side
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionConfig

        # Asymmetric populations: side "b" evaluates twice as many episodes
        # per generation as side "a", so it drains its slice ~2x faster.
        return CoEvolutionConfig(
            mode="competitive_2pop",
            schedule=schedule,
            sides=[
                Side(name="a", role_prefixes=("agent_0",),
                     optimizer="ea", optimizer_config={"population_size": 4}),
                Side(name="b", role_prefixes=("agent_1", "agent_2"),
                     optimizer="ea", optimizer_config={"population_size": 8}),
            ],
            k_opponents=1,
            max_steps_per_episode=25,
            seed=11,
            **kwargs,
        )

    @staticmethod
    def _train_cfg(coev_cfg, budget, budget_on_consumed=True):
        return TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=budget,
            env_config={"max_cycles": 25},
            coevolution_config=coev_cfg,
            budget_on_consumed=budget_on_consumed,
        )

    def test_invalid_budget_split_rejected(self):
        with pytest.raises(ValueError, match="budget_split"):
            self._coev_cfg(budget_split="equal")

    def test_invalid_exhausted_policy_rejected(self):
        with pytest.raises(ValueError, match="on_side_exhausted"):
            self._coev_cfg(budget_split="per_side", on_side_exhausted="stop")

    def test_per_side_requires_consumed_budget(self):
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        coev_cfg = self._coev_cfg(budget_split="per_side")
        config = self._train_cfg(coev_cfg, budget=1200, budget_on_consumed=False)
        with pytest.raises(ValueError, match="budget_on_consumed"):
            CoEvolutionTrainer().train(config)

    def test_freeze_equalizes_per_side_consumption(self):
        """freeze: every side spends its full slice; the fast-draining side
        realizes fewer generations but the same env steps."""
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        budget = 1200  # share = 600/side; a ~100/gen, b ~200/gen
        coev_cfg = self._coev_cfg(budget_split="per_side", on_side_exhausted="freeze")
        result = CoEvolutionTrainer().train(self._train_cfg(coev_cfg, budget))

        per_side = result.metrics["env_steps_consumed_per_side"]
        share = budget // 2
        # Both sides spend their slice; overshoot <= one own generation.
        assert per_side["a"] >= share
        assert per_side["a"] <= share + 4 * 25
        assert per_side["b"] >= share
        assert per_side["b"] <= share + 8 * 25
        # The fast-draining side froze earlier in generation terms.
        hist = result.metrics["per_side_history"]
        assert len(hist["b"]) < len(hist["a"])

    def test_cutoff_stops_at_first_exhausted_side(self):
        """cutoff: the run ends when the fast-draining side exhausts; the
        slow side is left under its slice and the pool is under-spent."""
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        budget = 1200
        coev_cfg = self._coev_cfg(budget_split="per_side", on_side_exhausted="cutoff")
        result = CoEvolutionTrainer().train(self._train_cfg(coev_cfg, budget))

        per_side = result.metrics["env_steps_consumed_per_side"]
        share = budget // 2
        assert per_side["b"] >= share, "fast side must have exhausted its slice"
        assert per_side["a"] < share, "slow side must be cut off under its slice"
        assert result.metrics["env_steps_consumed"] < budget

    def test_freeze_alternating_schedule(self):
        """The alternating schedule honors per-side slices too."""
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        budget = 1200
        coev_cfg = self._coev_cfg(
            schedule="alternating",
            budget_split="per_side",
            on_side_exhausted="freeze",
        )
        result = CoEvolutionTrainer().train(self._train_cfg(coev_cfg, budget))

        per_side = result.metrics["env_steps_consumed_per_side"]
        share = budget // 2
        assert per_side["a"] >= share
        assert per_side["b"] >= share
        assert per_side["a"] <= share + 4 * 25
        assert per_side["b"] <= share + 8 * 25

    def test_shared_default_unchanged(self):
        """Default budget_split='shared' keeps the pooled semantics."""
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        budget = 600
        coev_cfg = self._coev_cfg()  # defaults: shared
        result = CoEvolutionTrainer().train(self._train_cfg(coev_cfg, budget))
        consumed = result.metrics["env_steps_consumed"]
        assert consumed >= budget
        # Overshoot <= one pooled generation (a 100 + b 200).
        assert consumed <= budget + 300


class TestEqualStepPacing:
    """``CoEvolutionConfig.side_pacing='equal_steps'`` — sides are handed
    generations by real consumed env steps rather than one each per iteration,
    so the expensive side no longer drains its slice early and freezes."""

    @staticmethod
    def _coev_cfg(schedule="synchronous", **kwargs):
        from deap_pettingzoo.baselines.coevolution.sides import Side
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionConfig

        # Side "b" evaluates 4x the episodes of side "a" per generation, so
        # under lockstep it spends its slice 4x faster.
        return CoEvolutionConfig(
            mode="competitive_2pop",
            schedule=schedule,
            sides=[
                Side(name="a", role_prefixes=("agent_0",),
                     optimizer="ea", optimizer_config={"population_size": 2}),
                Side(name="b", role_prefixes=("agent_1", "agent_2"),
                     optimizer="ea", optimizer_config={"population_size": 8}),
            ],
            k_opponents=1,
            max_steps_per_episode=25,
            seed=11,
            **kwargs,
        )

    @staticmethod
    def _train_cfg(coev_cfg, budget, budget_on_consumed=True):
        return TrainingConfig(
            env_name="simple_spread_v3",
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=budget,
            env_config={"max_cycles": 25},
            coevolution_config=coev_cfg,
            budget_on_consumed=budget_on_consumed,
        )

    def _active_fractions(self, budget=2400, **kwargs):
        """Run a pairing and return, per side, how far into the run it last stepped.

        1.0 means the side was still evolving at the final iteration; the
        campaign's frozen sides sat at a median 0.66.
        """
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        timeline: list = []
        coev_cfg = self._coev_cfg(
            generation_callback=lambda gen, states: timeline.append(
                (gen, {n: len(s.history) for n, s in states.items()})
            ),
            **kwargs,
        )
        result = CoEvolutionTrainer().train(self._train_cfg(coev_cfg, budget))
        total = timeline[-1][0]
        fractions = {}
        for name in timeline[-1][1]:
            final = timeline[-1][1][name]
            first_at_final = min(g for g, counts in timeline if counts[name] == final)
            fractions[name] = first_at_final / total
        return fractions, result

    def test_invalid_side_pacing_rejected(self):
        with pytest.raises(ValueError, match="side_pacing"):
            self._coev_cfg(side_pacing="fair")

    def test_equal_steps_requires_consumed_budget(self):
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        coev_cfg = self._coev_cfg(side_pacing="equal_steps")
        config = self._train_cfg(coev_cfg, budget=1200, budget_on_consumed=False)
        with pytest.raises(ValueError, match="budget_on_consumed"):
            CoEvolutionTrainer().train(config)

    def test_lockstep_freezes_the_expensive_side_early(self):
        """Baseline: the behaviour equal_steps exists to remove.

        Side "b" costs 4x side "a" per generation, so under lockstep it spends
        its equal slice in the first quarter of the run and the remaining
        three quarters are not co-evolution at all.
        """
        fractions, _ = self._active_fractions(
            budget_split="per_side", on_side_exhausted="freeze",
        )
        assert fractions["a"] == pytest.approx(1.0)
        assert fractions["b"] < 0.4

    def test_equal_steps_keeps_both_sides_running(self):
        """Neither side stops materially before the other."""
        fractions, result = self._active_fractions(
            budget_split="per_side",
            on_side_exhausted="freeze",
            side_pacing="equal_steps",
        )
        assert fractions["a"] == pytest.approx(1.0)
        assert fractions["b"] >= 0.8, (
            f"expensive side froze at {fractions['b']:.2f} of the run"
        )

        # Equal slices are still spent in full ...
        per_side = result.metrics["env_steps_consumed_per_side"]
        share = 2400 // 2
        assert per_side["a"] >= share
        assert per_side["b"] >= share
        # ... the cheap side just takes more generations to spend its own.
        hist = result.metrics["per_side_history"]
        assert len(hist["a"]) > len(hist["b"])

    def test_equal_costs_still_step_together(self):
        """Pacing must not serialise sides that cost the same per generation."""
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
            max_steps_per_episode=25,
            seed=11,
            budget_split="per_side",
            on_side_exhausted="freeze",
            side_pacing="equal_steps",
        )
        result = CoEvolutionTrainer().train(self._train_cfg(coev_cfg, 2400))

        hist = result.metrics["per_side_history"]
        gens = result.metrics["num_generations"]
        # Every side stepped in every iteration — identical to lockstep.
        assert len(hist["a"]) == gens
        assert len(hist["b"]) == gens

    def test_pacing_never_stalls_when_one_side_is_far_ahead(self):
        """The laggard always steps, so the loop cannot spin without progress."""
        from deap_pettingzoo.baselines.coevolution.sides import GenerationStats
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        def _stats(name, steps):
            return GenerationStats(
                generation=1, side_name=name, mean=0.0, std=0.0,
                min=0.0, max=0.0, best_fitness=0.0, env_steps=steps,
            )

        # No history yet: nothing to pace against, everybody steps.
        assert CoEvolutionTrainer._paced_skip({"a": [], "b": []}, ["a", "b"]) == set()

        # "b" is three of its own generations ahead — held back, "a" is not.
        history = {
            "a": [_stats("a", 1000)],
            "b": [_stats("b", 1000)] * 3,
        }
        assert CoEvolutionTrainer._paced_skip(history, ["a", "b"]) == {"b"}

        # Level sides both step, preserving simultaneity when costs match.
        history = {"a": [_stats("a", 1000)], "b": [_stats("b", 1000)]}
        assert CoEvolutionTrainer._paced_skip(history, ["a", "b"]) == set()

    def test_equal_steps_alternating_schedule(self):
        """The alternating rotation honours pacing too."""
        from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer

        budget = 2400
        coev_cfg = self._coev_cfg(
            schedule="alternating",
            budget_split="per_side",
            on_side_exhausted="freeze",
            side_pacing="equal_steps",
        )
        result = CoEvolutionTrainer().train(self._train_cfg(coev_cfg, budget))

        per_side = result.metrics["env_steps_consumed_per_side"]
        share = budget // 2
        assert per_side["a"] >= share
        assert per_side["b"] >= share
        hist = result.metrics["per_side_history"]
        assert len(hist["a"]) > len(hist["b"])


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
