"""
CMA-ES trainer implementation using :class:`deap.cma.Strategy`.

This module mirrors :mod:`deap_pettingzoo.baselines.ea_trainer` but drives
an (μ/μ_w, λ)-CMA-ES search over the flat weight vector of a
:class:`FixedTopologyGenome`. The resulting best-found genome is wrapped
as a :class:`PolicyProtocol` via :class:`GenomePolicyWrapper` so that it
is drop-in compatible with :class:`UnifiedEvaluator` and
:func:`compare_baselines`.
"""

import time
from typing import Any, Dict, Optional

import numpy as np
from deap import tools

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.baselines.ea_trainer import GenomePolicyWrapper
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.integration.cma_integration import (
    make_cma_ind_init,
    make_cma_stats,
    register_cma_toolbox,
    setup_cma_one_plus_lambda_strategy,
    setup_cma_strategy,
)


class CMAESTrainer(BaselineTrainer):
    """CMA-ES trainer built on top of :class:`deap.cma.Strategy`.

    Uses the same :class:`FixedTopologyGenome` representation as
    :class:`EATrainer`, so the two trainers can be compared head-to-head
    under identical budget/environment configurations via
    :func:`compare_baselines`.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        population_size: int = 50,
        sigma0: float = 0.3,
        activation: str = "tanh",
        weight_range: tuple = (-1.0, 1.0),
        variant: str = "mu_mu_lambda",
    ):
        """Initialise the CMA-ES trainer.

        Args:
            hidden_size: Hidden layer size for the evolved MLP.
            population_size: ``lambda_`` — number of samples per
                generation. Defaulted to ``50`` to match
                :class:`EATrainer` rather than DEAP's dimension-dependent
                default, so head-to-head comparisons evaluate an equal
                number of rollouts per generation.
            sigma0: Initial step size of the CMA search distribution.
            activation: Activation function for the MLP.
            weight_range: Probed bounds used when materialising genomes
                from sampled vectors (CMA itself is unbounded — this
                only affects the genome's bookkeeping).
            variant: CMA-ES flavour. ``"mu_mu_lambda"`` (default) uses
                :class:`deap.cma.Strategy` — the standard (μ/μ_w, λ)
                rank-μ update. ``"one_plus_lambda"`` uses
                :class:`deap.cma.StrategyOnePlusLambda` — a single-parent
                elitist variant that is typically more sample-efficient
                on unimodal objectives.
        """
        self.hidden_size = hidden_size
        self.population_size = population_size
        self.sigma0 = sigma0
        self.activation = activation
        self.weight_range = weight_range
        if variant not in {"mu_mu_lambda", "one_plus_lambda"}:
            raise ValueError(
                f"Unknown CMA-ES variant {variant!r}. "
                "Expected 'mu_mu_lambda' or 'one_plus_lambda'."
            )
        self.variant = variant

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train CMA-ES with the given configuration.

        Args:
            config: Unified training configuration.

        Returns:
            :class:`TrainingResult` whose ``policy`` wraps the best-found
            genome as a :class:`PolicyProtocol`.
        """
        start_time = time.time()

        params = self._translate_budget(config)
        num_generations = params["num_generations"]
        max_steps_per_episode = params["max_steps_per_episode"]

        # Environment + evaluator (homogeneous agents share one genome).
        env = PettingZooWrapper(config.env_name, env_config=config.env_config or None)
        sample_obs = env.reset(seed=0)
        agent_id = env.possible_agents[0]

        input_size = len(sample_obs[agent_id])
        action_space = env.action_space(agent_id)
        if hasattr(action_space, "n"):  # Discrete
            output_size = action_space.n
        else:  # Box
            output_size = action_space.shape[0]

        evaluator = SingleEpisodeEvaluator(
            env_wrapper=env,
            max_steps=max_steps_per_episode,
            aggregation_method="sum",
        )

        # Seed NumPy's global RNG — deap.cma.Strategy samples via
        # numpy.random.standard_normal, so this is the only control knob.
        seed = config.algorithm_params.get("seed", None)
        if seed is not None:
            np.random.seed(int(seed))

        genome_config = {
            "layer_sizes": [input_size, self.hidden_size, output_size],
            "activation": self.activation,
            "weight_range": self.weight_range,
        }

        strategy, dimension = self._build_strategy(genome_config)
        ind_init = make_cma_ind_init(genome_config)
        toolbox = register_cma_toolbox(strategy, ind_init, evaluator)

        stats = make_cma_stats()
        hof = tools.HallOfFame(1)

        fitness_history = []

        for gen in range(num_generations):
            population = toolbox.generate()
            for ind, fit in zip(population, map(toolbox.evaluate, population)):
                ind.fitness.values = fit

            toolbox.update(population)
            hof.update(population)

            record = stats.compile(population)
            fitness_history.append(
                {
                    "generation": gen,
                    "mean": record["avg"],
                    "std": record["std"],
                    "min": record["min"],
                    "max": record["max"],
                }
            )

        env.close()
        training_time = time.time() - start_time

        best_individual = hof[0]
        best_genome = best_individual.genome
        policy = GenomePolicyWrapper(best_genome, config.env_name)

        return TrainingResult(
            algorithm="cma-es",
            policy=policy,
            metrics={
                "fitness_history": fitness_history,
                "final_fitness": best_individual.fitness.values[0],
                "num_generations": num_generations,
                "population_size": self.population_size,
                "dimension": dimension,
            },
            metadata={
                "genome_config": genome_config,
                "cma_config": {
                    "hidden_size": self.hidden_size,
                    "population_size": self.population_size,
                    "sigma0": self.sigma0,
                    "activation": self.activation,
                    "weight_range": list(self.weight_range),
                    "variant": self.variant,
                },
            },
            training_time=training_time,
        )

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        """CMA-ES supports env-step and eval-episode budgets."""
        return mode in {BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES}

    def get_capabilities(self) -> Dict[str, Any]:
        """Report CMA-ES capability flags (matches EA's)."""
        return {
            "discrete_action": True,
            "continuous_action": True,
            "needs_global_state": False,
            "on_policy": False,
        }

    def _translate_budget(self, config: TrainingConfig) -> Dict[str, Any]:
        """Translate a budget into ``num_generations`` and episode length.

        Matches :meth:`EATrainer._translate_budget` so identical budget
        configurations yield the same number of generations for both
        trainers.
        """
        max_steps_per_episode = config.algorithm_params.get(
            "max_steps_per_episode", 200
        )

        if config.budget_mode == BudgetMode.ENV_STEPS:
            total_env_steps = config.budget_value
            num_generations = total_env_steps // (
                self.population_size * max_steps_per_episode
            )
            num_generations = max(1, num_generations)
        elif config.budget_mode == BudgetMode.EVAL_EPISODES:
            total_episodes = config.budget_value
            num_generations = total_episodes // self.population_size
            num_generations = max(1, num_generations)
        else:
            raise ValueError(
                f"CMAESTrainer does not support budget mode {config.budget_mode}"
            )

        return {
            "num_generations": int(num_generations),
            "max_steps_per_episode": int(max_steps_per_episode),
        }
    def _build_strategy(self, genome_config: Dict[str, Any]):
        """Dispatch to the CMA-ES flavour selected by ``self.variant``."""
        if self.variant == "mu_mu_lambda":
            return setup_cma_strategy(
                genome_config=genome_config,
                sigma0=self.sigma0,
                population_size=self.population_size,
            )
        if self.variant == "one_plus_lambda":
            return setup_cma_one_plus_lambda_strategy(
                genome_config=genome_config,
                sigma0=self.sigma0,
                population_size=self.population_size,
            )
        raise ValueError(  # pragma: no cover — guarded in __init__.
            f"Unknown CMA-ES variant {self.variant!r}"
        )