"""
Baseline comparison orchestration.

This module provides high-level functions for comparing multiple
baseline algorithms under matched budget constraints.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    ComparisonConfig,
)
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator

logger = logging.getLogger(__name__)


# Algorithm registry
_TRAINER_REGISTRY: Dict[str, type] = {}


def register_trainer(name: str, trainer_class: type) -> None:
    """Register a trainer class for an algorithm.
    
    Args:
        name: Algorithm name (e.g., 'ea', 'maddpg').
        trainer_class: BaselineTrainer subclass.
    """
    _TRAINER_REGISTRY[name] = trainer_class


def _sync_from_catalog() -> None:
    """Register any catalog trainer bindings not yet in the trainer registry.

    Makes the trainer registry a self-healing cache over the catalog: a
    third-party :func:`deap_pettingzoo.catalog.register_algorithm` becomes
    resolvable through :func:`get_trainer` on the next miss, with no import-order
    assumptions and no second registration call. Importing a trainer module does
    not pull its optional extra (heavy imports are deferred to ``train()``), so
    a missing extra simply skips that binding.
    """
    import importlib

    from deap_pettingzoo.catalog.algorithms import iter_trainer_bindings

    for name, module_path, class_name in iter_trainer_bindings():
        if name in _TRAINER_REGISTRY:
            continue
        try:
            module = importlib.import_module(module_path)
            # Walk a possibly-nested ``"Outer.Inner"`` qualname (a plain
            # getattr would AttributeError on it).
            trainer_cls = module
            for part in class_name.split("."):
                trainer_cls = getattr(trainer_cls, part)
            register_trainer(name, trainer_cls)
        except (ImportError, AttributeError):
            # A missing optional extra (ImportError) or a malformed binding
            # (AttributeError) skips only this entry — one bad registration
            # must never poison resolution of the other algorithms.
            continue


def get_trainer(name: str, **kwargs) -> BaselineTrainer:
    """Get a trainer instance by name.

    Args:
        name: Algorithm name.
        **kwargs: Trainer-specific initialization arguments.

    Returns:
        Trainer instance.

    Raises:
        ValueError: If algorithm not registered.
    """
    if name not in _TRAINER_REGISTRY:
        # Self-heal: a third-party algorithm registered into the catalog after
        # this module imported is picked up here before we give up.
        _sync_from_catalog()
    if name not in _TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Available: {list(_TRAINER_REGISTRY.keys())}"
        )

    return _TRAINER_REGISTRY[name](**kwargs)


#: Base offset for the held-out eval RNG. Episode seeds are derived from
#: ``_EVAL_SEED_BASE + repeat_index`` so the *i*-th repeat of every algorithm is
#: scored on the same paired set of evaluation episodes.
_EVAL_SEED_BASE = 90_000_000


def _flatten_algo_hyperparams(algo_config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a legacy ``algorithm_configs`` entry into one hyperparameter dict.

    The legacy nested shape (``{"trainer_params": {...}, "algorithm_params":
    {...}}``) is collapsed into a single flat mapping; :func:`build_configs`
    then re-routes each key to ``trainer_params`` / ``algorithm_params`` by its
    catalog ``ParamTarget``, so the caller no longer has to know the split.
    """
    flat: Dict[str, Any] = {}
    for namespace in ("trainer_params", "algorithm_params"):
        block = algo_config.get(namespace)
        if isinstance(block, dict):
            flat.update(block)
    return flat


def compare_baselines(
    config: ComparisonConfig,
    validate: bool = True,
) -> Dict[str, Any]:
    """Compare multiple baseline algorithms.

    High-level function that:

    1. Builds a unified :class:`BuiltPlan` via :func:`build_configs` so this
       legacy entry point inherits the same ``env_config`` threading, SuperSuit
       auto-padding, fitness/recording/parallelism handling, and per-repeat
       seeds the subprocess runner uses.
    2. Trains each algorithm with the matched budget across ``num_seeds``.
    3. Evaluates every trained policy with one shared :class:`UnifiedEvaluator`
       on the **same** ``env_config`` used for training, with eval episodes
       paired by repeat index.
    4. Aggregates per-algorithm statistics and saves results (if ``output_dir``).

    Args:
        config: Comparison configuration.
        validate: If True (default), run compatibility checks before training
            and skip algorithms that are incompatible with the environment.

    Returns:
        Dictionary with comparison results:
        {
            'ea': {
                'seeds': [
                    {
                        'seed': 0,
                        'training_time': 12.3,
                        'training_metrics': {...},
                        'eval_result': EvalResult(...),
                    },
                    ...
                ],
                'mean_reward': 45.6,
                'std_reward': 3.2,
                'best_seed': 1,
            },
            'maddpg': {...},
            'config': {...},
        }
    """
    from collections import OrderedDict

    from deap_pettingzoo.baselines.experiment_spec import (
        BudgetSpec,
        ExperimentSpec,
        build_configs,
    )

    # Shared evaluator
    evaluator = UnifiedEvaluator(max_steps=config.max_steps_per_episode)

    results: Dict[str, Any] = {
        'config': {
            'env_name': config.env_name,
            'env_config': dict(config.env_config),
            'budget_mode': config.budget_mode.value,
            'budget_value': config.budget_value,
            'num_seeds': config.num_seeds,
            'eval_episodes': config.eval_episodes,
        }
    }

    # Build the unified plan once; group the resulting sub-runs by (canonical) algorithm, preserving the requested order.
    spec = ExperimentSpec(
        environment=config.env_name,
        algorithms=list(config.algorithms),
        env_config=dict(config.env_config),
        hyperparameters={
            algo: _flatten_algo_hyperparams(config.algorithm_configs.get(algo, {}))
            for algo in config.algorithms
        },
        budget=BudgetSpec(
            mode=config.budget_mode.value,
            value=config.budget_value,
            num_seeds=config.num_seeds,
        ),
    )
    plan = build_configs(spec, output_root=config.output_dir)

    subs_by_algo: "OrderedDict[str, list]" = OrderedDict()
    for sub in plan.sub_runs:
        subs_by_algo.setdefault(sub.algorithm, []).append(sub)

    for algorithm, subs in subs_by_algo.items():
        print(f"\n{'='*70}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"{'='*70}")

        # One trainer instance for the per-algorithm capability/compat checks.
        trainer = get_trainer(algorithm, **subs[0].trainer_params)

        if not trainer.supports_budget_mode(config.budget_mode):
            print(f"Warning: {algorithm} does not support {config.budget_mode.value}")
            logger.warning(
                "%s does not support budget mode %s; skipping",
                algorithm, config.budget_mode.value,
            )
            continue

        if validate:
            from deap_pettingzoo.baselines.compatibility import validate_compatibility
            compat = validate_compatibility(trainer, config.env_name, algorithm)
            if not compat.compatible:
                print(f"Warning: {algorithm} is incompatible with {config.env_name}:")
                for issue in compat.issues:
                    print(f"  - {issue}")
                logger.warning(
                    "%s incompatible with %s: %s",
                    algorithm, config.env_name, "; ".join(compat.issues),
                )
                results[algorithm] = {'skipped': True, 'issues': compat.issues}
                continue
            if compat.warnings:
                for w in compat.warnings:
                    print(f"[{algorithm}] Warning: {w}")
                    logger.warning("[%s] %s", algorithm, w)

        # Train + evaluate each repeat. ``sub.config`` already carries the
        # threaded env_config, paired training seed, and output_dir.
        seed_results = []
        for sub in subs:
            print(f"\n--- Seed {sub.seed + 1}/{config.num_seeds} ---")

            run_trainer = get_trainer(algorithm, **sub.trainer_params)
            print(f"Training {algorithm}...")
            training_result = run_trainer.train(sub.config)
            print(f"Training completed in {training_result.training_time:.2f}s")

            print(f"Evaluating on {config.eval_episodes} episodes...")
            # Multi-agent RL wrappers expose ``policy_map()`` for an
            # order-independent per-agent dispatch (robust to a shrinking
            # active-agent set); single-policy backends use the broadcast path.
            _policy = training_result.policy
            _policy_map = _policy.policy_map() if hasattr(_policy, "policy_map") else None
            _eval_kwargs = dict(
                env_name=sub.config.env_name,
                num_episodes=config.eval_episodes,
                seed=_EVAL_SEED_BASE + int(sub.seed),  # paired by repeat index
                env_config=sub.config.env_config,
                render=False,
            )
            if _policy_map:
                eval_result = evaluator.evaluate(None, policy_map=_policy_map, **_eval_kwargs)
            else:
                eval_result = evaluator.evaluate(_policy, **_eval_kwargs)
            print(f"Evaluation: mean={eval_result.mean_reward:.2f}, "
                  f"std={eval_result.std_reward:.2f}")

            seed_results.append({
                'seed': sub.seed,
                'training_time': training_result.training_time,
                'training_metrics': training_result.metrics,
                'eval_result': eval_result.to_dict(),
            })

        # Aggregate across seeds
        if seed_results:
            mean_rewards = [sr['eval_result']['mean_reward'] for sr in seed_results]
            best_seed_idx = int(np.argmax(mean_rewards))

            results[algorithm] = {
                'seeds': seed_results,
                'mean_reward': float(np.mean(mean_rewards)),
                'std_reward': float(np.std(mean_rewards)),
                'min_reward': float(np.min(mean_rewards)),
                'max_reward': float(np.max(mean_rewards)),
                'best_seed': best_seed_idx,
                'eval_metric_provenance': 'held_out_unified',
                'capabilities': trainer.get_capabilities(),
            }

            print(f"\n{algorithm.upper()} Summary:")
            print(f"  Mean reward: {results[algorithm]['mean_reward']:.2f} "
                  f"± {results[algorithm]['std_reward']:.2f}")
            print(f"  Best seed: {best_seed_idx} "
                  f"(reward: {mean_rewards[best_seed_idx]:.2f})")

    # Save results
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        results_path = config.output_dir / 'comparison_results.json'

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")

    # Cleanup
    evaluator.close()

    return results


# Auto-register built-in trainers
def _register_builtins():
    """Register built-in trainers, sourcing the mapping from the catalog.

    The algorithm -> trainer-class bindings come from
    :func:`deap_pettingzoo.catalog.algorithms.iter_trainer_bindings`, so the
    trainer registry and the unified catalog can never drift apart (this is how
    ``pso`` is registered as a standalone algorithm in the UI catalog).

    Importing a trainer module does NOT require its optional extra (the heavy
    ``import agilerl`` / ``import torch`` calls are deferred until ``train()``),
    so every binding registers; only calling ``train()`` without the matching
    extra raises ImportError.
    """
    _sync_from_catalog()

    # Co-evolution is a meta-trainer, not a standalone catalog algorithm, so it
    # is registered explicitly rather than via the catalog binding list.
    try:
        from deap_pettingzoo.baselines.coevolution import CoEvolutionTrainer
        register_trainer('coevolution', CoEvolutionTrainer)
    except ImportError:
        pass


_register_builtins()
