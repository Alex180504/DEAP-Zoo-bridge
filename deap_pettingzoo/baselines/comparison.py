"""
Baseline comparison orchestration.

This module provides high-level functions for comparing multiple
baseline algorithms under matched budget constraints.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    ComparisonConfig,
    TrainingConfig,
    TrainingResult,
    EvalResult,
)
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator


# Algorithm registry
_TRAINER_REGISTRY: Dict[str, type] = {}


def register_trainer(name: str, trainer_class: type) -> None:
    """Register a trainer class for an algorithm.
    
    Args:
        name: Algorithm name (e.g., 'ea', 'maddpg').
        trainer_class: BaselineTrainer subclass.
    """
    _TRAINER_REGISTRY[name] = trainer_class


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
        raise ValueError(
            f"Unknown algorithm '{name}'. "
            f"Available: {list(_TRAINER_REGISTRY.keys())}"
        )
    
    return _TRAINER_REGISTRY[name](**kwargs)


def compare_baselines(config: ComparisonConfig) -> Dict[str, Any]:
    """Compare multiple baseline algorithms.
    
    High-level function that:
    1. Trains each algorithm with matched budget across multiple seeds
    2. Evaluates all trained policies using unified evaluator
    3. Aggregates results and computes statistics
    4. Saves results to disk (if output_dir specified)
    
    Args:
        config: Comparison configuration.
        
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
    # Create evaluator
    evaluator = UnifiedEvaluator(
        max_steps=config.algorithm_configs.get('max_steps_per_episode', 500)
    )
    
    # Results storage
    results = {
        'config': {
            'env_name': config.env_name,
            'budget_mode': config.budget_mode.value,
            'budget_value': config.budget_value,
            'num_seeds': config.num_seeds,
            'eval_episodes': config.eval_episodes,
        }
    }
    
    # Train and evaluate each algorithm
    for algorithm in config.algorithms:
        print(f"\n{'='*70}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"{'='*70}")
        
        # Get algorithm-specific config
        algo_config = config.algorithm_configs.get(algorithm, {})
        
        # Create trainer
        trainer = get_trainer(algorithm, **algo_config.get('trainer_params', {}))
        
        # Check budget mode support
        if not trainer.supports_budget_mode(config.budget_mode):
            print(f"Warning: {algorithm} does not support {config.budget_mode.value}")
            continue
        
        # Train across seeds
        seed_results = []
        for seed_idx in range(config.num_seeds):
            print(f"\n--- Seed {seed_idx + 1}/{config.num_seeds} ---")
            
            # Create training config
            training_config = TrainingConfig(
                env_name=config.env_name,
                budget_mode=config.budget_mode,
                budget_value=config.budget_value,
                num_seeds=1,  # Single seed per training run
                algorithm_params={
                    **algo_config.get('algorithm_params', {}),
                    'seed': seed_idx,
                },
                output_dir=config.output_dir / algorithm / f"seed_{seed_idx}" if config.output_dir else None,
            )
            
            # Train
            print(f"Training {algorithm}...")
            training_result = trainer.train(training_config)
            print(f"Training completed in {training_result.training_time:.2f}s")
            
            # Evaluate
            print(f"Evaluating on {config.eval_episodes} episodes...")
            eval_result = evaluator.evaluate(
                policy=training_result.policy,
                env_name=config.env_name,
                num_episodes=config.eval_episodes,
                seed=seed_idx * 1000,  # Deterministic evaluation seeds
                render=False
            )
            print(f"Evaluation: mean={eval_result.mean_reward:.2f}, "
                  f"std={eval_result.std_reward:.2f}")
            
            # Store results
            seed_results.append({
                'seed': seed_idx,
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
    """Register built-in trainers."""
    try:
        from deap_pettingzoo.baselines.ea_trainer import EATrainer
        register_trainer('ea', EATrainer)
    except ImportError:
        pass
    
    # MADDPG will be registered when implemented
    # try:
    #     from deap_pettingzoo.baselines.maddpg_trainer import MADDPGTrainer
    #     register_trainer('maddpg', MADDPGTrainer)
    # except ImportError:
    #     pass


_register_builtins()
