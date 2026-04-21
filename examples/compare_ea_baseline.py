"""
Example: Compare EA baseline on simple_spread_v3

This script demonstrates the baseline comparison API by training
and evaluating an EA algorithm. It serves as a minimal working
example before MADDPG integration is complete.
"""

from pathlib import Path
from deap_pettingzoo.baselines import (
    ComparisonConfig,
    BudgetMode,
    compare_baselines,
)


def main():
    """Run EA baseline comparison."""
    
    print("=" * 80)
    print("Baseline Comparison Example: EA on simple_spread_v3")
    print("=" * 80)
    print()
    
    # Configure comparison
    config = ComparisonConfig(
        algorithms=['ea'],  # Only EA for now
        env_name='simple_spread_v3',
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=50000,  # Small budget for quick demo
        num_seeds=2,  # Run with 2 different seeds
        eval_episodes=20,  # Evaluate on 20 episodes
        algorithm_configs={
            'ea': {
                'trainer_params': {
                    'hidden_size': 32,
                    'population_size': 20,
                    'tournament_size': 3,
                    'crossover_rate': 0.7,
                    'mutation_rate': 0.3,
                },
                'algorithm_params': {
                    'max_steps_per_episode': 200,
                },
            },
        },
        output_dir=Path('results/ea_baseline'),
    )
    
    # Run comparison
    print("Starting baseline comparison...")
    print(f"  Environment: {config.env_name}")
    print(f"  Budget: {config.budget_value} {config.budget_mode.value}")
    print(f"  Seeds: {config.num_seeds}")
    print(f"  Evaluation episodes: {config.eval_episodes}")
    print()
    
    results = compare_baselines(config)
    
    # Display summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    for algorithm in config.algorithms:
        if algorithm in results:
            algo_results = results[algorithm]
            print(f"\n{algorithm.upper()}:")
            print(f"  Mean reward: {algo_results['mean_reward']:.2f} ± {algo_results['std_reward']:.2f}")
            print(f"  Min reward: {algo_results['min_reward']:.2f}")
            print(f"  Max reward: {algo_results['max_reward']:.2f}")
            print(f"  Best seed: {algo_results['best_seed']}")
            
            # Show per-seed breakdown
            print(f"\n  Per-seed results:")
            for seed_result in algo_results['seeds']:
                seed = seed_result['seed']
                mean_r = seed_result['eval_result']['mean_reward']
                std_r = seed_result['eval_result']['std_reward']
                time = seed_result['training_time']
                print(f"    Seed {seed}: {mean_r:.2f} ± {std_r:.2f} (trained in {time:.1f}s)")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {config.output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
