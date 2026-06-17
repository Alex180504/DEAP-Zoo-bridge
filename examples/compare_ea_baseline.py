"""
Example: Compare EA vs MADDPG on simple_spread_v3

This script demonstrates the full baseline comparison API:
- Pre-flight compatibility check via the extensibility module
- EA (Evolutionary Algorithm via DEAP) training
- MADDPG (Multi-Agent DDPG via AgileRL) training — requires the
  optional ``rl-baselines`` extra
  (``pip install 'deap-pettingzoo[rl-baselines]'``); AgileRL is imported
  lazily inside :meth:`MADDPGTrainer.train`, so the script runs
  end-to-end regardless of whether the extra is installed.
- Side-by-side reward statistics and a compatibility matrix
"""

from pathlib import Path

from deap_pettingzoo.baselines import (
    ComparisonConfig,
    BudgetMode,
    compare_baselines,
    build_compatibility_matrix,
    format_compatibility_matrix,
    get_compatible_algorithms,
    inspect_environment,
)

ENV_NAME = 'simple_spread_v3'
ALGORITHMS = ['ea', 'maddpg']


def _section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print('=' * 80)


def main():
    """Run EA vs MADDPG comparison with compatibility pre-flight."""

    _section("EA vs MADDPG Baseline Comparison")

    # ------------------------------------------------------------------
    # 1. Environment introspection
    # ------------------------------------------------------------------
    _section(f"Environment: {ENV_NAME}")
    env_info = inspect_environment(ENV_NAME)
    print(f"  Agents            : {env_info.num_agents}  ({', '.join(env_info.agent_ids)})")
    print(f"  Observation dim   : {env_info.obs_dim}")
    print(f"  Action dim        : {env_info.act_dim}")
    print(f"  Continuous actions: {env_info.has_continuous_actions}")
    print(f"  Discrete actions  : {env_info.has_discrete_actions}")
    print(f"  Global state      : {env_info.has_global_state}")

    # ------------------------------------------------------------------
    # 2. Compatibility matrix
    # ------------------------------------------------------------------
    _section("Compatibility Check")
    matrix = build_compatibility_matrix(
        algorithms=ALGORITHMS,
        env_names=[ENV_NAME],
    )
    print(format_compatibility_matrix(matrix, show_warnings=True))

    compatible = get_compatible_algorithms(ENV_NAME)
    print(f"\nAlgorithms compatible with '{ENV_NAME}': {compatible}")

    # Only run algorithms that are both registered and compatible
    algorithms_to_run = [a for a in ALGORITHMS if a in compatible]
    if not algorithms_to_run:
        print("No compatible algorithms found. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Comparison run
    # ------------------------------------------------------------------
    _section("Training & Evaluation")

    config = ComparisonConfig(
        algorithms=algorithms_to_run,
        env_name=ENV_NAME,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=50_000,   # small budget for quick demo
        num_seeds=2,
        eval_episodes=20,
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
            'maddpg': {
                # MADDPGTrainer.__init__ takes (device, verbose). Algorithm
                # hyperparameters go under 'algorithm_params' and are
                # forwarded to AgileRL's MADDPG(...) constructor.
                'trainer_params': {
                    'device': 'cpu',
                    # Periodic progress prints so long CPU training does
                    # not look like a hang.
                    'verbose': True,
                },
                'algorithm_params': {
                    'seed': 0,
                    'batch_size': 128,
                    'memory_size': 10_000,
                    'learning_delay': 500,
                    # Throttle gradient updates: learn once every 100 env
                    # steps after ``learning_delay``. Drastically faster on
                    # CPU than the previous per-step learning.
                    'learn_step': 100,
                    'lr_actor': 3e-4,
                    'lr_critic': 3e-4,
                },
            },
        },
        output_dir=Path('results/ea_vs_maddpg'),
    )

    print(f"\n  Budget  : {config.budget_value:,} {config.budget_mode.value}")
    print(f"  Seeds   : {config.num_seeds}")
    print(f"  Eval ep.: {config.eval_episodes}")
    print(f"  Algos   : {algorithms_to_run}")

    # validate=True (default) — skips incompatible algorithms automatically
    results = compare_baselines(config, validate=True)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    _section("COMPARISON SUMMARY")

    rows = []
    for algorithm in algorithms_to_run:
        algo_results = results.get(algorithm)
        if algo_results is None or algo_results.get('skipped'):
            issues = algo_results.get('issues', ['unknown reason']) if algo_results else ['not run']
            print(f"\n{algorithm.upper()}: SKIPPED")
            for issue in issues:
                print(f"  - {issue}")
            continue

        mean_r = algo_results['mean_reward']
        std_r  = algo_results['std_reward']
        min_r  = algo_results['min_reward']
        max_r  = algo_results['max_reward']
        best_s = algo_results['best_seed']

        rows.append((algorithm, mean_r, std_r))

        print(f"\n{algorithm.upper()}:")
        print(f"  Mean reward : {mean_r:.2f} ± {std_r:.2f}")
        print(f"  Range       : [{min_r:.2f}, {max_r:.2f}]")
        print(f"  Best seed   : {best_s}")
        print(f"  Per-seed breakdown:")
        for seed_result in algo_results['seeds']:
            seed   = seed_result['seed']
            sr_m   = seed_result['eval_result']['mean_reward']
            sr_std = seed_result['eval_result']['std_reward']
            t      = seed_result['training_time']
            print(f"    Seed {seed}: {sr_m:.2f} ± {sr_std:.2f}  (trained in {t:.1f}s)")

    # Head-to-head if both completed
    if len(rows) == 2:
        _section("Head-to-Head")
        rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
        winner, w_mean, w_std = rows_sorted[0]
        loser,  l_mean, _     = rows_sorted[1]
        delta = w_mean - l_mean
        print(f"  Winner : {winner.upper()} ({w_mean:.2f} ± {w_std:.2f})")
        print(f"  Delta  : {winner.upper()} > {loser.upper()} by {delta:.2f} reward units")

    _section(f"Results saved to: {config.output_dir}")


if __name__ == '__main__':
    main()

