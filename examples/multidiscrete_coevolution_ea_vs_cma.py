"""Co-evolution example on a synthetic MultiDiscrete env — EA vs CMA-ES.

The built-in MPE2 environments only expose ``Discrete`` action spaces, so
there is no shipped multi-agent env with a native ``MultiDiscrete`` action
space to demonstrate the co-evolution API against. This example vendors a
tiny, fully-homogeneous two-agent :class:`~pettingzoo.utils.env.ParallelEnv`
(``MultiDiscreteParallelEnv`` below) whose reward is ``-sum(action)`` per
agent.

That reward is intentionally simple and **not adversarial** — each agent's
reward does not depend on the other agent's actions, so this is a demo of
the co-evolution *API surface* (sides, opponent sampling, Hall-of-Fame),
not a meaningful competitive benchmark. Swap in a real interactive env for
actual competitive results.

``FeedforwardNeuralPolicy`` natively supports ``MultiDiscrete`` action
spaces via ``MultiDiscreteHead``, so no flattening wrapper is needed.

Uses ``CoEvolutionTrainer`` / ``competitive_2pop``.

Configuration lives at the top of this file via plain module-level
constants — edit the toggles and budget constants to control experiment scope.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium.spaces as spaces
import numpy as np
from pettingzoo.utils.env import ParallelEnv

from deap_pettingzoo.baselines.base import BudgetMode, RecordingConfig, TrainingConfig
from deap_pettingzoo.baselines.coevolution import CoEvolutionConfig, Side
from deap_pettingzoo.baselines.coevolution.trainer import CoEvolutionTrainer
from deap_pettingzoo.environments.registry import register_env

# Allow running as a plain script *or* as a module (-m).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ===========================================================================
# Top-level config (edit these — no CLI parsing)
# ===========================================================================

# ---- Pairing toggle ---------------------------------------------------------

RUN_EA_VS_CMA: bool = True

# ---- Reproducibility / hardware -------------------------------------------

SEED_BASE: int = 0
NUM_SEEDS: int = 1
N_WORKERS: int = 4

# ---- Recording ------------------------------------------------------------
# The synthetic env below has no visual state (``render()`` returns None),
# so champion-vs-champion video recording is disabled here. Point
# ``RecordingConfig(enabled=True, ...)`` at a real rendering env instead.
RECORD_VIDEOS: bool = False

# ---- Budget ------------------------------------------------------------
# Small budget for a quick smoke run. Bump COEV_TOTAL_STEPS for a longer run.

COEV_TOTAL_STEPS: int = 20_000

# ---- Synthetic MultiDiscrete env config ------------------------------------

ENV_OBS_DIM: int = 4
ENV_NVEC: tuple = (3, 4)  # sum(nvec) = 7 network outputs
ENV_MAX_CYCLES: int = 50

# ---- EA algorithm config ---------------------------------------------------

EA_HIDDEN: int = 32
EA_POP: int = 16
EA_OPTIMIZER_CONFIG: Dict[str, Any] = {
    "hidden_size": EA_HIDDEN,
    "population_size": EA_POP,
    "crossover_rate": 0.7,
    "mutation_rate": 0.3,
    "tournament_size": 3,
    "crossover_alpha": 0.5,
    "mutation_sigma": 0.2,
    "mutation_indpb": 0.1,
    "weight_range": (-1.0, 1.0),
    "activation": "tanh",
    "output_activation": "identity",
}

# ---- CMA-ES algorithm config -----------------------------------------------

CMA_HIDDEN: int = 32
CMA_POP: int = 16
CMA_OPTIMIZER_CONFIG: Dict[str, Any] = {
    "hidden_size": CMA_HIDDEN,
    "population_size": CMA_POP,
    "sigma0": 0.3,
    "activation": "tanh",
    "output_activation": "identity",
}

# ---- Coevolution hyperparams -----------------------------------------------

K_OPPONENTS: int = 4
HOF_CAPACITY: int = 5

# ---- Output dir ------------------------------------------------------------

OUTPUT_DIR = Path("results/multidiscrete_coevolution")

# ===========================================================================
# Environment names
# ===========================================================================

ENV_NAME_COEV = "multidiscrete_coevolution_v0"


# ===========================================================================
# Synthetic MultiDiscrete env
# ===========================================================================


class MultiDiscreteParallelEnv(ParallelEnv):
    """Two-agent homogeneous env with a shared ``MultiDiscrete`` action space.

    The reward is ``-sum(action)`` per agent, so lower sub-action indices are
    better — a simple, learnable, but non-adversarial signal.
    """

    metadata = {"name": ENV_NAME_COEV, "is_parallelizable": True}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        *,
        obs_dim: int = ENV_OBS_DIM,
        nvec=ENV_NVEC,
        max_cycles: int = ENV_MAX_CYCLES,
    ) -> None:
        self.render_mode = render_mode
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self._obs_dim = int(obs_dim)
        self._nvec = np.asarray(nvec, dtype=np.int64)
        self._max_cycles = int(max_cycles)
        self._step_count = 0
        self._rng = np.random.default_rng(0)
        self._obs_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(self._obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self._act_spaces = {
            a: spaces.MultiDiscrete(self._nvec.copy()) for a in self.possible_agents
        }

    def observation_space(self, agent: str):
        return self._obs_spaces[agent]

    def action_space(self, agent: str):
        return self._act_spaces[agent]

    def _obs(self, agent: str) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=self._obs_dim).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options=None):
        self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._step_count = 0
        observations = {a: self._obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        self._step_count += 1
        rewards = {}
        for a in self.agents:
            act = np.asarray(actions[a]).ravel()
            rewards[a] = float(-np.sum(act))
        done = self._step_count >= self._max_cycles
        observations = {a: self._obs(a) for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: done for a in self.agents}
        infos = {a: {} for a in self.agents}
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return None

    def close(self):
        return None


def parallel_env(
    render_mode: Optional[str] = None, **kwargs
) -> MultiDiscreteParallelEnv:
    """Top-level, picklable factory returning a :class:`MultiDiscreteParallelEnv`."""
    return MultiDiscreteParallelEnv(render_mode=render_mode, **kwargs)


# ===========================================================================
# Environment registration
# ===========================================================================


def _register_envs() -> None:
    """Register the synthetic MultiDiscrete coevolution environment."""
    register_env(
        ENV_NAME_COEV,
        parallel_env,
        default_episode_len=ENV_MAX_CYCLES,
        continuous_default=False,
        notes="Synthetic homogeneous MultiDiscrete env for the coevolution API demo.",
        tags={"multidiscrete", "coevolution", "synthetic"},
        api="parallel",
        overwrite=True,
    )


# ===========================================================================
# Side helpers
# ===========================================================================


def _make_ea_side(name: str, role_prefix: str) -> Side:
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="ea",
        optimizer_config=dict(EA_OPTIMIZER_CONFIG),
    )


def _make_cma_side(name: str, role_prefix: str) -> Side:
    """Build a CMA-ES :class:`Side`."""
    return Side(
        name=name,
        role_prefixes=(role_prefix,),
        optimizer="cma",
        optimizer_config=dict(CMA_OPTIMIZER_CONFIG),
    )


# ===========================================================================
# Core coevolution runner
# ===========================================================================


def _run_coev(
    red_side: Side,
    blue_side: Side,
    seed: int,
    run_dir: Path,
    label: str,
    budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one coevolution training run and return a result dict."""
    coev_cfg = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[red_side, blue_side],
        schedule="synchronous",
        k_opponents=K_OPPONENTS,
        aggregation="mean",
        reporting_side=red_side.name,
        max_steps_per_episode=ENV_MAX_CYCLES,
        seed=seed,
        num_workers=N_WORKERS,
        parallel_backend="process",
    )
    train_cfg = TrainingConfig(
        env_name=ENV_NAME_COEV,
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=int(COEV_TOTAL_STEPS if budget is None else budget),
        num_seeds=1,
        coevolution_config=coev_cfg,
        max_steps_per_episode=ENV_MAX_CYCLES,
        output_dir=run_dir,
        recording=RecordingConfig(enabled=RECORD_VIDEOS),
    )
    t0 = time.time()
    result = CoEvolutionTrainer().train(train_cfg)
    elapsed = time.time() - t0
    return {
        "policy": result.policy,
        "metrics": result.metrics,
        "training_time_s": elapsed,
        "seed": seed,
        "label": label,
    }


# ===========================================================================
# Pairing runner
# ===========================================================================


def _section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def run_ea_vs_cma(seed: int, run_dir: Path) -> Dict[str, Any]:
    """agent_0=EA, agent_1=CMA-ES."""
    _section(f"EA (agent_0) vs CMA-ES (agent_1) | seed={seed}")
    return _run_coev(
        red_side=_make_ea_side("agent_0", "agent_0"),
        blue_side=_make_cma_side("agent_1", "agent_1"),
        seed=seed,
        run_dir=run_dir,
        label="ea_vs_cma",
    )


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    _register_envs()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairing_registry = [
        ("ea_vs_cma", RUN_EA_VS_CMA, run_ea_vs_cma),
    ]

    summary: List[Dict[str, Any]] = []

    for pairing_name, enabled, run_fn in pairing_registry:
        if not enabled:
            print(f"[skip] {pairing_name}")
            continue

        for seed_offset in range(NUM_SEEDS):
            seed = SEED_BASE + seed_offset
            run_dir = OUTPUT_DIR / pairing_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            try:
                result = run_fn(seed, run_dir)
                summary.append(
                    {
                        "pairing": pairing_name,
                        "seed": seed,
                        "training_time_s": result.get("training_time_s", 0.0),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                print(f"[ERROR] {pairing_name} seed={seed}: {exc}")
                traceback.print_exc()
                summary.append(
                    {
                        "pairing": pairing_name,
                        "seed": seed,
                        "training_time_s": 0.0,
                        "status": f"error: {exc}",
                    }
                )

    # ---- Summary table ----
    _section("Summary")
    header = f"{'Pairing':<20}  {'Seed':>4}  {'Time (s)':>9}  Status"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['pairing']:<20}  {row['seed']:>4}  "
            f"{row['training_time_s']:>9.1f}  {row['status']}"
        )


if __name__ == "__main__":
    main()
