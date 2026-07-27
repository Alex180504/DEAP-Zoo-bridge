"""Opponent-difficulty curriculum for the aquarium vs-APF training suites.

Mechanism
---------
All phases register under **one** env name via :class:`PhasedEnvFactory`.
Every built env is a :class:`PhaseSwitchingEnv` that re-reads a small
phase file at ``reset()`` and rebuilds its inner env when the phase has
advanced. The training driver owns a :class:`CurriculumController` that
writes the phase index as *consumed* env steps cross the configured
budget fractions — directly from the EA/CMA generation loops, or through
the ``progress_callback`` hook of the library trainers (PSO, MAP-Elites,
HyperNEAT, MATD3). This covers every trainer in the campaign because
process workers build a fresh env per evaluation task (build-time phase
read) and long-lived RL envs re-check once per episode (reset-time read).

The final phase is always the canonical symmetric config, and probes /
final champion evaluations run on a separately registered canonical env,
so curriculum runs stay directly comparable with flat-difficulty runs.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

from examples.pursuit_evasion.aquarium import experiment_config as XC


def phase_env_overrides(learner_role: str) -> List[Dict[str, Any]]:
    """Per-phase env-config overrides annealing the *opponent's* speed.

    Args:
        learner_role: ``"pursuer"`` (scripted prey slowed) or ``"evader"``
            (scripted predators slowed).
    """
    if learner_role not in ("pursuer", "evader"):
        raise ValueError(f"learner_role must be pursuer|evader, got {learner_role!r}")
    key = "prey_max_velocity" if learner_role == "pursuer" else "predator_max_velocity"
    return [{key: float(v)} for v in XC.CURRICULUM_OPPONENT_SPEEDS]


def default_phase_file(tag: str) -> str:
    """Per-driver-process phase-file path (shared with spawned workers).

    The path is baked into the registered env factory, so evaluation
    workers — which run on the same node and filesystem — read the
    driver's file regardless of their own import-time registration.
    """
    return os.path.join(
        tempfile.gettempdir(), f"aquarium_curriculum_{tag}_{os.getpid()}.phase"
    )


def read_phase(path: str) -> int:
    """Current phase index; 0 (easiest) when the file does not exist yet."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return max(0, int(f.read().strip() or 0))
    except (OSError, ValueError):
        return 0


def write_phase(path: str, phase: int) -> None:
    """Atomically publish *phase* so mid-read workers never see a torn file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(str(int(phase)))
    os.replace(tmp, path)


class PhaseSwitchingEnv:
    """Parallel-env facade that swaps its inner env when the phase advances.

    The phase file is re-read on every ``reset()``; a phase change closes
    the current inner env and builds the next phase's from its factory.
    All other attribute access is delegated, so the registry's wrapper
    chain (observation features etc.) composes on the outside unchanged —
    valid because phase overrides never alter observation/action spaces.
    """

    def __init__(
        self,
        phase_factories: Sequence[Callable[..., Any]],
        phase_file: str,
        render_mode: Optional[str] = None,
        build_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._phase_factories = tuple(phase_factories)
        self._phase_file = phase_file
        self._render_mode = render_mode
        self._build_kwargs = dict(build_kwargs or {})
        self._phase = self._current_phase()
        self.inner = self._build(self._phase)

    def _current_phase(self) -> int:
        return min(read_phase(self._phase_file), len(self._phase_factories) - 1)

    def _build(self, phase: int):
        return self._phase_factories[phase](
            render_mode=self._render_mode, **self._build_kwargs
        )

    def reset(self, seed=None, options=None):
        phase = self._current_phase()
        if phase != self._phase:
            try:
                self.inner.close()
            except (Exception, SystemExit):
                pass
            self._phase = phase
            self.inner = self._build(phase)
        return self.inner.reset(seed=seed, options=options)

    def __getattr__(self, name: str):
        return getattr(self.__dict__["inner"], name)


class PhasedEnvFactory:
    """Picklable registry factory producing :class:`PhaseSwitchingEnv`s.

    ``phase_factories`` are plain picklable factory instances (one per
    phase, e.g. ``_ScriptedAquariumFactory``), so the whole object ships
    to evaluation workers inside the registry snapshot.

    ``controlled_keys`` are stripped from incoming build kwargs:
    ``build_env(name, env_config=...)`` splats the caller's env config
    over the factory's, so a trainer passing the canonical config would
    otherwise silently overwrite the annealed values and neutralise the
    curriculum.
    """

    def __init__(
        self,
        phase_factories: Sequence[Callable[..., Any]],
        phase_file: str,
        controlled_keys: Sequence[str] = (),
    ) -> None:
        self.phase_factories = tuple(phase_factories)
        self.phase_file = str(phase_file)
        self.controlled_keys = frozenset(controlled_keys)

    def __call__(self, render_mode: Optional[str] = None, **kwargs):
        build_kwargs = {
            k: v for k, v in kwargs.items() if k not in self.controlled_keys
        }
        return PhaseSwitchingEnv(
            self.phase_factories,
            self.phase_file,
            render_mode=render_mode,
            build_kwargs=build_kwargs,
        )


class CurriculumController:
    """Driver-side phase publisher keyed on consumed env steps.

    ``update()`` is called from the EA/CMA generation loops with the
    consumed-step counter; :meth:`progress_callback` adapts the same logic
    to the library trainers' ``ProgressEvent`` stream (whose ``env_steps``
    is the consumed axis for every campaign trainer).
    """

    def __init__(self, phase_file: str, budget_steps: int) -> None:
        self._phase_file = phase_file
        self._budget = max(1, int(budget_steps))
        self._fracs = tuple(XC.CURRICULUM_PHASE_FRACS)
        self._published: Optional[int] = None
        self.update(0)

    def phase_at(self, env_steps: int) -> int:
        phase = 0
        for i, frac in enumerate(self._fracs):
            if env_steps >= frac * self._budget:
                phase = i
        return phase

    def update(self, env_steps: int) -> None:
        phase = self.phase_at(int(env_steps))
        if phase != self._published:
            write_phase(self._phase_file, phase)
            self._published = phase

    def progress_callback(self, event: Any) -> None:
        """``ProgressCallback`` adapter for the library trainers."""
        self.update(int(getattr(event, "env_steps", 0)))


def register_curriculum_scripted_aquarium(
    env_name: str,
    env_config: Dict[str, Any],
    phase_overrides: Sequence[Dict[str, Any]],
    phase_file: str,
    apf_config: Any = None,
    *,
    reward_shape_alpha: float = 0.0,
    fov_shape_alpha: float = 0.0,
    wrappers: Sequence[Callable[[Any], Any]] = (),
) -> str:
    """Register a phase-switching scripted-prey env (pursuer training)."""
    from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
        AquariumAPFConfig,
        _ScriptedAquariumFactory,
    )
    from deap_pettingzoo.environments.registry import register_env

    if apf_config is None:
        apf_config = AquariumAPFConfig()
    factories = [
        _ScriptedAquariumFactory(
            env_config={**env_config, **overrides},
            apf_config=apf_config,
            reward_shape_alpha=reward_shape_alpha,
            fov_shape_alpha=fov_shape_alpha,
        )
        for overrides in phase_overrides
    ]
    controlled = {k for overrides in phase_overrides for k in overrides}
    register_env(
        env_name,
        PhasedEnvFactory(factories, phase_file, controlled_keys=controlled),
        default_episode_len=int(env_config.get("max_time_steps", 3000)),
        continuous_default=bool(env_config.get("continuous_actions", False)),
        notes="Curriculum (phase-switching) scripted-prey aquarium env.",
        tags={"aquarium", "scripted_opponent", "curriculum", "third_party"},
        api="parallel",
        wrappers=tuple(wrappers),
        overwrite=True,
    )
    return env_name


def register_curriculum_scripted_predator_aquarium(
    env_name: str,
    env_config: Dict[str, Any],
    phase_overrides: Sequence[Dict[str, Any]],
    phase_file: str,
    apf_config: Any = None,
    *,
    reward_shape_alpha: float = 0.0,
    close_range_penalty: float = 0.0,
    close_range_sigma2: float = 0.01,
    wrappers: Sequence[Callable[[Any], Any]] = (),
) -> str:
    """Register a phase-switching scripted-predator env (evader training)."""
    from examples.pursuit_evasion.aquarium.scripted_opponent_env import (
        AquariumAPFConfig,
        _ScriptedPredatorAquariumFactory,
    )
    from deap_pettingzoo.environments.registry import register_env

    if apf_config is None:
        apf_config = AquariumAPFConfig()
    factories = [
        _ScriptedPredatorAquariumFactory(
            env_config={**env_config, **overrides},
            apf_config=apf_config,
            reward_shape_alpha=reward_shape_alpha,
            close_range_penalty=close_range_penalty,
            close_range_sigma2=close_range_sigma2,
        )
        for overrides in phase_overrides
    ]
    controlled = {k for overrides in phase_overrides for k in overrides}
    register_env(
        env_name,
        PhasedEnvFactory(factories, phase_file, controlled_keys=controlled),
        default_episode_len=int(env_config.get("max_time_steps", 3000)),
        continuous_default=bool(env_config.get("continuous_actions", False)),
        notes="Curriculum (phase-switching) scripted-predator aquarium env.",
        tags={"aquarium", "scripted_opponent", "curriculum", "evader_training", "third_party"},
        api="parallel",
        wrappers=tuple(wrappers),
        overwrite=True,
    )
    return env_name
