"""``ScriptedOpponentEnvWrapper`` for ``marl_aquarium.aquarium_v0``.

Wraps the aquarium parallel env so its public PettingZoo API surface
shows *only* the predator agents.  Prey are stepped internally each
``step()`` call by a scripted policy
(:class:`~examples.pursuit_evasion.aquarium.apf_policy.AquariumAPFEvaderPolicy`)
and never exposed to the learner-facing API.  This lets us train any
deap-bridge / AgileRL trainer (EA, CMA-ES, MAPPO, ...) on aquarium with
*pursuers only* against a fixed scripted prey controller.

The wrapper also exposes ``self.prey_caught`` — a per-episode integer
counter, derived from the predator-team reward (``+predator_reward``
per catch in aquarium) — so callers can use a sparse
``catches-per-episode`` fitness signal.
"""

from __future__ import annotations

import math
import types as _types
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from deap_pettingzoo.environments.registry import (
    is_registered,
    register_env,
)

from .apf_policy import AquariumAPFConfig, AquariumAPFEvaderPolicy


def _patch_aquarium_wrappers(env) -> None:  # noqa: ANN001
    """Fix PettingZoo crashes when prey die with keep_prey_count_constant=False.

    When a fish is caught, marl-aquarium removes it from all env dicts
    WITHOUT setting ``terminations[fish] = True``.
    ``parallel_to_aec_wrapper`` then appends these "zombie" agents back
    into ``self.agents`` (from ``self._observations``) while they remain
    absent from ``self.terminations``.  ``_deads_step_first`` iterates
    ``self.agents`` and accesses ``self.terminations[agent]`` directly,
    causing a ``KeyError``.

    Fix: patch ``_deads_step_first`` on the ``parallel_to_aec_wrapper``
    instance to silently drop zombie agents (present in ``self.agents``,
    absent from ``self.terminations``, and gone from the raw env's live
    agent list) before processing any normally-terminated agents.  This
    keeps ``self.agents`` consistent so the outer
    ``aec_to_parallel_wrapper.step()`` reward loop never encounters an
    agent that is missing from ``self.rewards``.
    """

    def _safe_deads_step_first(self) -> None:  # noqa: ANN001
        # Identify zombie agents: they appear in self.agents (copied from
        # self._observations by parallel_to_aec_wrapper) but were already
        # removed from the raw env and are therefore absent from
        # self.terminations.
        live_set: set = set(getattr(self.env, "agents", []))
        for a in list(self.agents):
            if a not in self.terminations and a not in live_set:
                # Remove zombie cleanly without going through _was_dead_step.
                self.agents.remove(a)
                self.rewards.pop(a, None)
                self._cumulative_rewards.pop(a, None)
                self.infos.pop(a, None)

        # Standard PettingZoo logic for normally-terminated agents.
        _deads_order = [
            agent for agent in self.agents
            if self.terminations.get(agent, False) or self.truncations.get(agent, False)
        ]
        if _deads_order:
            self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        return self.agent_selection

    visited: set = set()
    node = env
    for _ in range(15):
        if id(node) in visited:
            break
        visited.add(id(node))
        if type(node).__name__ == "parallel_to_aec_wrapper":
            node._deads_step_first = _types.MethodType(_safe_deads_step_first, node)
        advanced = False
        for attr in ("aec_env", "env"):
            if hasattr(node, attr):
                node = getattr(node, attr)
                advanced = True
                break
        if not advanced:
            break


class ScriptedOpponentEnvWrapper:
    """Parallel-API facade over a marl-aquarium env that hides prey agents.

    Exposes only predator agents to the public API.  Prey are stepped
    internally by per-agent scripted policies (default: APF flee).

    Attributes:
        possible_agents: Predator agent ids only.
        agents: Currently-active predator agent ids.
        prey_caught: Cumulative catches in the current episode (reset
            on every :meth:`reset`).
        last_team_reward: Sum of predator rewards from the most recent
            :meth:`step` (useful for diagnostics / shaping).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "scripted_aquarium"}

    def __init__(
        self,
        inner_env,
        *,
        predator_observe_count: int,
        prey_observe_count: int,
        action_count: int = 16,
        predator_reward: int = 10,
        learner_prefix: str = "predator",
        scripted_prefix: str = "prey",
        apf_config: Optional[AquariumAPFConfig] = None,
        reward_shape_alpha: float = 0.0,
        fov_shape_alpha: float = 0.0,
    ) -> None:
        """
        Args:
            inner_env: A parallel-API marl-aquarium env (the output of
                ``marl_aquarium.aquarium_v0.parallel_env(...)``).
            predator_observe_count: Aquarium ``predator_observe_count`` —
                used to size the prey APF policies' obs layout.
            prey_observe_count: Aquarium ``prey_observe_count``.
            action_count: Aquarium ``action_count`` (default 16).
            predator_reward: Aquarium ``predator_reward`` (default 10).
                Used to derive the catch counter from the team reward
                stream.
            learner_prefix: Agent-name prefix for learner-side agents
                (default ``"predator"`` — exposed via the public API).
            scripted_prefix: Agent-name prefix for scripted-side agents
                (default ``"prey"`` — driven internally by APF).
            apf_config: Optional :class:`AquariumAPFConfig`.
        """
        self.inner = inner_env
        self.predator_observe_count = int(predator_observe_count)
        self.prey_observe_count = int(prey_observe_count)
        self.action_count = int(action_count)
        self.predator_reward = float(predator_reward)
        self.learner_prefix = learner_prefix
        self.scripted_prefix = scripted_prefix
        self._apf_config = apf_config or AquariumAPFConfig()
        self.reward_shape_alpha = float(reward_shape_alpha)
        self.fov_shape_alpha = float(fov_shape_alpha)

        # Filter possible_agents to learner-prefixed only.
        self.possible_agents = [
            a for a in self.inner.possible_agents if a.startswith(self.learner_prefix)
        ]
        self.agents: List[str] = list(self.possible_agents)
        self._all_possible_agents = list(self.inner.possible_agents)

        # Per-prey APF policy + last-observation cache.
        self._scripted_policies: Dict[str, AquariumAPFEvaderPolicy] = {}
        self._scripted_last_obs: Dict[str, np.ndarray] = {}
        for agent_id in self._all_possible_agents:
            if agent_id.startswith(self.scripted_prefix):
                self._scripted_policies[agent_id] = AquariumAPFEvaderPolicy(
                    predator_observe_count=self.prey_observe_count,
                    prey_observe_count=self.prey_observe_count,
                    action_count=self.action_count,
                    config=self._apf_config,
                )

        # Per-episode catch counter (sparse fitness signal).
        self.prey_caught: int = 0
        self.last_team_reward: float = 0.0
        self._prev_min_prey_distances: Dict[str, Optional[float]] = {}

        # Navigate to the raw env so we can read ``prey.recently_died``.
        # When aquarium is configured with ``keep_prey_count_constant=True``
        # (the default), caught prey are respawned instead of removed, so
        # the internal ``catches`` list stays empty and predator rewards are
        # always 0.  We compensate by detecting catches via the flag and
        # injecting the correct reward in ``step()``.
        self._raw_env = self._find_raw_env(inner_env)

        # Public observation/action space dicts (PettingZoo convention),
        # filtered to learner side only.
        try:
            self.observation_spaces = {
                a: self.inner.observation_space(a) for a in self.possible_agents
            }
            self.action_spaces = {
                a: self.inner.action_space(a) for a in self.possible_agents
            }
        except Exception:
            self.observation_spaces = {}
            self.action_spaces = {}

    # ParallelEnv API ------------------------------------------------------

    def observation_space(self, agent: str):
        return self.inner.observation_space(agent)

    def action_space(self, agent: str):
        return self.inner.action_space(agent)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset and return ``(obs, infos)`` per modern PettingZoo parallel API.

        AgileRL's trainer expects this tuple shape.  The deap-bridge
        :class:`MixedPolicyEvaluator` and CRN pmap helper expect just
        the obs dict — for those code paths, wrap this env with
        :class:`DictResetEnv` (defined below).
        """
        observations, infos = self.inner.reset(seed=seed, options=options)
        # Cache prey observations for the next step's APF call.
        self._scripted_last_obs = {
            a: np.asarray(observations[a], dtype=np.float64)
            for a in observations
            if a.startswith(self.scripted_prefix)
        }
        # Reset all scripted policies (no-op for APF, but contract-correct).
        for pol in self._scripted_policies.values():
            pol.reset()
        # Filter to learner-side.
        self.agents = [
            a for a in self.inner.agents if a.startswith(self.learner_prefix)
        ]
        learner_obs = {a: observations[a] for a in self.agents if a in observations}
        learner_infos = {a: infos.get(a, {}) for a in self.agents}
        # Reset episode counters.
        self.prey_caught = 0
        self.last_team_reward = 0.0
        self._prev_min_prey_distances = self._compute_min_prey_distances(self.agents)
        return learner_obs, learner_infos

    def step(
        self,
        actions: Dict[str, Any],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        # Build full action dict: learner actions + scripted prey actions.
        full_actions: Dict[str, Any] = {}
        for agent_id in self.inner.agents:
            if agent_id.startswith(self.learner_prefix):
                if agent_id not in actions:
                    raise ValueError(
                        f"ScriptedOpponentEnvWrapper.step: missing action "
                        f"for learner agent {agent_id!r}"
                    )
                full_actions[agent_id] = actions[agent_id]
            elif agent_id.startswith(self.scripted_prefix):
                pol = self._scripted_policies.get(agent_id)
                if pol is None:
                    full_actions[agent_id] = 0
                    continue
                last_obs = self._scripted_last_obs.get(agent_id)
                if last_obs is None:
                    # First step after a respawn — emit the no-op /
                    # forward-keep action; the next step will have a
                    # real observation cached.
                    full_actions[agent_id] = 0
                else:
                    try:
                        full_actions[agent_id] = int(pol.get_action(last_obs))
                    except Exception:
                        full_actions[agent_id] = 0
            else:
                # Unknown prefix — pass through with a no-op.
                full_actions[agent_id] = 0

        observations, rewards, terminations, truncations, infos = self.inner.step(
            full_actions
        )

        # Update prey-obs cache for the next step.
        self._scripted_last_obs = {
            a: np.asarray(observations[a], dtype=np.float64)
            for a in observations
            if a.startswith(self.scripted_prefix)
        }

        # Catch counter — derive from predator team reward.
        # In aquarium each catch grants ``predator_reward`` to predators
        # within ``catch_radius`` (split among them).  Summed across the
        # team, total predator-side reward equals
        # ``predator_reward * num_catches_this_step``.
        team_reward = sum(
            float(rewards.get(a, 0.0))
            for a in rewards
            if a.startswith(self.learner_prefix)
        )

        # When keep_prey_count_constant=True (aquarium default), caught prey
        # are respawned without being marked dead, so the internal catches
        # list is always empty and team_reward == 0 even after a catch.
        # Detect catches directly from the recently_died flag on raw prey.
        _recently_died = 0
        if team_reward == 0.0 and self._raw_env is not None:
            try:
                _recently_died = sum(
                    1 for p in self._raw_env.prey
                    if getattr(p, 'recently_died', False)
                )
            except Exception:
                pass

        # Filter output dicts to learner-side only.
        self.agents = [
            a for a in self.inner.agents if a.startswith(self.learner_prefix)
        ]
        learner_keys = self.possible_agents  # keep keys stable across steps
        out_obs = {a: observations[a] for a in learner_keys if a in observations}
        out_rewards = {a: float(rewards.get(a, 0.0)) for a in learner_keys}

        # Progress-based reward shaping: reward only when predators close
        # distance to prey, rather than for merely occupying a stable lane.
        if self.reward_shape_alpha > 0.0 and self._raw_env is not None:
            try:
                prey = list(getattr(self._raw_env, "prey", []))
                torus = getattr(self._raw_env, "torus", None)
                if prey and torus is not None:
                    current_min_dists = self._compute_min_prey_distances(learner_keys)
                    for idx, agent_id in enumerate(learner_keys):
                        current_min_dist = current_min_dists.get(agent_id)
                        prev_min_dist = self._prev_min_prey_distances.get(agent_id)
                        if current_min_dist is None or prev_min_dist is None:
                            continue
                        progress = prev_min_dist - current_min_dist
                        out_rewards[agent_id] += self.reward_shape_alpha * (progress / 200.0)
                    self._prev_min_prey_distances = current_min_dists
            except Exception:
                pass

        # FOV-presence reward shaping: small bonus per prey visible in the
        # predator's view cone each step.  Unlike proximity shaping this
        # rewards *steering toward* prey — the policy must actually face a
        # prey to earn the signal, and the bonus is observation-coherent
        # (the policy can only observe prey it earns reward for).
        if self.fov_shape_alpha > 0.0 and self._raw_env is not None:
            try:
                for idx, agent_id in enumerate(learner_keys):
                    n_visible = self._count_fov_visible_prey(idx)
                    out_rewards[agent_id] += self.fov_shape_alpha * n_visible
            except Exception:
                pass

        out_terms = {a: bool(terminations.get(a, False)) for a in learner_keys}
        out_truncs = {a: bool(truncations.get(a, False)) for a in learner_keys}
        out_infos = {a: infos.get(a, {}) for a in learner_keys}

        # Inject recently_died-based catch reward and update catch counter.
        if _recently_died > 0:
            team_bonus = float(self.predator_reward) * _recently_died
            if learner_keys:
                per_agent_bonus = team_bonus / float(len(learner_keys))
                out_rewards = {
                    a: out_rewards[a] + per_agent_bonus for a in learner_keys
                }
            self.prey_caught += _recently_died
            team_reward = team_bonus
        elif self.predator_reward > 0:
            self.prey_caught += int(round(team_reward / self.predator_reward))

        # When keep_prey_count_constant=False, terminate the episode as soon
        # as no scripted prey remain alive in the inner env.
        if not any(a.startswith(self.scripted_prefix) for a in self.inner.agents):
            out_terms = {a: True for a in learner_keys}

        self.last_team_reward = team_reward

        # Surface per-step diagnostics on info[agent_0] so loggers can
        # pick them up without poking at wrapper attributes.
        if learner_keys:
            head = out_infos.setdefault(learner_keys[0], {})
            head.setdefault("prey_caught_total", int(self.prey_caught))
            head.setdefault("team_reward_step", float(team_reward))

        return out_obs, out_rewards, out_terms, out_truncs, out_infos

    # ------------------------------------------------------------------
    # Internal helpers

    @staticmethod
    def _find_raw_env(env):
        """Walk the PettingZoo wrapper chain to find the raw env.

        The raw env is identified by having both ``predators`` and ``prey``
        attributes (marl-aquarium's ``raw_env`` class).
        """
        inner = env
        for _ in range(12):
            if hasattr(inner, 'predators') and hasattr(inner, 'prey'):
                return inner
            advanced = False
            for attr in ('aec_env', 'env'):
                if hasattr(inner, attr):
                    inner = getattr(inner, attr)
                    advanced = True
                    break
            if not advanced:
                break
        return None

    def _count_fov_visible_prey(self, predator_idx: int) -> int:
        """Return number of live prey visible inside predator's FOV cone."""
        if self._raw_env is None:
            return 0
        predators = list(getattr(self._raw_env, "predators", []))
        prey = list(getattr(self._raw_env, "prey", []))
        torus = getattr(self._raw_env, "torus", None)
        if predator_idx >= len(predators) or not prey or torus is None:
            return 0
        # If FOV is disabled, every prey counts as visible.
        if not getattr(self._raw_env, "fov_enabled", False):
            return len(prey)
        pred = predators[predator_idx]
        view_dist = getattr(self._raw_env, "predator_view_distance", 2000)
        fov = getattr(self._raw_env, "predator_fov", 360)
        check = getattr(torus, "check_if_entity_is_in_view_in_torus", None)
        if check is None:
            return len(prey)
        return sum(1 for p in prey if check(pred, p, view_dist, fov))

    def _compute_min_prey_distances(self, learner_agents: List[str]) -> Dict[str, Optional[float]]:
        """Return nearest FOV-visible prey torus distance for each learner agent.

        Only prey inside the predator's view cone count.  When no prey is
        visible the entry is ``None`` so the shaping step skips that agent
        (no reward for chasing prey you cannot observe).
        If ``fov_enabled`` is False on the raw env every prey counts as
        visible (backward-compatible behaviour).
        """
        if self._raw_env is None:
            return {}
        all_prey = list(getattr(self._raw_env, "prey", []))
        predators = list(getattr(self._raw_env, "predators", []))
        torus = getattr(self._raw_env, "torus", None)
        if not all_prey or torus is None:
            return {}

        fov_enabled: bool = bool(getattr(self._raw_env, "fov_enabled", False))
        view_dist = getattr(self._raw_env, "predator_view_distance", 2000)
        fov = getattr(self._raw_env, "predator_fov", 360)
        check = getattr(torus, "check_if_entity_is_in_view_in_torus", None)

        min_dists: Dict[str, Optional[float]] = {}
        for idx, agent_id in enumerate(learner_agents):
            if idx >= len(predators):
                continue
            pred = predators[idx]
            if fov_enabled and check is not None:
                visible = [p for p in all_prey if check(pred, p, view_dist, fov)]
            else:
                visible = all_prey
            if not visible:
                min_dists[agent_id] = None  # no visible prey — skip shaping
            else:
                min_dists[agent_id] = min(
                    float(torus.get_distance_in_torus(pred.position, p.position))
                    for p in visible
                )
        return min_dists

    def render(self, mode: Optional[str] = None):  # pragma: no cover - viz
        try:
            return self.inner.render()
        except SystemExit:
            return None

    @property
    def unwrapped(self):
        """Gymnasium-style unwrap hook used by AgileRL's vec env.

        ``AsyncPettingZooVecEnv.__init__`` reads
        ``dummy_env.unwrapped.render_mode`` to decide whether to forward
        ``render`` calls into the worker subprocesses.  This wrapper *is*
        the outermost public-API layer (the inner aquarium env exposes
        prey agents that we hide), so we return ``self`` rather than
        the raw env to keep ``possible_agents`` / spaces consistent.
        """
        return self

    @property
    def render_mode(self):
        """Surface the inner env's ``render_mode`` for the vec-env probe."""
        return getattr(self.inner, "render_mode", None)

    def close(self) -> None:
        try:
            self.inner.close()
        except SystemExit:
            # marl-aquarium calls ``sys.exit()`` from ``raw_env.close()``;
            # already handled by deap-bridge but guard here too.
            pass

    def state(self) -> np.ndarray:  # pragma: no cover - optional API
        try:
            return np.asarray(self.inner.state())
        except (AttributeError, NotImplementedError):
            return np.zeros(0)


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


class _ScriptedAquariumFactory:
    """Picklable factory for a registered scripted-prey aquarium env.

    ``register_env(...)`` accepts a callable.  We use a class instance
    rather than ``functools.partial`` so the captured config is
    explicitly addressable for debugging.

    ``__call__(render_mode=..., **kwargs)`` builds a fresh
    :class:`ScriptedOpponentEnvWrapper` around a new
    ``aquarium_v0.parallel_env(...)`` instance.
    """

    def __init__(
        self,
        env_config: Dict[str, Any],
        apf_config: AquariumAPFConfig,
        reward_shape_alpha: float = 0.0,
        fov_shape_alpha: float = 0.0,
    ) -> None:
        self.env_config = dict(env_config)
        self.apf_config = apf_config
        self.reward_shape_alpha = float(reward_shape_alpha)
        self.fov_shape_alpha = float(fov_shape_alpha)

    def __call__(self, render_mode: Optional[str] = None, **kwargs):
        from marl_aquarium import aquarium_v0  # local import — optional dep

        cfg = {**self.env_config, **kwargs}
        # Aquarium's parallel_env expects ``render_mode`` as a *string*
        # ("human" / "rgb_array"); ``None`` is not in its enum.  Map
        # None → "rgb_array" (off-screen, no pygame display) so the
        # registry's default render_mode=None path doesn't crash.
        if render_mode is None:
            render_mode = "rgb_array"
        inner = aquarium_v0.parallel_env(render_mode=render_mode, **cfg)
        # Fix PettingZoo KeyError when prey die with keep_prey_count_constant=False.
        _patch_aquarium_wrappers(inner)
        return ScriptedOpponentEnvWrapper(
            inner,
            predator_observe_count=int(cfg["predator_observe_count"]),
            prey_observe_count=int(cfg["prey_observe_count"]),
            action_count=int(cfg.get("action_count", 16)),
            predator_reward=float(cfg.get("predator_reward", 10)),
            apf_config=self.apf_config,
            reward_shape_alpha=self.reward_shape_alpha,
            fov_shape_alpha=self.fov_shape_alpha,
        )


def register_scripted_aquarium(
    env_name: str,
    env_config: Dict[str, Any],
    apf_config: Optional[AquariumAPFConfig] = None,
    *,
    reward_shape_alpha: float = 0.0,
    fov_shape_alpha: float = 0.0,
    wrappers: Sequence[Callable[[Any], Any]] = (),
    overwrite: bool = True,
) -> str:
    """Register a scripted-prey aquarium env in deap-bridge's env registry.

    Returns the registered name (so the caller can pass it to
    :class:`PettingZooWrapper` / trainers).

    Args:
        env_name: Name to register under.  Must be unique unless
            ``overwrite=True``.
        env_config: Aquarium env kwargs (``predator_count``,
            ``prey_count``, ``predator_observe_count``,
            ``prey_observe_count``, ``max_time_steps``, etc.).
        apf_config: APF prey config.  Defaults to
            :class:`AquariumAPFConfig`'s default values.
        wrappers: Optional ordered wrapper chain applied after the
            scripted-opponent parallel env is built.
        overwrite: If a previous registration with the same name exists,
            replace it.  Default ``True`` so re-running the experiment
            script in the same Python process is a no-op.
    """
    if apf_config is None:
        apf_config = AquariumAPFConfig()
    factory = _ScriptedAquariumFactory(
        env_config=env_config,
        apf_config=apf_config,
        reward_shape_alpha=reward_shape_alpha,
        fov_shape_alpha=fov_shape_alpha,
    )
    if is_registered(env_name) and not overwrite:
        return env_name
    register_env(
        env_name,
        factory,
        default_episode_len=int(env_config.get("max_time_steps", 3000)),
        continuous_default=False,
        notes=(
            "Scripted-prey wrapper around marl_aquarium.aquarium_v0; "
            "exposes only predator agents."
        ),
        tags={"aquarium", "scripted_opponent", "third_party"},
        api="parallel",
        wrappers=tuple(wrappers),
        overwrite=True,
    )
    return env_name


# ---------------------------------------------------------------------------
# Compatibility adapter — strip infos from reset() for deap-bridge code paths
# ---------------------------------------------------------------------------


class DictResetEnv:
    """Thin adapter that forwards everything to ``inner`` but makes
    ``reset()`` return just the obs ``dict`` (not the modern
    ``(obs, infos)`` tuple).

    The deap-bridge :class:`MixedPolicyEvaluator` and the CRN parallel
    pmap helper (`make_mixed_parallel_evaluate_map`) call
    ``env.reset(seed=...)`` and immediately index the result as a dict.
    Use this adapter when handing the env to those code paths; do *not*
    use it for AgileRL trainers, which expect the modern tuple shape.
    """

    def __init__(self, inner) -> None:
        self.inner = inner

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def reset(self, seed=None, options=None):
        result = self.inner.reset(seed=seed, options=options)
        if isinstance(result, tuple) and len(result) == 2:
            obs, _infos = result
            return obs
        return result

    def step(self, actions):
        return self.inner.step(actions)

    def observation_space(self, agent):
        return self.inner.observation_space(agent)

    def action_space(self, agent):
        return self.inner.action_space(agent)

    def close(self) -> None:
        try:
            self.inner.close()
        except SystemExit:
            pass

    def render(self, *args, **kwargs):  # pragma: no cover
        return self.inner.render(*args, **kwargs)

