"""Scripted-opponent env wrappers for ``marl_aquarium.aquarium_v0``.

Wraps the aquarium parallel env so its public PettingZoo API surface
shows *only* one side of the predator/prey roster; the other side is
stepped internally each ``step()`` call by a scripted APF policy
(:mod:`~examples.pursuit_evasion.aquarium.apf_policy`) and never
exposed to the learner-facing API.  This lets us train any
algorithm on aquarium against a fixed scripted opponent, from either side:

* :class:`AquariumScriptedPreyWrapper` — predators are the learners,
  prey are scripted (fleeing).  Exposes ``self.prey_caught``, a
  per-episode integer counter derived from the predator-team reward
  (``+predator_reward`` per catch in aquarium), for a sparse
  ``catches-per-episode`` fitness signal.
* :class:`AquariumScriptedPredatorWrapper` — prey are the learners,
  predators are scripted (pursuing).  Exposes ``self.prey_lost``, a
  per-episode integer counter of learner agents caught.
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

from .apf_policy import (
    AquariumAPFConfig,
    AquariumAPFEvaderPolicy,
    AquariumAPFPursuerPolicy,
)


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


class AquariumScriptedPreyWrapper:
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

        # Must resolve raw env before creating APF policies so we can
        # detect ``observable_walls`` / ``continuous_actions`` and set the
        # correct obs layout + action encoding.
        self._raw_env = self._find_raw_env(inner_env)
        self._continuous = bool(getattr(self._raw_env, "continuous_actions", False))

        # Per-prey APF policy + last-observation cache.
        # Detect wall-feature count from the raw env so the APF policy
        # slices its observation correctly in both torus and rectangle modes.
        _wf = 4 if getattr(self._raw_env, "observable_walls", False) else 0
        self._scripted_policies: Dict[str, AquariumAPFEvaderPolicy] = {}
        self._scripted_last_obs: Dict[str, np.ndarray] = {}
        for agent_id in self._all_possible_agents:
            if agent_id.startswith(self.scripted_prefix):
                self._scripted_policies[agent_id] = AquariumAPFEvaderPolicy(
                    predator_observe_count=self.prey_observe_count,
                    prey_observe_count=self.prey_observe_count,
                    action_count=self.action_count,
                    config=self._apf_config,
                    wall_features=_wf,
                    continuous=self._continuous,
                )

        # Per-episode catch counter (sparse fitness signal).
        self.prey_caught: int = 0
        self.last_team_reward: float = 0.0
        self._prev_min_prey_distances: Dict[str, Optional[float]] = {}

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

    # ParallelEnv API

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
        fallback = self._fallback_action()
        full_actions: Dict[str, Any] = {}
        for agent_id in self.inner.agents:
            if agent_id.startswith(self.learner_prefix):
                if agent_id not in actions:
                    raise ValueError(
                        f"AquariumScriptedPreyWrapper.step: missing action "
                        f"for learner agent {agent_id!r}"
                    )
                full_actions[agent_id] = actions[agent_id]
            elif agent_id.startswith(self.scripted_prefix):
                pol = self._scripted_policies.get(agent_id)
                last_obs = self._scripted_last_obs.get(agent_id)
                if pol is None or last_obs is None:
                    # No policy, or first step after a respawn (no cached
                    # observation yet) — emit the mode-appropriate no-op.
                    full_actions[agent_id] = fallback
                else:
                    try:
                        full_actions[agent_id] = pol.get_action(last_obs)
                    except Exception:
                        full_actions[agent_id] = fallback
            else:
                # Unknown prefix — pass through with a no-op.
                full_actions[agent_id] = fallback

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

        # Update catch counter from the predator team reward.
        if self.predator_reward > 0:
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

    # Internal helpers

    def _fallback_action(self):
        """Mode-appropriate no-op action for scripted agents."""
        if self._continuous:
            return np.zeros(2, dtype=np.float32)
        return 0

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


# Registry helper

class _ScriptedAquariumFactory:
    """Picklable factory for a registered scripted-prey aquarium env.

    ``register_env(...)`` accepts a callable.  Use a class instance
    rather than ``functools.partial`` so the captured config is
    explicitly addressable for debugging.

    ``__call__(render_mode=..., **kwargs)`` builds a fresh
    :class:`AquariumScriptedPreyWrapper` around a new
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
        return AquariumScriptedPreyWrapper(
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
        continuous_default=bool(env_config.get("continuous_actions", False)),
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


# Role-flipped wrapper — scripted APF predators, prey are the learners

class AquariumScriptedPredatorWrapper:
    """Parallel-API facade over a marl-aquarium env that hides predator agents.

    Mirror of :class:`AquariumScriptedPreyWrapper` with prefixes swapped
    (prey are learners; predators are scripted APF pursuers).  Prey-side
    reward composition:

    * Survival: ``+prey_reward`` per active prey per step (emitted natively
      by the inner env; passes through unmodified).
    * Catch penalty: ``-prey_punishment`` injected on the step a prey is
      terminated (with ``keep_prey_count_constant=False`` the native env
      emits 0 on that step because the caught prey is removed before
      ``get_prey_rewards()`` runs).
    * Optional distance shaping: ``+reward_shape_alpha * Δ(min_pred_dist) / 200``
      per active prey per step — reward for opening distance from the
      nearest FOV-visible predator.
    * Optional close-range Gaussian danger penalty near the catch annulus.

    The episode terminates when no learner-prefix prey remain.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "scripted_aquarium_evader"}

    def __init__(
        self,
        inner_env,
        *,
        predator_observe_count: int,
        prey_observe_count: int,
        action_count: int = 16,
        learner_prefix: str = "prey",
        scripted_prefix: str = "predator",
        apf_config: Optional[AquariumAPFConfig] = None,
        reward_shape_alpha: float = 0.0,
        close_range_penalty: float = 0.0,
        close_range_sigma2: float = 0.01,
    ) -> None:
        self.inner = inner_env
        self.predator_observe_count = int(predator_observe_count)
        self.prey_observe_count = int(prey_observe_count)
        self.action_count = int(action_count)
        self.learner_prefix = learner_prefix
        self.scripted_prefix = scripted_prefix
        self._apf_config = apf_config or AquariumAPFConfig()
        self.reward_shape_alpha = float(reward_shape_alpha)
        self.close_range_penalty = float(close_range_penalty)
        self.close_range_sigma2 = float(close_range_sigma2)

        self.possible_agents = [
            a for a in self.inner.possible_agents if a.startswith(self.learner_prefix)
        ]
        self.agents: List[str] = list(self.possible_agents)
        self._all_possible_agents = list(self.inner.possible_agents)

        # Resolve raw env first: wall/continuous detection drives the APF
        # policies' obs layout and action encoding.
        self._raw_env = AquariumScriptedPreyWrapper._find_raw_env(inner_env)
        self._continuous = bool(getattr(self._raw_env, "continuous_actions", False))
        _wf = 4 if getattr(self._raw_env, "observable_walls", False) else 0

        self._scripted_policies: Dict[str, AquariumAPFPursuerPolicy] = {}
        self._scripted_last_obs: Dict[str, np.ndarray] = {}
        for agent_id in self._all_possible_agents:
            if agent_id.startswith(self.scripted_prefix):
                self._scripted_policies[agent_id] = AquariumAPFPursuerPolicy(
                    predator_observe_count=self.predator_observe_count,
                    prey_observe_count=self.prey_observe_count,
                    action_count=self.action_count,
                    config=self._apf_config,
                    wall_features=_wf,
                    continuous=self._continuous,
                )

        self.prey_lost: int = 0
        self._initial_prey_count: int = 0
        self.last_team_reward: float = 0.0
        self._prev_min_pred_distances: Dict[str, Optional[float]] = {}

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

    # ParallelEnv API

    def observation_space(self, agent: str):
        return self.inner.observation_space(agent)

    def action_space(self, agent: str):
        return self.inner.action_space(agent)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        observations, infos = self.inner.reset(seed=seed, options=options)
        self._scripted_last_obs = {
            a: np.asarray(observations[a], dtype=np.float64)
            for a in observations
            if a.startswith(self.scripted_prefix)
        }
        for pol in self._scripted_policies.values():
            pol.reset()
        self.agents = [
            a for a in self.inner.agents if a.startswith(self.learner_prefix)
        ]
        learner_obs = {a: observations[a] for a in self.agents if a in observations}
        learner_infos = {a: dict(infos.get(a, {})) for a in self.agents}
        self.prey_lost = 0
        self._initial_prey_count = (
            len(getattr(self._raw_env, "prey", [])) if self._raw_env is not None else len(self.agents)
        )
        self.last_team_reward = 0.0
        self._prev_min_pred_distances = self._compute_min_pred_distances(self.agents)
        if learner_infos and self.agents:
            head = learner_infos.setdefault(self.agents[0], {})
            head.setdefault("prey_lost_total", 0)
            head.setdefault("prey_remaining", len(self.agents))
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
        fallback = self._fallback_action()
        full_actions: Dict[str, Any] = {}
        for agent_id in self.inner.agents:
            if agent_id.startswith(self.learner_prefix):
                if agent_id not in actions:
                    raise ValueError(
                        f"AquariumScriptedPredatorWrapper.step: missing action "
                        f"for learner agent {agent_id!r}"
                    )
                full_actions[agent_id] = actions[agent_id]
            elif agent_id.startswith(self.scripted_prefix):
                pol = self._scripted_policies.get(agent_id)
                last_obs = self._scripted_last_obs.get(agent_id)
                if pol is None or last_obs is None:
                    full_actions[agent_id] = fallback
                else:
                    try:
                        full_actions[agent_id] = pol.get_action(last_obs)
                    except Exception:
                        full_actions[agent_id] = fallback
            else:
                full_actions[agent_id] = fallback

        observations, rewards, terminations, truncations, infos = self.inner.step(
            full_actions
        )

        # Update predator obs cache.
        self._scripted_last_obs = {
            a: np.asarray(observations[a], dtype=np.float64)
            for a in observations
            if a.startswith(self.scripted_prefix)
        }

        # Filter outputs to learner side (active prey) only.
        active_learners = [
            a for a in self.inner.agents if a.startswith(self.learner_prefix)
        ]
        self.agents = active_learners
        learner_keys = self.possible_agents
        public_learners = [
            a for a in observations if a.startswith(self.learner_prefix)
        ]
        inactive_learners = set(learner_keys) - set(public_learners)
        _live_obs = [observations[a] for a in public_learners if a in observations]
        _zero_obs = (
            np.zeros_like(_live_obs[0])
            if _live_obs
            else np.zeros(
                self.inner.observation_space(self.possible_agents[0]).shape,
                dtype=np.float32,
            )
        )
        out_obs = {a: observations.get(a, _zero_obs) for a in learner_keys}
        out_rewards: Dict[str, float] = {
            a: float(rewards.get(a, 0.0)) for a in learner_keys
        }

        # Catch penalty: with keep_prey_count_constant=False the inner env
        # gives reward=0 on the catch step (the caught prey is removed from
        # self.prey before get_prey_rewards() iterates); read prey_punishment
        # from the raw env and inject it here.
        caught_this_step: List[str] = [
            a for a in public_learners
            if bool(terminations.get(a, False)) and not bool(truncations.get(a, False))
        ]
        recently_died = len(caught_this_step)
        if recently_died > 0:
            self.prey_lost += recently_died
            punishment = 0.0
            if self._raw_env is not None:
                punishment = float(getattr(self._raw_env, "prey_punishment", 0.0))
            for a in caught_this_step:
                out_rewards[a] -= punishment

        # Distance shaping (prey perspective): reward opening distance from
        # the nearest FOV-visible predator; optional close-range penalty.
        if self.reward_shape_alpha > 0.0 and self._raw_env is not None:
            try:
                current_min_dists = self._compute_min_pred_distances(active_learners)
                for agent_id in active_learners:
                    cur = current_min_dists.get(agent_id)
                    prev = self._prev_min_pred_distances.get(agent_id)
                    if cur is None or prev is None:
                        continue
                    progress = cur - prev   # positive when prey moved AWAY
                    out_rewards[agent_id] += self.reward_shape_alpha * (progress / 200.0)

                    if self.close_range_penalty > 0.0:
                        arena_scale = float(
                            max(
                                getattr(self._raw_env, "width", 2000),
                                getattr(self._raw_env, "height", 2000),
                            )
                        )
                        d_norm = cur / max(arena_scale, 1e-9)
                        out_rewards[agent_id] -= self.close_range_penalty * float(
                            np.exp(-(d_norm * d_norm) / max(self.close_range_sigma2, 1e-9))
                        )
                self._prev_min_pred_distances = current_min_dists
            except Exception:
                pass

        out_terms = {
            a: bool(terminations.get(a, False)) or a in inactive_learners
            for a in learner_keys
        }
        out_truncs = {a: bool(truncations.get(a, False)) for a in learner_keys}
        out_infos = {a: dict(infos.get(a, {})) for a in learner_keys}

        # Episode-end: no learner-side agents remain (all prey caught).
        all_caught = not any(
            a.startswith(self.learner_prefix) for a in self.inner.agents
        )
        if all_caught:
            out_terms = {a: True for a in learner_keys}

        team_reward = sum(out_rewards.values())
        self.last_team_reward = team_reward

        prey_remaining = len(active_learners)
        if learner_keys:
            head = out_infos.setdefault(learner_keys[0], {})
            head.setdefault("prey_lost_total", int(self.prey_lost))
            head.setdefault("prey_remaining", int(prey_remaining))
            head.setdefault("team_reward_step", float(team_reward))

        return out_obs, out_rewards, out_terms, out_truncs, out_infos

    # Internal logic

    def _fallback_action(self):
        """Mode-appropriate no-op action for scripted agents."""
        if self._continuous:
            return np.zeros(2, dtype=np.float32)
        return 0

    def _compute_min_pred_distances(
        self, learner_agents: List[str]
    ) -> Dict[str, Optional[float]]:
        """Return nearest FOV-visible predator torus distance per prey."""
        if self._raw_env is None:
            return {}
        all_preds = list(getattr(self._raw_env, "predators", []))
        all_prey = list(getattr(self._raw_env, "prey", []))
        torus = getattr(self._raw_env, "torus", None)
        if not all_preds or not all_prey or torus is None:
            return {}
        prey_by_name = {p.id(): p for p in all_prey if hasattr(p, "id")}

        fov_enabled: bool = bool(getattr(self._raw_env, "fov_enabled", False))
        view_dist = getattr(self._raw_env, "prey_view_distance", 2000)
        fov = getattr(self._raw_env, "prey_fov", 360)
        check = getattr(torus, "check_if_entity_is_in_view_in_torus", None)

        min_dists: Dict[str, Optional[float]] = {}
        for agent_id in learner_agents:
            prey = prey_by_name.get(agent_id)
            if prey is None:
                continue
            if fov_enabled and check is not None:
                visible = [pr for pr in all_preds if check(prey, pr, view_dist, fov)]
            else:
                visible = all_preds
            if not visible:
                min_dists[agent_id] = None
            else:
                min_dists[agent_id] = min(
                    float(torus.get_distance_in_torus(prey.position, pr.position))
                    for pr in visible
                )
        return min_dists

    def render(self, mode: Optional[str] = None):  # pragma: no cover - viz
        try:
            return self.inner.render()
        except SystemExit:
            return None

    @property
    def unwrapped(self):
        return self

    @property
    def render_mode(self):
        return getattr(self.inner, "render_mode", None)

    def close(self) -> None:
        try:
            self.inner.close()
        except SystemExit:
            pass

    def state(self) -> np.ndarray:  # pragma: no cover
        try:
            return np.asarray(self.inner.state())
        except (AttributeError, NotImplementedError):
            return np.zeros(0)


class _ScriptedPredatorAquariumFactory:
    """Picklable factory for a registered scripted-predator aquarium env."""

    def __init__(
        self,
        env_config: Dict[str, Any],
        apf_config: AquariumAPFConfig,
        reward_shape_alpha: float = 0.0,
        close_range_penalty: float = 0.0,
        close_range_sigma2: float = 0.01,
    ) -> None:
        self.env_config = dict(env_config)
        self.apf_config = apf_config
        self.reward_shape_alpha = float(reward_shape_alpha)
        self.close_range_penalty = float(close_range_penalty)
        self.close_range_sigma2 = float(close_range_sigma2)

    def __call__(self, render_mode: Optional[str] = None, **kwargs):
        from marl_aquarium import aquarium_v0  # local import — optional dep

        cfg = {**self.env_config, **kwargs}
        if render_mode is None:
            render_mode = "rgb_array"
        inner = aquarium_v0.parallel_env(render_mode=render_mode, **cfg)
        _patch_aquarium_wrappers(inner)
        return AquariumScriptedPredatorWrapper(
            inner,
            predator_observe_count=int(cfg["predator_observe_count"]),
            prey_observe_count=int(cfg["prey_observe_count"]),
            action_count=int(cfg.get("action_count", 16)),
            apf_config=self.apf_config,
            reward_shape_alpha=self.reward_shape_alpha,
            close_range_penalty=self.close_range_penalty,
            close_range_sigma2=self.close_range_sigma2,
        )


def register_scripted_predator_aquarium(
    env_name: str,
    env_config: Dict[str, Any],
    apf_config: Optional[AquariumAPFConfig] = None,
    *,
    reward_shape_alpha: float = 0.0,
    close_range_penalty: float = 0.0,
    close_range_sigma2: float = 0.01,
    wrappers: Sequence[Callable[[Any], Any]] = (),
    overwrite: bool = True,
) -> str:
    """Register a scripted-predator aquarium env (prey are the learners).

    Role-flipped counterpart of :func:`register_scripted_aquarium`; see
    that function for the shared parameter semantics.  The extra knobs
    control prey-side shaping (distance shaping + close-range penalty).
    """
    if apf_config is None:
        apf_config = AquariumAPFConfig()
    factory = _ScriptedPredatorAquariumFactory(
        env_config=env_config,
        apf_config=apf_config,
        reward_shape_alpha=reward_shape_alpha,
        close_range_penalty=close_range_penalty,
        close_range_sigma2=close_range_sigma2,
    )
    if is_registered(env_name) and not overwrite:
        return env_name
    register_env(
        env_name,
        factory,
        default_episode_len=int(env_config.get("max_time_steps", 3000)),
        continuous_default=bool(env_config.get("continuous_actions", False)),
        notes=(
            "Scripted-predator wrapper around marl_aquarium.aquarium_v0; "
            "exposes only prey agents."
        ),
        tags={"aquarium", "scripted_opponent", "evader_training", "third_party"},
        api="parallel",
        wrappers=tuple(wrappers),
        overwrite=True,
    )
    return env_name


# Compatibility adapter — strip infos from reset() for deap-bridge code paths

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

