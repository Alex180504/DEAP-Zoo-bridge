"""Auto-registration of built-in / optional PettingZoo environments.

Imported for side-effects by :mod:`deap_pettingzoo.environments`.  It tries
to register every MPE2 environment and, if installed, ``marl-aquarium``'s
``aquarium_v0``.  Import failures are swallowed silently so missing
optional extras never break the core package.

Third-party envs should **not** live here — register them at the call site
with :func:`~deap_pettingzoo.environments.registry.register_env`.
"""

from __future__ import annotations

from .registry import is_registered, register_env

"""
MPE2 registration metadata.

MPE2 is currently a required dependency in deap-bridge's ``pyproject.toml``,
but registration stays wrapped in ``try/except`` so
``environments/__init__.py`` never hard-crashes.

``_MPE_NOTES`` flags environments that may be heterogeneous across agents or
use non-Box/Discrete action spaces, allowing downstream policies,
compatibility checks, and tests to raise clear errors instead of obscure shape
mismatches.
"""
_MPE_NOTES = {
    "simple_v3": "",
    "simple_adversary_v3": (
        "Heterogeneous agents (adversary vs good). FeedforwardNeuralPolicy "
        "requires one genome per role; use MixedPolicyEvaluator."
    ),
    "simple_crypto_v3": (
        "Heterogeneous agents (alice/bob/eve) with communication channels."
    ),
    "simple_push_v3": (
        "Heterogeneous agents (adversary vs good). One genome per role."
    ),
    "simple_reference_v3": (
        "MultiDiscrete action space (movement + communication). "
        "FeedforwardNeuralPolicy does NOT support MultiDiscrete — use "
        "AgileRL trainers."
    ),
    "simple_speaker_listener_v3": (
        "Heterogeneous: speaker has Discrete comm-only action, listener has "
        "Discrete movement action; observation dims differ. One genome per "
        "role required."
    ),
    "simple_spread_v3": "",
    "simple_tag_v3": "",
    "simple_world_comm_v3": (
        "MultiDiscrete action space and heterogeneous agents. "
        "FeedforwardNeuralPolicy does NOT support MultiDiscrete — use "
        "AgileRL trainers."
    ),
}


def _try_register_mpe2() -> None:
    """Register every available MPE2 parallel environment."""
    import importlib

    try:
        importlib.import_module("mpe2")
    except ImportError:  # pragma: no cover
        return

    mpe_envs = [
        "simple_v3",
        "simple_adversary_v3",
        "simple_crypto_v3",
        "simple_push_v3",
        "simple_reference_v3",
        "simple_speaker_listener_v3",
        "simple_spread_v3",
        "simple_tag_v3",
        "simple_world_comm_v3",
    ]
    for env_name in mpe_envs:
        if is_registered(env_name):
            continue
        try:
            module = importlib.import_module(f"mpe2.{env_name}")
        except ImportError:  # pragma: no cover - partial install
            continue
        if not callable(getattr(module, "parallel_env", None)):  # pragma: no cover
            continue
        register_env(
            env_name,
            module,
            default_episode_len=25,  # MPE default max_cycles
            continuous_default=False,
            notes=_MPE_NOTES.get(env_name, ""),
            tags={"mpe2"},
        )


def _patch_aquarium() -> None:  # pragma: no cover
    """Monkey-patch a bug in ``marl_aquarium.env.aquarium.raw_env``.

    ``prey_nearby_sharks_observations`` uses a wrong padding formula when
    fewer sharks than ``predator_observe_count`` are in view::

        # BUGGY (operator-precedence error):
        observations += [0] * self.obs_size * (n_nearest_shark - len(observations))

        # CORRECT (matches the sibling prey_nearby_fish_observations):
        observations += [0] * (n_nearest_shark * self.obs_size - len(observations))

    The FOV branch also lacked the per-observation cap guard present in the
    fish sibling.  Both issues cause an ``AssertionError`` whenever
    ``fov_enabled=True`` and fewer sharks are visible than
    ``predator_observe_count``.

    This patch is applied once at import time so it affects every
    ``raw_env`` instance created afterwards.
    """
    try:
        from marl_aquarium.env.aquarium import raw_env  # type: ignore
    except ImportError:
        return

    from typing import List, Sequence

    def _fixed_prey_nearby_sharks_observations(
        self, observer, all_sharks: Sequence, n_nearest_shark: int
    ) -> List[float]:
        observations: List[float] = []
        if self.fov_enabled:
            for shark in all_sharks:
                if (
                    len(observations) < n_nearest_shark * self.obs_size
                    and self.torus.check_if_entity_is_in_view_in_torus(
                        observer, shark, self.prey_view_distance, self.prey_fov
                    )
                ):
                    observations += self.nearby_animal_observation(observer, shark)
        else:
            for shark in self.prey_get_n_closest_animals(
                observer, all_sharks, self.predator_observe_count
            ):
                observations += self.nearby_animal_observation(observer, shark)

        if len(observations) < n_nearest_shark * self.obs_size:
            observations += [0] * (n_nearest_shark * self.obs_size - len(observations))

        assert len(observations) == n_nearest_shark * self.obs_size
        return observations

    def _fixed_predator_nearby_shark_observations(self, observer) -> List[float]:
        # Bug: original code has no `else` before the non-FOV loop, so when
        # fov_enabled=True both branches run and observations exceed the expected
        # predator_observe_count * obs_size, triggering the assert in
        # get_predator_observations.  Fix: make the non-FOV loop properly `else`.
        observations: List[float] = []
        if self.fov_enabled:
            for shark in self.predators:
                if (
                    len(observations) < self.predator_observe_count * self.obs_size
                    and self.torus.check_if_entity_is_in_view_in_torus(
                        observer, shark, self.prey_view_distance, self.prey_fov
                    )
                ):
                    observations += self.nearby_animal_observation(observer, shark)
        else:
            for shark in self.predators:
                if shark is not observer:
                    observations += self.nearby_animal_observation(observer, shark)

        if len(observations) < self.predator_observe_count * self.obs_size:
            observations += [0] * (
                self.predator_observe_count * self.obs_size - len(observations)
            )
        return observations

    def _fixed_predator_nearby_fish_observations(self, observer) -> List[float]:
        # Bug: the FOV branch had no cap guard, so if more fish than
        # prey_observe_count are in view the list grows beyond the expected
        # size and breaks the assert in get_predator_observations.
        observations: List[float] = []
        if self.fov_enabled:
            for fish in self.prey:
                if (
                    len(observations) < self.prey_observe_count * self.obs_size
                    and self.torus.check_if_entity_is_in_view_in_torus(
                        observer, fish, self.prey_view_distance, self.prey_fov
                    )
                ):
                    observations += self.nearby_animal_observation(observer, fish)
        else:
            closest_fish = self.predator_get_n_closest_fish(observer)
            for fish in closest_fish:
                if fish is not observer:
                    observations += self.nearby_animal_observation(observer, fish)

        if len(observations) < self.prey_observe_count * self.obs_size:
            observations += [0] * (
                self.prey_observe_count * self.obs_size - len(observations)
            )
        return observations

    raw_env.prey_nearby_sharks_observations = _fixed_prey_nearby_sharks_observations
    raw_env.predator_nearby_shark_observations = _fixed_predator_nearby_shark_observations
    raw_env.predator_nearby_fish_observations = _fixed_predator_nearby_fish_observations


def _try_register_aquarium() -> None:
    """Register ``aquarium_v0`` if ``marl-aquarium`` is installed."""
    try:  # pragma: no cover - depends on optional extra
        from marl_aquarium import aquarium_v0  # type: ignore
    except ImportError:  # pragma: no cover
        return
    _patch_aquarium()  # pragma: no cover
    if is_registered("aquarium_v0"):  # pragma: no cover
        return
    register_env(  # pragma: no cover
        "aquarium_v0",
        aquarium_v0,
        default_episode_len=3000,
        continuous_default=False,
        notes=(
            "Third-party env. Dynamic agent set (fish may die during "
            "episodes); long episodes."
        ),
        tags={"aquarium", "third_party"},
    )


_try_register_mpe2()
_try_register_aquarium()
