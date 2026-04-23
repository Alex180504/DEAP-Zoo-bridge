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


# ---------------------------------------------------------------------------
# MPE2 (required dependency as of deap-bridge's pyproject.toml, but still
# wrapped in try/except so environments/__init__.py never hard-crashes).
# ---------------------------------------------------------------------------

# Per-env notes.  Keys that may be truly heterogeneous across agents or use
# non-Box/Discrete action spaces are flagged so downstream code (policies,
# compatibility checks, tests) can surface meaningful errors instead of
# obscure shape mismatches.
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


def _try_register_aquarium() -> None:
    """Register ``aquarium_v0`` if ``marl-aquarium`` is installed."""
    try:  # pragma: no cover - depends on optional extra
        from marl_aquarium import aquarium_v0  # type: ignore
    except ImportError:  # pragma: no cover
        return
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
