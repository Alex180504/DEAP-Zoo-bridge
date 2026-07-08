"""Auto-registration of built-in / optional PettingZoo environments.

Imported for side-effects by :mod:`deap_pettingzoo.environments`.  It tries
to register every MPE2 environment and, if installed, ``marl-aquarium``'s
``aquarium_v0``.  Import failures are swallowed silently so missing
optional extras never break the core package.

Third-party envs should **not** live here — register them at the call site
with :func:`~deap_pettingzoo.environments.registry.register_env`.
"""

from __future__ import annotations

from deap_pettingzoo.catalog.schema import (
    ParamSpec,
    ParamTarget,
    ParamType,
    TeamSpec,
)

from .registry import is_registered, register_env


# MPE2 registration metadata.

# MPE2 is currently a required dependency in deap-bridge's ``pyproject.toml``,
# but registration stays wrapped in ``try/except`` so
#:``environments/__init__.py`` never hard-crashes.

# ``_MPE_NOTES`` flags environments that may be heterogeneous across agents or
# use non-Box/Discrete action spaces, allowing downstream policies,
# compatibility checks, and tests to raise clear errors instead of obscure shape
# mismatches.

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
        "Homogeneous Discrete(50) action space (movement x communication "
        "flattened into a single Discrete). Trainable by all backends."
    ),
    "simple_speaker_listener_v3": (
        "Heterogeneous: speaker has Discrete comm-only action, listener has "
        "Discrete movement action; observation dims differ. One genome per "
        "role required."
    ),
    "simple_spread_v3": "",
    "simple_tag_v3": "",
    "simple_world_comm_v3": (
        "Heterogeneous Discrete action spaces across roles "
        "(leadadversary/adversary/agent differ). FeedforwardNeuralPolicy "
        "requires one genome per role; use MixedPolicyEvaluator or an "
        "AgileRL trainer."
    ),
}


# Declarative env-config schemas + team metadata (for the catalog / setup UI).

# MPE2 ``parallel_env`` factories use ``**kwargs`` passthrough, so the real
# constructor parameters cannot be introspected reliably (see
# ``deap_pettingzoo/catalog/introspection.py``). The schemas below are
# hand-authored from the verified ``raw_env.__init__`` signatures.

_EC = ParamTarget.ENV_CONFIG
_INT, _FLOAT, _BOOL = ParamType.INT, ParamType.FLOAT, ParamType.BOOL


def _mpe_common_params() -> list:
    """Return the env-config params shared by every MPE2 environment."""
    return [
        ParamSpec(
            "max_cycles",
            _INT,
            25,
            minimum=1,
            step=1,
            target=_EC,
            group="Episode",
            help="Maximum steps before the episode truncates.",
        ),
        ParamSpec(
            "continuous_actions",
            _BOOL,
            False,
            target=_EC,
            group="Actions",
            help="Use Box (continuous) actions instead of Discrete.",
        ),
        ParamSpec(
            "dynamic_rescaling",
            _BOOL,
            False,
            target=_EC,
            advanced=True,
            group="Actions",
            help="Dynamically rescale entity sizes / positions.",
        ),
    ]


#: Per-env extra constructor params (agent / landmark counts, ratios).
_MPE_EXTRA_PARAMS = {
    "simple_adversary_v3": [
        ParamSpec(
            "N",
            _INT,
            2,
            minimum=1,
            step=1,
            target=_EC,
            group="Agents",
            help="Number of good agents (and landmarks).",
        ),
    ],
    "simple_reference_v3": [
        ParamSpec(
            "local_ratio",
            _FLOAT,
            0.5,
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            target=_EC,
            group="Reward",
            help="Weight of local vs global reward.",
        ),
    ],
    "simple_spread_v3": [
        ParamSpec(
            "N",
            _INT,
            3,
            minimum=1,
            step=1,
            target=_EC,
            group="Agents",
            help="Number of agents (and landmarks).",
        ),
        ParamSpec(
            "local_ratio",
            _FLOAT,
            0.5,
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            target=_EC,
            group="Reward",
            help="Weight of local vs global reward.",
        ),
    ],
    "simple_tag_v3": [
        ParamSpec(
            "num_good",
            _INT,
            1,
            minimum=1,
            step=1,
            target=_EC,
            group="Agents",
            help="Number of prey (good) agents.",
        ),
        ParamSpec(
            "num_adversaries",
            _INT,
            3,
            minimum=1,
            step=1,
            target=_EC,
            group="Agents",
            help="Number of predator (adversary) agents.",
        ),
        ParamSpec(
            "num_obstacles",
            _INT,
            2,
            minimum=0,
            step=1,
            target=_EC,
            group="Agents",
            help="Number of static obstacles.",
        ),
    ],
    "simple_world_comm_v3": [
        ParamSpec(
            "num_good",
            _INT,
            2,
            minimum=1,
            step=1,
            target=_EC,
            group="Agents",
            help="Number of good agents.",
        ),
        ParamSpec(
            "num_adversaries",
            _INT,
            4,
            minimum=1,
            step=1,
            target=_EC,
            group="Agents",
            help="Number of adversaries.",
        ),
        ParamSpec(
            "num_obstacles",
            _INT,
            1,
            minimum=0,
            step=1,
            target=_EC,
            group="Agents",
            help="Number of static obstacles.",
        ),
        ParamSpec(
            "num_food",
            _INT,
            2,
            minimum=0,
            step=1,
            target=_EC,
            group="Agents",
            help="Number of food landmarks.",
        ),
        ParamSpec(
            "num_forests",
            _INT,
            2,
            minimum=0,
            step=1,
            target=_EC,
            group="Agents",
            help="Number of forest landmarks.",
        ),
    ],
}


#: Declared teams for competitive MPE2 envs (predator/prey-style).
#: Cooperative and heterogeneous-non-competitive envs declare no teams (single group).
_MPE_TEAMS = {
    "simple_tag_v3": (
        TeamSpec("predators", ("adversary",), role="predator"),
        TeamSpec("prey", ("agent",), role="prey"),
    ),
    "simple_adversary_v3": (
        TeamSpec("adversaries", ("adversary",), role="predator"),
        TeamSpec("good", ("agent",), role="prey"),
    ),
    "simple_push_v3": (
        TeamSpec("adversaries", ("adversary",), role="predator"),
        TeamSpec("good", ("agent",), role="prey"),
    ),
    "simple_world_comm_v3": (
        TeamSpec("adversaries", ("adversary", "leadadversary"), role="predator"),
        TeamSpec("good", ("agent",), role="prey"),
    ),
}


def _mpe_params(env_name: str) -> tuple:
    """Build the full env-config ParamSpec tuple for an MPE2 environment."""
    return tuple(_MPE_EXTRA_PARAMS.get(env_name, []) + _mpe_common_params())


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
            params=_mpe_params(env_name),
            teams=_MPE_TEAMS.get(env_name, ()),
        )


def _patch_aquarium() -> None:  # pragma: no cover
    """Check that marl-aquarium >= v0.1.11 is installed (FOV bugs fixed upstream).

    The former monkey-patches for observation padding, FOV cap guards, and
    predator sensor-param fixes are no longer needed — the fork
    (Alex180504/marl-aquarium) includes these fixes natively.

    Emits a ``RuntimeWarning`` if an older version is detected, since the
    known FOV bugs will silently corrupt observations.
    """
    try:
        from marl_aquarium import __version__ as _aq_ver  # type: ignore
    except ImportError:  # pragma: no cover
        return
    # Parse version: "0.1.10" -> (0, 1, 10)
    try:
        _parts = tuple(int(p) for p in str(_aq_ver).split(".")[:3])
    except Exception:
        _parts = (0, 0, 0)
    if _parts < (0, 1, 11):
        import warnings

        warnings.warn(
            f"marl-aquarium v{_aq_ver} has known FOV observation bugs "
            f"(padding formula, missing cap guards, wrong sensor params). "
            f"Upgrade to v0.1.11+ (Alex180504/marl-aquarium) to fix.",
            RuntimeWarning,
            stacklevel=2,
        )


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
        params=(
            ParamSpec(
                "predator_count",
                _INT,
                3,
                minimum=1,
                step=1,
                target=_EC,
                group="Agents",
                help="Number of predator fish.",
            ),
            ParamSpec(
                "prey_count",
                _INT,
                6,
                minimum=1,
                step=1,
                target=_EC,
                group="Agents",
                help="Number of prey fish.",
            ),
            ParamSpec(
                "max_time_steps",
                _INT,
                3000,
                minimum=1,
                step=50,
                target=_EC,
                group="Episode",
                help="Maximum environment steps per episode.",
            ),
            ParamSpec(
                "observable_walls",
                _BOOL,
                False,
                target=_EC,
                advanced=True,
                group="Topology",
                help="Rectangle (clamped) topology with wall features "
                "instead of torus wrap.",
            ),
        ),
        teams=(
            TeamSpec("predators", ("predator",), role="predator"),
            TeamSpec("prey", ("prey",), role="prey"),
        ),
    )


_try_register_mpe2()
_try_register_aquarium()
