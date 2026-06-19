"""Honours-project pursuit-evasion research code (MPE simple_tag).

Specialises ``mpe2.simple_tag_v3`` for equal-ability pursuer / evader
experiments with:

* :class:`PursuitEvasionEnv` — :class:`~deap_pettingzoo.environments.PettingZooWrapper`
  subclass that enforces equal ability, applies a boundary penalty, exposes
  a K-nearest ``ObsLayout``, and optionally advances a prey-speed curriculum.
* :class:`PursuitEvasionConfig` — dataclass for the above.
* :class:`ObsLayout` — observation-slice metadata consumed by scripted policies.
* :class:`APFEvaderPolicy` — scripted artificial-potential-field evader.

This is *experiment-side* research code, **not** part of the
env-agnostic ``deap_pettingzoo`` library; it only consumes the library's
generic env registry. Importing this package registers ``pursuit_evasion``
in that registry so it can be used with the generic baseline trainers /
evaluators.
"""

from __future__ import annotations

from .env import PursuitEvasionEnv, PursuitEvasionConfig, ObsLayout
from .apf_policy import APFEvaderPolicy, APFConfig

from deap_pettingzoo.environments.registry import is_registered, register_env


def _register() -> None:
    if is_registered("pursuit_evasion"):
        return
    register_env(
        "pursuit_evasion",
        lambda render_mode=None, **cfg: PursuitEvasionEnv(
            config=PursuitEvasionConfig(**cfg) if cfg else None,
            render_mode=render_mode,
        ).env,
        default_episode_len=200,
        continuous_default=False,
        notes="Honours research env; specialises mpe2.simple_tag_v3.",
        tags={"contrib", "pursuit_evasion"},
    )


_register()


__all__ = [
    "PursuitEvasionEnv",
    "PursuitEvasionConfig",
    "ObsLayout",
    "APFEvaderPolicy",
    "APFConfig",
]
