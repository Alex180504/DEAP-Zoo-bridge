"""Deprecated shim. Import from :mod:`deap_pettingzoo.contrib.pursuit_evasion`."""

from __future__ import annotations

import warnings

warnings.warn(
    "deap_pettingzoo.environments.pursuit_evasion has moved to "
    "deap_pettingzoo.contrib.pursuit_evasion. Update your imports; "
    "the old location will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from deap_pettingzoo.contrib.pursuit_evasion.env import (  # noqa: F401,E402
    PursuitEvasionEnv,
    PursuitEvasionConfig,
    ObsLayout,
)

__all__ = ["PursuitEvasionEnv", "PursuitEvasionConfig", "ObsLayout"]
