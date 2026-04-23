"""Deprecated shim. Import from :mod:`deap_pettingzoo.contrib.pursuit_evasion`."""

from __future__ import annotations

import warnings

warnings.warn(
    "deap_pettingzoo.policies.apf_evader has moved to "
    "deap_pettingzoo.contrib.pursuit_evasion.apf_policy. Update your imports; "
    "the old location will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from deap_pettingzoo.contrib.pursuit_evasion.apf_policy import (  # noqa: F401,E402
    APFEvaderPolicy,
    APFConfig,
)

__all__ = ["APFEvaderPolicy", "APFConfig"]
