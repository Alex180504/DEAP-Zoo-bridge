"""Advanced-feature gate for the experiment service.

The "advanced" surface — the dependency-install panel (``/env/install``) and the
custom-script submission/execution path (``/experiments/script``) — is arbitrary
code/command execution **by design**, consistent with the documented
trusted-input / localhost model (see ``docs/known_issues.md``).

It is gated behind the ``DEAP_ENABLE_ADVANCED`` environment variable. The default
is **enabled** (so the single-user / localhost workflow works out of the box), but
operators of a shared deployment can set ``DEAP_ENABLE_ADVANCED=0`` to disable the
endpoints (they then return HTTP 403) and hide the Advanced UI section.
"""

from __future__ import annotations

import os

_FALSY = {"0", "false", "no", "off", "disable", "disabled", ""}


def advanced_enabled() -> bool:
    """Return whether the advanced (arbitrary-code) features are enabled.

    Reads ``DEAP_ENABLE_ADVANCED``; defaults to ``True`` when unset. Any of
    ``0 / false / no / off / disable(d)`` (case-insensitive) disables it.
    """
    raw = os.environ.get("DEAP_ENABLE_ADVANCED")
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSY
