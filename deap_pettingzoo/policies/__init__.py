"""Policy implementations for converting genomes to agent behaviors.

``APFEvaderPolicy`` / ``APFConfig`` used to live in this package.  They
moved to :mod:`deap_pettingzoo.contrib.pursuit_evasion`; a deprecation
shim at ``deap_pettingzoo.policies.apf_evader`` forwards them for one
release.  New code should import from the contrib subpackage.
"""

from .feedforward_neural import FeedforwardNeuralPolicy
from .action_heads import ActionHead, BoxHead, DiscreteHead, make_head

__all__ = [
    'FeedforwardNeuralPolicy',
    'ActionHead',
    'BoxHead',
    'DiscreteHead',
    'make_head',
]


# PEP 562: lazy, deprecation-warning-emitting forwarder for the APF names.
def __getattr__(name):  # pragma: no cover - trivial forwarder
    if name in ("APFEvaderPolicy", "APFConfig"):
        import warnings
        warnings.warn(
            f"{name} moved to deap_pettingzoo.contrib.pursuit_evasion; "
            "update your imports.",
            DeprecationWarning,
            stacklevel=2,
        )
        from deap_pettingzoo.contrib.pursuit_evasion.apf_policy import (
            APFEvaderPolicy,
            APFConfig,
        )
        return {"APFEvaderPolicy": APFEvaderPolicy, "APFConfig": APFConfig}[name]
    raise AttributeError(f"module 'deap_pettingzoo.policies' has no attribute {name!r}")
