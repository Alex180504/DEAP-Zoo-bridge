"""Project-wide pytest configuration.

Currently only escalates :class:`RuntimeWarning` to ``error`` for a
small list of modules that would otherwise silently swallow
optimisation-pipeline failures (the original ``agent.learn()`` 2-vs-3
tensor-dim mismatch was hidden by exactly that pattern). Scope is
deliberately *module-scoped*, not project-wide: third-party libraries
(NumPy, JAX, PettingZoo, AgileRL) emit legitimate ``RuntimeWarning``s
that are not bugs in this codebase.
"""

from __future__ import annotations

# Windows-only: importing pyarrow *after* torch in the same process hard
# crashes (access violation in pyarrow.pandas_compat.convert_column) the
# first time pandas touches parquet I/O. Eagerly importing
# pyarrow here, before any test module gets a chance to import torch/jax,
# fixes the DLL init order for the whole session regardless of collection
# order. No-op (and harmless) if pyarrow is not installed.
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

import warnings


# Modules whose RuntimeWarnings are upgraded to errors during the test
# run. Each entry is matched against the ``module`` of the warning
# call site (i.e. where ``warnings.warn`` was invoked from), not
# against the test module.
_STRICT_RUNTIME_WARNING_MODULES: tuple[str, ...] = (
    r"deap_pettingzoo\.baselines\.coevolution\.rl_side",
)


def pytest_configure(config) -> None:  # noqa: D401 - pytest hook
    """Install scoped ``RuntimeWarning -> error`` filters."""
    for module_pattern in _STRICT_RUNTIME_WARNING_MODULES:
        warnings.filterwarnings(
            "error",
            category=RuntimeWarning,
            module=module_pattern,
        )
