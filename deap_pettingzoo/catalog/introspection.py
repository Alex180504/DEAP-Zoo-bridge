"""Hybrid hyperparameter discovery helpers.

The catalog discovers an algorithm's hyperparameters using a **hybrid**
strategy (see the UI-overhaul plan):

* **Introspection** supplies the *names* and *defaults* — read from
  ``inspect.signature`` of the trainer constructor and/or a class-level
  ``DEFAULT_PARAMS`` mapping. This keeps defaults honest: they track the code
  automatically.
* A **declarative overlay** (the ``ParamSpec`` tuple a trainer returns from
  ``describe_hyperparameters()``) supplies the *types, ranges, grouping and
  widget hints* that signatures cannot express — MPE2 / AgileRL factories use
  ``**kwargs`` passthrough, so pure introspection is unreliable.

This module provides the small reconciliation utilities that merge the two
sources. The authoritative ``ParamSpec`` list always comes from the overlay;
introspection is used to **fill in / refresh defaults** and to **detect drift**
(an overlay param that no longer matches the constructor default).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from deap_pettingzoo.catalog.schema import ParamSpec


def signature_defaults(func: Callable[..., Any]) -> Dict[str, Any]:
    """Return ``{param_name: default}`` for every defaulted keyword argument.

    Parameters without a default (required positional args) and the
    ``self`` / ``*args`` / ``**kwargs`` entries are skipped. Never raises —
    objects without an inspectable signature yield an empty mapping.
    """

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return {}

    defaults: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is not inspect.Parameter.empty:
            defaults[name] = param.default
    return defaults


def collect_defaults(
    constructor: Optional[Callable[..., Any]] = None,
    default_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge constructor-signature defaults with a ``DEFAULT_PARAMS`` mapping.

    The explicit ``default_params`` mapping (used by the AgileRL trainers,
    whose real knobs live in a class attribute rather than the ``__init__``
    signature) takes precedence over signature-derived defaults.
    """

    merged: Dict[str, Any] = {}
    if constructor is not None:
        merged.update(signature_defaults(constructor))
    if default_params:
        merged.update(dict(default_params))
    return merged


def refresh_defaults(
    specs: Sequence[ParamSpec],
    discovered: Mapping[str, Any],
) -> List[ParamSpec]:
    """Return ``specs`` with each default replaced by the discovered value.

    For every :class:`ParamSpec` whose ``name`` appears in ``discovered`` and
    whose discovered default is coercible to the spec's type, the spec is
    rebuilt with that default. Specs absent from ``discovered`` — or whose
    discovered value cannot be coerced (e.g. ``None`` sentinels, tuples) — are
    passed through unchanged. This keeps hand-authored type/range metadata
    while letting code-level defaults flow through automatically.
    """

    refreshed: List[ParamSpec] = []
    for spec in specs:
        if spec.name in discovered:
            raw = discovered[spec.name]
            try:
                new_default = spec.coerce(raw)
            except ValueError:
                refreshed.append(spec)
                continue
            if new_default != spec.default:
                spec = ParamSpec(
                    name=spec.name,
                    type=spec.type,
                    default=new_default,
                    label=spec.label,
                    help=spec.help,
                    minimum=spec.minimum,
                    maximum=spec.maximum,
                    step=spec.step,
                    choices=spec.choices,
                    group=spec.group,
                    advanced=spec.advanced,
                    target=spec.target,
                    item_type=spec.item_type,
                    nullable=spec.nullable,
                )
        refreshed.append(spec)
    return refreshed


def drift_report(
    specs: Sequence[ParamSpec],
    discovered: Mapping[str, Any],
) -> List[str]:
    """Return human-readable notes where an overlay default disagrees with code.

    Used by the Phase 0 tests as a guard rail: if a trainer constructor
    default changes but the declarative overlay is not updated, the drift is
    surfaced rather than silently shipping a stale default.
    """

    notes: List[str] = []
    for spec in specs:
        if spec.name not in discovered:
            continue
        raw = discovered[spec.name]
        try:
            coerced = spec.coerce(raw)
        except ValueError:
            continue
        if coerced != spec.default:
            notes.append(
                f"{spec.name}: overlay default {spec.default!r} != "
                f"constructor default {coerced!r}"
            )
    return notes
