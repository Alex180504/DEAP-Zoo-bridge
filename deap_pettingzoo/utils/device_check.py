"""GPU contention detection for mixed-paradigm co-evolution runs.

When an :class:`RLSideState` (torch / CUDA) coexists with a
:class:`QDaxSideState` (jax / GPU) inside one
:class:`CoEvolutionTrainer.train` call, both can race for the same
device. This module provides a small detector that the trainer calls
once per ``train()`` to emit a single :class:`RuntimeWarning` listing
the offending side names. No automatic downgrade — the user retains
control over per-side ``device`` configuration.
"""

from __future__ import annotations

import warnings
from typing import Sequence

_warned_devices: set = set()


def _torch_cuda_available() -> bool:
    """Best-effort check for a usable CUDA torch build. Never raises."""
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover — torch missing / broken
        return False


def cuda_available() -> bool:
    """Public alias of :func:`_torch_cuda_available`.

    Used by the catalog to gate the ``"cuda"`` choice of the ``device``
    :class:`~deap_pettingzoo.catalog.schema.ParamSpec` so the Setup UI never
    offers a device that would silently fall back (see :func:`resolve_device`).
    """
    return _torch_cuda_available()


def resolve_device(requested: object) -> str:
    """Resolve a requested torch device to one that actually works.

    This guards against the common containerised-deployment failure where an
    experiment (or a pasted script) asks for ``"cuda"`` but the backend ships a
    **CPU-only** torch build, prompting AgileRL to crash with
    ``AssertionError: Torch not compiled with CUDA enabled``. Instead of
    crashing, fall back to CPU (with a one-time warning).

    Rules:
    * ``None`` -> auto: ``"cuda"`` if available, else ``"cpu"``.
    * a CUDA device (e.g. ``"cuda"`` / ``"cuda:0"``) when CUDA is unavailable
      -> ``"cpu"`` (+ one :class:`RuntimeWarning` per distinct requested device).
    * anything else -> ``str(requested)`` unchanged.
    """
    cuda_ok = _torch_cuda_available()
    if requested is None:
        return "cuda" if cuda_ok else "cpu"
    dev = str(requested)
    if torch_uses_cuda(dev) and not cuda_ok:
        if dev not in _warned_devices:
            _warned_devices.add(dev)
            warnings.warn(
                f"Requested torch device {dev!r} but CUDA is not available in "
                "this environment (e.g. a CPU-only torch build in the service "
                "container). Falling back to 'cpu'. Training will run but more "
                "slowly; for GPU, build the CUDA service image and enable GPU "
                "access (see Dockerfile.server).",
                RuntimeWarning,
                stacklevel=2,
            )
        return "cpu"
    return dev


def torch_uses_cuda(device: object) -> bool:
    """Return ``True`` iff ``device`` resolves to a CUDA device.

    Accepts strings (``"cuda"``, ``"cuda:0"``, ``"cpu"``), ``torch.device``
    instances, or ``None``. Anything not parseable as CUDA returns
    ``False`` — the helper is deliberately conservative so an unknown
    device string never triggers a spurious warning.
    """
    if device is None:
        return False
    text = str(device).strip().lower()
    return text.startswith("cuda")


def jax_uses_gpu() -> bool:
    """Return ``True`` iff jax has a GPU backend available.

    Soft-imports jax. When jax is not installed (no QDax extra), no
    contention is possible, so we return ``False``. Any failure
    interrogating jax devices (driver missing, CUDA mismatch, ...)
    also returns ``False`` — the detector should never raise.
    """
    try:
        import jax  # type: ignore
    except ImportError:
        return False
    try:
        backend = jax.default_backend()
    except Exception:  # pragma: no cover — driver-dependent
        return False
    if isinstance(backend, str) and backend.lower() == "gpu":
        return True
    try:
        devices = jax.devices()
    except Exception:  # pragma: no cover — driver-dependent
        return False
    for d in devices:
        platform = getattr(d, "platform", "").lower()
        if platform in {"gpu", "cuda", "rocm"}:
            return True
    return False


def warn_on_gpu_contention(
    rl_cuda_sides: Sequence[str],
    qdax_sides: Sequence[str],
) -> bool:
    """Emit one :class:`RuntimeWarning` when both groups are non-empty.

    Args:
        rl_cuda_sides: Names of RL sides whose torch device resolved to
            a CUDA device.
        qdax_sides: Names of QDax sides present in the run (regardless
            of resolved jax backend — this helper does the
            :func:`jax_uses_gpu` check itself).

    Returns:
        ``True`` if a warning was emitted, ``False`` otherwise. Useful
        for tests.
    """
    if not rl_cuda_sides or not qdax_sides:
        return False
    if not jax_uses_gpu():
        return False
    rl_list = ", ".join(repr(n) for n in rl_cuda_sides)
    qdax_list = ", ".join(repr(n) for n in qdax_sides)
    warnings.warn(
        "Co-evolution GPU contention detected: RL side(s) "
        f"{rl_list} resolved to a CUDA torch device while QDax side(s) "
        f"{qdax_list} are running on a jax GPU backend. Both will "
        "compete for the same accelerator. Recommend setting "
        "optimizer_config['device']='cpu' on one of them, or running "
        "the QDax side on the numpy scoring path.",
        RuntimeWarning,
        stacklevel=2,
    )
    return True


__all__ = [
    "torch_uses_cuda",
    "resolve_device",
    "jax_uses_gpu",
    "warn_on_gpu_contention",
    "cuda_available",
]
