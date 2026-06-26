"""Deterministic seeding helpers for reproducibility.

Provides a single :func:`seed_everything` call that seeds Python's
``random``, NumPy's global RNG, and (when available) PyTorch's CPU/CUDA
RNGs, with an opt-in deterministic-algorithms flag.

Usage::

    from deap_pettingzoo.utils.rng import seed_everything
    seed_everything(seed)
"""

from __future__ import annotations

import random as _random
from typing import Optional

import numpy as _np


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch global RNGs.

    Args:
        seed: Integer seed value.
        deterministic: If ``True``, additionally enable PyTorch
            deterministic algorithms (``torch.use_deterministic_algorithms``
            with ``warn_only=True``) and cuDNN determinism.  Has no effect
            when PyTorch is not installed.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    _random.seed(seed)
    _np.random.seed(seed)

    _seed_torch(seed, deterministic=deterministic)


def _seed_torch(seed: int, *, deterministic: bool = False) -> None:
    """Seed PyTorch RNGs if available (no-op otherwise)."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:  # pragma: no cover — older PyTorch
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
