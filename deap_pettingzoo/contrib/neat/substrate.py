"""Substrate builders — map environment observation/action spaces to
HyperNEAT substrate coordinates.

A substrate is a geometric (2-D) arrangement of input and output nodes
in ``[-1, 1] × [-1, 1]``.  The CPPN is queried at these coordinates to
determine ANN connection weights.

Convenience builders
--------------------
* :func:`make_uniform_substrate` — equally-spaced 1-D layout.
* :func:`make_2d_grid_substrate` — grid layout (e.g. image-like obs).
* :func:`substrate_for_env` — auto-detect from environment specs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from deap_pettingzoo.contrib.neat._pureples.substrate import (
    Coord,
    Substrate,
)


def make_uniform_substrate(
    num_inputs: int,
    num_outputs: int,
    *,
    hidden_layers: Optional[Sequence[int]] = None,
) -> Substrate:
    """Create a substrate with equally-spaced 1-D coordinate layouts.

    Inputs are placed along the **bottom** edge ``(x=-1..+1, y=-1)``.
    Outputs along the **top** edge ``(x=-1..+1, y=+1)``.
    Hidden nodes (if any) are placed on horizontal lines between them.

    Parameters
    ----------
    num_inputs:
        Number of input features.
    num_outputs:
        Number of output units.
    hidden_layers:
        List of layer sizes, e.g. ``[4, 4]`` for two hidden layers of
        4 nodes each.  ``None`` means no hidden nodes (ES-HyperNEAT
        discovers them automatically).

    Returns
    -------
    Substrate
    """
    input_coords = _linspace_1d(num_inputs, y=-1.0)
    output_coords = _linspace_1d(num_outputs, y=1.0)

    hidden_coords: List[List[Coord]] = []
    if hidden_layers:
        n_layers = len(hidden_layers)
        for i, size in enumerate(hidden_layers):
            y = -1.0 + 2.0 * (i + 1) / (n_layers + 1)
            hidden_coords.append(_linspace_1d(size, y=y))

    return Substrate(input_coords, output_coords, hidden_coords)


def make_2d_grid_substrate(
    grid_shape: Tuple[int, int],
    num_outputs: int,
    *,
    hidden_layers: Optional[Sequence[int]] = None,
) -> Substrate:
    """Create a substrate for 2-D grid observations (e.g. images).

    Input coordinates are arranged in an ``(rows, cols)`` grid centred
    at the origin with aspect-ratio preserved.

    Parameters
    ----------
    grid_shape:
        ``(rows, cols)`` of the input grid.
    num_outputs:
        Number of output units.
    hidden_layers:
        Optional hidden layer sizes.

    Returns
    -------
    Substrate
    """
    rows, cols = grid_shape
    input_coords: List[Coord] = []
    for r in range(rows):
        y = 1.0 - 2.0 * r / max(1, rows - 1) if rows > 1 else 0.0
        for c in range(cols):
            x = -1.0 + 2.0 * c / max(1, cols - 1) if cols > 1 else 0.0
            input_coords.append((x, y))

    output_coords = _linspace_1d(num_outputs, y=1.0)

    hidden_coords: List[List[Coord]] = []
    if hidden_layers:
        n_layers = len(hidden_layers)
        for i, size in enumerate(hidden_layers):
            y = -0.5 + 1.0 * (i + 1) / (n_layers + 1)
            hidden_coords.append(_linspace_1d(size, y=y))

    return Substrate(input_coords, output_coords, hidden_coords)


def substrate_for_env(
    env_obs_size: int,
    env_act_size: int,
    *,
    layout: str = "uniform",
    hidden_layers: Optional[Sequence[int]] = None,
) -> Substrate:
    """Auto-select a substrate layout from environment dimensions.

    Parameters
    ----------
    env_obs_size:
        Number of observation features.
    env_act_size:
        Number of action units.
    layout:
        ``"uniform"`` (default) or ``"2d_grid"``.
    hidden_layers:
        Optional hidden layer sizes.

    Returns
    -------
    Substrate
    """
    if layout == "2d_grid":
        # Estimate a square-ish grid.
        import math

        side = int(math.ceil(math.sqrt(env_obs_size)))
        return make_2d_grid_substrate(
            (side, side), env_act_size, hidden_layers=hidden_layers
        )
    return make_uniform_substrate(
        env_obs_size, env_act_size, hidden_layers=hidden_layers
    )


# Internal helpers


def _linspace_1d(n: int, y: float) -> List[Coord]:
    """Return *n* equally-spaced coordinates along a line at height *y*."""
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, y)]
    return [(-1.0 + 2.0 * i / (n - 1), y) for i in range(n)]
