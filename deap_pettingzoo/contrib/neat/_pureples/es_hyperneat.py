"""
ES-HyperNEAT — evolvable-substrate HyperNEAT via quadtree variance analysis.

Extracted from `pureples.es_hyperneat.es_hyperneat
<https://github.com/ukuleleplayer/pureples/blob/master/pureples/es_hyperneat/es_hyperneat.py>`_
(MIT licensed).

Patches for neat-python ≥ 2.0
------------------------------
* ``neat.activations.ActivationFunctionSet`` used as-is (still available).
* Removed ``draw_es`` import (our integration provides its own
  visualisation; the ``filename`` parameter of
  :meth:`ESNetwork.create_phenotype_network` is accepted but ignored).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import neat
import numpy as np

from .hyperneat import query_cppn
from .substrate import Coord, Substrate


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


class QuadPoint:
    """A node in the ES-HyperNEAT quadtree.

    Attributes
    ----------
    x, y:
        Centre of the quad.
    width:
        Width (and height) of the quad.
    lvl:
        Depth level (root = 1).
    w:
        CPPN-queried weight at this point.
    cs:
        Four child :class:`QuadPoint` instances (or ``None`` at leaves).
    """

    __slots__ = ("x", "y", "width", "lvl", "w", "cs")

    def __init__(self, x: float, y: float, width: float, lvl: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.lvl = lvl
        self.w: float = 0.0
        self.cs: List[Optional[QuadPoint]] = [None, None, None, None]


class Connection:
    """A directed connection between two substrate coordinates.

    Attributes
    ----------
    x1, y1:
        Source coordinate.
    x2, y2:
        Target coordinate.
    weight:
        Connection weight.
    """

    __slots__ = ("x1", "y1", "x2", "y2", "weight")

    def __init__(
        self, x1: float, y1: float, x2: float, y2: float, weight: float
    ) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.weight = weight

    def __hash__(self) -> int:
        return hash((self.x1, self.y1, self.x2, self.y2, self.weight))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            return NotImplemented
        return (
            self.x1 == other.x1
            and self.y1 == other.y1
            and self.x2 == other.x2
            and self.y2 == other.y2
            and self.weight == other.weight
        )

    def __repr__(self) -> str:
        return (
            f"Connection(({self.x1:.3f},{self.y1:.3f})→"
            f"({self.x2:.3f},{self.y2:.3f}), w={self.weight:.3f})"
        )


#: Identity activation — returns input unchanged.  Used for output nodes
#: in ES-HyperNEAT phenotypes so that argmax over DiscreteHead logits has
#: a wide dynamic range (sigmoid would squash all logits to [0, 1]).
def _identity_activation(x: float) -> float:
    return float(x)


# ---------------------------------------------------------------------------
# ES-HyperNEAT parameters
# ---------------------------------------------------------------------------

# Default parameters — "Medium" variant (balanced: enough depth for
# moderate-sized substrates without excessive computational cost).
ES_HYPERNEAT_PARAMS_DEFAULTS: Dict[str, Any] = {
    "initial_depth": 1,
    "max_depth": 2,
    "variance_threshold": 0.03,
    "band_threshold": 0.1,
    "iteration_level": 1,
    "division_threshold": 0.5,
    "max_weight": 8.0,
    "activation": "sigmoid",
    "weight_threshold": 0.1,
}

# "Medium" variant.
ES_HYPERNEAT_PARAMS_MEDIUM: Dict[str, Any] = {
    "initial_depth": 1,
    "max_depth": 2,
    "variance_threshold": 0.03,
    "band_threshold": 0.3,
    "iteration_level": 1,
    "division_threshold": 0.5,
    "max_weight": 8.0,
    "activation": "sigmoid",
    "weight_threshold": 0.1,
}

# "Large" variant.
ES_HYPERNEAT_PARAMS_LARGE: Dict[str, Any] = {
    "initial_depth": 2,
    "max_depth": 3,
    "variance_threshold": 0.03,
    "band_threshold": 0.3,
    "iteration_level": 1,
    "division_threshold": 0.5,
    "max_weight": 8.0,
    "activation": "sigmoid",
    "weight_threshold": 0.1,
}


# ---------------------------------------------------------------------------
# ESNetwork
# ---------------------------------------------------------------------------


class ESNetwork:
    """Evolvable-substrate network constructor.

    Given a CPPN and a substrate (input + output coordinates only —
    hidden nodes are discovered automatically), builds a phenotype
    :class:`~neat.nn.RecurrentNetwork` via the ES-HyperNEAT algorithm.

    Parameters
    ----------
    substrate:
        Substrate with input and output coordinates (hidden_coordinates
        is ignored — ES-HyperNEAT discovers hidden nodes automatically).
    cppn:
        The evolved CPPN.
    params:
        ES-HyperNEAT parameter dict.  See :data:`ES_HYPERNEAT_PARAMS_DEFAULTS`
        for keys and defaults.
    """

    def __init__(
        self,
        substrate: Substrate,
        cppn: neat.nn.FeedForwardNetwork,
        params: Dict[str, Any],
    ) -> None:
        self.substrate = substrate
        self.cppn = cppn
        self.initial_depth = int(params.get("initial_depth", 0))
        self.max_depth = int(params.get("max_depth", 1))
        self.variance_threshold = float(params.get("variance_threshold", 0.03))
        self.band_threshold = float(params.get("band_threshold", 0.3))
        self.iteration_level = int(params.get("iteration_level", 1))
        self.division_threshold = float(params.get("division_threshold", 0.5))
        self.max_weight = float(params.get("max_weight", 8.0))
        self._weight_threshold = float(params.get("weight_threshold", 0.2))
        self.connections: Set[Connection] = set()
        # Number of layers (activations to propagate) = 2^max_depth + 1.
        self.activations: int = 2 ** self.max_depth + 1
        activation_functions = neat.activations.ActivationFunctionSet()
        self.activation = activation_functions.get(
            params.get("activation", "sigmoid")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_phenotype_network(
        self, filename: Optional[str] = None
    ) -> neat.nn.RecurrentNetwork:
        """Create the phenotype network via ES-HyperNEAT.

        Parameters
        ----------
        filename:
            Ignored (accepted for API compatibility with upstream
            pureples; our integration handles visualisation separately).

        Returns
        -------
        neat.nn.RecurrentNetwork
            The phenotype network, ready for ``.activate(inputs)``.
        """
        input_coordinates = self.substrate.input_coordinates
        output_coordinates = self.substrate.output_coordinates

        input_nodes = list(range(len(input_coordinates)))
        output_nodes = list(
            range(len(input_nodes), len(input_nodes) + len(output_coordinates))
        )
        hidden_idx = len(input_coordinates) + len(output_coordinates)

        coordinates: list = []
        indices: list = []
        draw_connections: list = []
        node_evals: list = []
        nodes: dict = {}

        coordinates.extend(input_coordinates)
        coordinates.extend(output_coordinates)
        indices.extend(input_nodes)
        indices.extend(output_nodes)

        # Map input and output coordinates to their IDs.
        coords_to_id: Dict[Coord, int] = dict(zip(coordinates, indices))

        # Run ES-HyperNEAT to discover hidden nodes and connections.
        hidden_nodes, connections = self.es_hyperneat()

        # Map hidden coordinates to their IDs.
        for x, y in hidden_nodes:
            coords_to_id[x, y] = hidden_idx
            hidden_idx += 1

        # Build node_evals from discovered connections.
        for (x, y), idx in coords_to_id.items():
            for c in connections:
                if c.x2 == x and c.y2 == y:
                    draw_connections.append(c)
                    if idx in nodes:
                        initial = nodes[idx]
                        initial.append(
                            (coords_to_id[c.x1, c.y1], c.weight)
                        )
                        nodes[idx] = initial
                    else:
                        nodes[idx] = [
                            (coords_to_id[c.x1, c.y1], c.weight)
                        ]

        for idx, links in nodes.items():
            # ── activation selection ──────────────────────────────────
            # Hidden nodes use the ES-params activation (typically
            # sigmoid) for non-linearity.  Output nodes must use
            # identity (linear); otherwise sigmoid squashes every logit
            # into [0, 1], the argmax sees differences of ~0.01, and
            # the policy cannot learn meaningful action diversity.
            if idx in output_nodes:
                act = _identity_activation
            else:
                act = self.activation
            node_evals.append(
                (idx, act, sum, 0.0, 1.0, links)
            )

        return neat.nn.RecurrentNetwork(
            input_nodes, output_nodes, node_evals
        )

    # ------------------------------------------------------------------
    # Internal algorithm
    # ------------------------------------------------------------------

    @staticmethod
    def get_weights(p: Optional[QuadPoint]) -> List[float]:
        """Recursively collect all weights under a :class:`QuadPoint`."""
        temp: List[float] = []

        def loop(pp: Optional[QuadPoint]) -> None:
            if pp is not None and all(
                child is not None for child in pp.cs
            ):
                for i in range(4):
                    loop(pp.cs[i])
            else:
                if pp is not None:
                    temp.append(pp.w)

        loop(p)
        return temp

    def variance(self, p: Optional[QuadPoint]) -> float:
        """Compute the variance of weights under *p*."""
        if not p:
            return 0.0
        return float(np.var(self.get_weights(p)))  # type: ignore[no-any-return]

    def division_initialization(
        self, coord: Coord, outgoing: bool
    ) -> QuadPoint:
        """Initialize the quadtree by recursive division.

        Divisions happen when the level is below ``initial_depth``, or
        when below ``max_depth`` and the variance exceeds
        ``division_threshold``.
        """
        root = QuadPoint(0.0, 0.0, 1.0, 1)
        q: List[QuadPoint] = [root]

        while q:
            p = q.pop(0)

            p.cs[0] = QuadPoint(
                p.x - p.width / 2.0,
                p.y - p.width / 2.0,
                p.width / 2.0,
                p.lvl + 1,
            )
            p.cs[1] = QuadPoint(
                p.x - p.width / 2.0,
                p.y + p.width / 2.0,
                p.width / 2.0,
                p.lvl + 1,
            )
            p.cs[2] = QuadPoint(
                p.x + p.width / 2.0,
                p.y + p.width / 2.0,
                p.width / 2.0,
                p.lvl + 1,
            )
            p.cs[3] = QuadPoint(
                p.x + p.width / 2.0,
                p.y - p.width / 2.0,
                p.width / 2.0,
                p.lvl + 1,
            )

            for c in p.cs:
                assert c is not None
                c.w = query_cppn(
                    coord,
                    (c.x, c.y),
                    outgoing,
                    self.cppn,
                    self.max_weight,
                    self._weight_threshold,
                )

            if (p.lvl < self.initial_depth) or (
                p.lvl < self.max_depth
                and self.variance(p) > self.division_threshold
            ):
                for child in p.cs:
                    q.append(child)  # type: ignore[arg-type]

        return root

    def pruning_extraction(
        self,
        coord: Coord,
        p: QuadPoint,
        outgoing: bool,
    ) -> None:
        """Determine which connections to express via variance pruning.

        High variance → subdivide further.  Low variance → check
        neighbouring points for band-level connectivity.
        """
        for c in p.cs:
            if c is None:
                continue
            if self.variance(c) > self.variance_threshold:
                self.pruning_extraction(coord, c, outgoing)
            else:
                d_left = abs(
                    c.w
                    - query_cppn(
                        coord,
                        (c.x - p.width, c.y),
                        outgoing,
                        self.cppn,
                        self.max_weight,
                        self._weight_threshold,
                    )
                )
                d_right = abs(
                    c.w
                    - query_cppn(
                        coord,
                        (c.x + p.width, c.y),
                        outgoing,
                        self.cppn,
                        self.max_weight,
                        self._weight_threshold,
                    )
                )
                d_top = abs(
                    c.w
                    - query_cppn(
                        coord,
                        (c.x, c.y - p.width),
                        outgoing,
                        self.cppn,
                        self.max_weight,
                        self._weight_threshold,
                    )
                )
                d_bottom = abs(
                    c.w
                    - query_cppn(
                        coord,
                        (c.x, c.y + p.width),
                        outgoing,
                        self.cppn,
                        self.max_weight,
                        self._weight_threshold,
                    )
                )

                con = None
                if (
                    max(min(d_top, d_bottom), min(d_left, d_right))
                    > self.band_threshold
                ):
                    if outgoing:
                        con = Connection(
                            coord[0], coord[1], c.x, c.y, c.w
                        )
                    else:
                        con = Connection(
                            c.x, c.y, coord[0], coord[1], c.w
                        )
                if (
                    con is not None
                    and con.weight != 0.0
                    and con.y1 < con.y2  # nodes connect upwards only
                    and not (
                        con.x1 == con.x2 and con.y1 == con.y2
                    )
                ):
                    self.connections.add(con)

    def es_hyperneat(
        self,
    ) -> Tuple[Set[Coord], Set[Connection]]:
        """Explore hidden nodes and their connections.

        Returns
        -------
        tuple
            ``(hidden_nodes, connections)`` where *hidden_nodes* is a set
            of ``(x, y)`` coordinates and *connections* is a set of
            :class:`Connection` objects.
        """
        inputs = self.substrate.input_coordinates
        outputs = self.substrate.output_coordinates
        hidden_nodes: Set[Coord] = set()
        unexplored_hidden_nodes: Set[Coord] = set()
        connections1: Set[Connection] = set()
        connections2: Set[Connection] = set()
        connections3: Set[Connection] = set()

        # Explore from inputs.
        for x, y in inputs:
            root = self.division_initialization((x, y), True)
            self.pruning_extraction((x, y), root, True)
            connections1 = connections1.union(self.connections)
            for c in connections1:
                hidden_nodes.add((c.x2, c.y2))
            self.connections = set()

        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)

        # Explore from hidden nodes (iteration_level times).
        for _ in range(self.iteration_level):
            for x, y in unexplored_hidden_nodes:
                root = self.division_initialization((x, y), True)
                self.pruning_extraction((x, y), root, True)
                connections2 = connections2.union(self.connections)
                for c in connections2:
                    hidden_nodes.add((c.x2, c.y2))
                self.connections = set()

            unexplored_hidden_nodes = hidden_nodes - unexplored_hidden_nodes

        # Explore to outputs.
        for x, y in outputs:
            root = self.division_initialization((x, y), False)
            self.pruning_extraction((x, y), root, False)
            connections3 = connections3.union(self.connections)
            self.connections = set()

        connections: Set[Connection] = connections1.union(
            connections2.union(connections3)
        )

        return self.clean_net(connections)

    def clean_net(
        self, connections: Set[Connection]
    ) -> Tuple[Set[Coord], Set[Connection]]:
        """Remove dead-end nodes that don't form a path from input to output.

        Returns
        -------
        tuple
            ``(true_nodes, true_connections)`` containing only nodes and
            connections on valid input→output paths.
        """
        inputs = set(self.substrate.input_coordinates)
        outputs = set(self.substrate.output_coordinates)

        # Forward pass from inputs.
        connected_to_inputs: Set[Coord] = set(inputs)
        add_happened = True
        while add_happened:
            add_happened = False
            for c in list(connections):
                if (c.x1, c.y1) in connected_to_inputs:
                    old_len = len(connected_to_inputs)
                    connected_to_inputs.add((c.x2, c.y2))
                    if len(connected_to_inputs) > old_len:
                        add_happened = True

        # Backward pass from outputs.
        connected_to_outputs: Set[Coord] = set(outputs)
        add_happened = True
        while add_happened:
            add_happened = False
            for c in list(connections):
                if (c.x2, c.y2) in connected_to_outputs:
                    old_len = len(connected_to_outputs)
                    connected_to_outputs.add((c.x1, c.y1))
                    if len(connected_to_outputs) > old_len:
                        add_happened = True

        # Intersection = nodes on valid paths.
        true_nodes = connected_to_inputs.intersection(connected_to_outputs)

        # Filter connections.
        true_connections: Set[Connection] = set()
        for c in connections:
            if (
                (c.x1, c.y1) in true_nodes
                and (c.x2, c.y2) in true_nodes
            ):
                true_connections.add(c)

        # Remove the input/output nodes themselves.
        true_nodes -= inputs.union(outputs)

        return true_nodes, true_connections


def find_pattern(
    cppn: neat.nn.FeedForwardNetwork,
    coord: Coord,
    res: int = 60,
    max_weight: float = 5.0,
) -> np.ndarray:
    """Query the CPPN for weights from *coord* to all points in a grid.

    Useful for visualising CPPN connectivity patterns.

    Parameters
    ----------
    cppn:
        The CPPN.
    coord:
        Source coordinate ``(x, y)``.
    res:
        Grid resolution (output is ``res × res``).
    max_weight:
        Maximum absolute weight (values are scaled to this).

    Returns
    -------
    np.ndarray
        2-D array of shape ``(res, res)``.
    """
    im = np.zeros((res, res))

    for x2 in range(res):
        for y2 in range(res):
            x2_scaled = -1.0 + (x2 / float(res)) * 2.0
            y2_scaled = -1.0 + (y2 / float(res)) * 2.0

            i = [coord[0], coord[1], x2_scaled, y2_scaled, 1.0]
            n = cppn.activate(i)[0]

            im[x2][y2] = n * max_weight

    return im
