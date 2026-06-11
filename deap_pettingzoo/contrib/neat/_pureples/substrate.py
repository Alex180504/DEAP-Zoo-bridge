"""
Substrate — the geometric layout of input, output, and (optionally) hidden
nodes for HyperNEAT phenotype queries.

:class:`Substrate` was extracted from `pureples.shared.substrate
<https://github.com/ukuleleplayer/pureples/blob/master/pureples/shared/substrate.py>`_
(MIT licensed).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


# A coordinate is a 2-D point in the range [-1, 1] × [-1, 1].
Coord = Tuple[float, float]


class Substrate:
    """Geometric layout of nodes for a HyperNEAT phenotype network.

    Attributes
    ----------
    input_coordinates:
        List of ``(x, y)`` tuples in ``[-1, 1]²`` — one per input feature.
    output_coordinates:
        List of ``(x, y)`` tuples in ``[-1, 1]²`` — one per output unit.
    hidden_coordinates:
        List of **layers**, where each layer is a list of ``(x, y)`` tuples.
        The first layer is the **topmost** (closest to outputs), the last
        is the **bottommost** (closest to inputs).  An empty list means
        "no hidden nodes" (standard for ES-HyperNEAT which discovers them
        automatically).
    """

    def __init__(
        self,
        input_coordinates: Sequence[Coord],
        output_coordinates: Sequence[Coord],
        hidden_coordinates: Optional[Sequence[Sequence[Coord]]] = None,
    ) -> None:
        self.input_coordinates = list(input_coordinates)
        self.output_coordinates = list(output_coordinates)
        self.hidden_coordinates = (
            [list(layer) for layer in hidden_coordinates]
            if hidden_coordinates is not None
            else []
        )

    @property
    def num_inputs(self) -> int:
        return len(self.input_coordinates)

    @property
    def num_outputs(self) -> int:
        return len(self.output_coordinates)

    @property
    def num_hidden_layers(self) -> int:
        return len(self.hidden_coordinates)

    def __repr__(self) -> str:
        return (
            f"Substrate(inputs={self.num_inputs}, outputs={self.num_outputs}, "
            f"hidden_layers={self.num_hidden_layers})"
        )
