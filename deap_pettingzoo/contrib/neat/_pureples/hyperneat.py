"""
HyperNEAT — query a CPPN at substrate coordinates to construct a phenotype ANN.

Extracted from `pureples.hyperneat.hyperneat
<https://github.com/ukuleleplayer/pureples/blob/master/pureples/hyperneat/hyperneat.py>`_
(MIT licensed).

Patches for neat-python ≥ 2.0
------------------------------
* ``neat.activations.ActivationFunctionSet`` used as-is (still available).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import neat

from .substrate import Coord, Substrate


def create_phenotype_network(
    cppn: neat.nn.FeedForwardNetwork,
    substrate: Substrate,
    activation_function: str = "sigmoid",
) -> neat.nn.RecurrentNetwork:
    """Build a phenotype ANN by querying *cppn* at *substrate* coordinates.

    Parameters
    ----------
    cppn:
        The evolved CPPN (a :class:`~neat.nn.FeedForwardNetwork`).
    substrate:
        The geometric layout of input/output/hidden nodes.
    activation_function:
        Activation function name for hidden/output nodes (registered
        in :class:`neat.activations.ActivationFunctionSet`).

    Returns
    -------
    neat.nn.RecurrentNetwork
        The phenotype network ready for ``.activate(inputs)`` calls.
    """
    input_coordinates = substrate.input_coordinates
    output_coordinates = substrate.output_coordinates
    hidden_coordinates = substrate.hidden_coordinates

    input_nodes = list(range(len(input_coordinates)))
    output_nodes = list(
        range(len(input_nodes), len(input_nodes) + len(output_coordinates))
    )

    counter = 0
    for layer in hidden_coordinates:
        counter += len(layer)

    hidden_nodes = range(
        len(input_nodes) + len(output_nodes),
        len(input_nodes) + len(output_nodes) + counter,
    )

    node_evals: list = []

    # Get activation function.
    activation_functions = neat.activations.ActivationFunctionSet()
    activation = activation_functions.get(activation_function)

    # Connect hidden to output.
    counter = 0
    for oc in output_coordinates:
        idx = 0
        for layer in hidden_coordinates:
            im = find_neurons(cppn, oc, layer, hidden_nodes[idx], False)
            idx += len(layer)
            if im:
                node_evals.append(
                    (output_nodes[counter], activation, sum, 0.0, 1.0, im)
                )
        counter += 1

    # Connect hidden to hidden — starting from the top layer.
    current_layer = 1
    idx = 0
    for layer in hidden_coordinates:
        idx += len(layer)
        counter = idx - len(layer)
        for i in range(current_layer, len(hidden_coordinates)):
            for hc in layer:
                im = find_neurons(
                    cppn, hc, hidden_coordinates[i], hidden_nodes[idx], False
                )
                if im:
                    node_evals.append(
                        (hidden_nodes[counter], activation, sum, 0.0, 1.0, im)
                    )
                counter += 1
            counter -= idx
        current_layer += 1

    # Connect input to hidden.
    counter = 0
    for layer in hidden_coordinates:
        for hc in layer:
            im = find_neurons(
                cppn, hc, input_coordinates, input_nodes[0], False
            )
            if im:
                node_evals.append(
                    (hidden_nodes[counter], activation, sum, 0.0, 1.0, im)
                )
            counter += 1

    return neat.nn.RecurrentNetwork(input_nodes, output_nodes, node_evals)


def find_neurons(
    cppn: neat.nn.FeedForwardNetwork,
    coord: Coord,
    nodes: Sequence[Coord],
    start_idx: int,
    outgoing: bool,
    max_weight: float = 5.0,
) -> list:
    """Find the nodes to which *coord* is connected via the CPPN.

    Returns a list of ``(target_idx, weight)`` tuples suitable for use
    as the ``links`` field in a ``node_evals`` entry.
    """
    result: list = []
    for i, node in enumerate(nodes):
        w = query_cppn(coord, node, outgoing, cppn, max_weight)
        if w != 0.0:
            result.append((start_idx + i, w))
    return result


def query_cppn(
    coord1: Coord,
    coord2: Coord,
    outgoing: bool,
    cppn: neat.nn.FeedForwardNetwork,
    max_weight: float = 5.0,
) -> float:
    """Query the CPPN for the weight from *coord1* to *coord2*.

    Parameters
    ----------
    coord1:
        Source ``(x, y)``.
    coord2:
        Target ``(x, y)``.
    outgoing:
        If ``True``, *coord1* is the source (output direction); otherwise
        *coord2* is the source (input direction).  This matters because
        the CPPN input vector is ``[x1, y1, x2, y2, bias]``.
    cppn:
        The CPPN query network.
    max_weight:
        Maximum absolute weight value (outputs are scaled to this).

    Returns
    -------
    float
        The queried weight (0.0 if below the 0.2 threshold).
    """
    if outgoing:
        i = [coord1[0], coord1[1], coord2[0], coord2[1], 1.0]
    else:
        i = [coord2[0], coord2[1], coord1[0], coord1[1], 1.0]
    w = cppn.activate(i)[0]
    if abs(w) > 0.2:
        if w > 0:
            w = (w - 0.2) / 0.8
        else:
            w = (w + 0.2) / 0.8
        return w * max_weight
    return 0.0
