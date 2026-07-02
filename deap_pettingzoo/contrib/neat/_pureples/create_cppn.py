"""
CPPN creator — build a :class:`~neat.nn.FeedForwardNetwork` from a genome.

Extracted from `pureples.shared.create_cppn
<https://github.com/ukuleleplayer/pureples/blob/master/pureples/shared/create_cppn.py>`_
(MIT licensed).

Patches for neat-python ≥ 2.0
------------------------------
* ``neat.graphs.feed_forward_layers`` used as-is (still available).
* ``config.genome_config`` access pattern unchanged in 2.x.
"""

from __future__ import annotations

from typing import Any

import neat
from neat.graphs import feed_forward_layers


def create_cppn(
    genome: neat.DefaultGenome,
    config: neat.Config,
    output_activation_function: str = "tanh",
) -> neat.nn.FeedForwardNetwork:
    """Build a CPPN (:class:`~neat.nn.FeedForwardNetwork`) from a genome.

    This is the standard "genome → phenotype" method used by pureples
    for HyperNEAT / ES-HyperNEAT.  The output nodes' activation function
    can be overridden (e.g. to ``"sigmoid"`` for weight-production CPPNs).

    Parameters
    ----------
    genome:
        A NEAT :class:`~neat.DefaultGenome`.
    config:
        The :class:`~neat.Config` that was used to evolve *genome*.
    output_activation_function:
        Activation function name for output nodes (``"tanh"``, ``"sigmoid"``,
        ``"gauss"``, ``"sin"``, etc.).

    Returns
    -------
    neat.nn.FeedForwardNetwork
        The CPPN ready for ``.activate(inputs)``.
    """
    # Gather expressed connections.
    connections = [cg.key for cg in genome.connections.values() if cg.enabled]

    layers = feed_forward_layers(
        config.genome_config.input_keys,
        config.genome_config.output_keys,
        connections,
    )
    node_evals: list = []
    for layer in layers:
        for node in layer:
            inputs: list = []
            for conn_key in connections:
                inode, onode = conn_key
                if onode == node:
                    cg = genome.connections[conn_key]
                    inputs.append((inode, cg.weight))

            ng = genome.nodes[node]
            aggregation_function = config.genome_config.aggregation_function_defs.get(
                ng.aggregation
            )
            activation_function = config.genome_config.activation_defs.get(
                ng.activation
            )
            # Override output-node activation.
            if node in config.genome_config.output_keys:
                activation_function = config.genome_config.activation_defs.get(
                    output_activation_function
                )

            node_evals.append(
                (
                    node,
                    activation_function,
                    aggregation_function,
                    ng.bias,
                    ng.response,
                    inputs,
                )
            )

    return neat.nn.FeedForwardNetwork(
        config.genome_config.input_keys,
        config.genome_config.output_keys,
        node_evals,
    )
