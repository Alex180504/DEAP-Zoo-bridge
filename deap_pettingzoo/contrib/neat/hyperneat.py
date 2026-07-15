"""HyperNEAT and ES-HyperNEAT policies satisfying :class:`PolicyProtocol`.

:class:`HyperNeatPolicy`
    Evolves a CPPN (Compositional Pattern Producing Network) via NEAT,
    then queries it at fixed substrate coordinates to build a phenotype
    :class:`~neat.nn.RecurrentNetwork`.  The phenotype is used for
    inference.

:class:`ESHyperNeatPolicy`
    Evolves a CPPN via NEAT, then uses the ES-HyperNEAT quadtree
    algorithm to discover hidden nodes and their connections
    automatically before building the phenotype network.  No hidden
    nodes need to be specified in the substrate — they are placed by
    the algorithm.
"""

# pyright: reportCallIssue=false

from __future__ import annotations

from typing import Any, Optional

import neat  # guarded by _require_neat
import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.policies.action_heads import (
    ActionHead,
    MaskedDiscreteHead,
    make_head,
)

from ._pureples.es_hyperneat import ES_HYPERNEAT_PARAMS_DEFAULTS, ESNetwork
from ._pureples.hyperneat import create_phenotype_network
from ._pureples.substrate import Substrate


class HyperNeatPolicy(PolicyProtocol):
    """Policy backed by a HyperNEAT phenotype network.

    A CPPN (evolved via NEAT) is queried at a fixed substrate to produce
    a phenotype :class:`~neat.nn.RecurrentNetwork`.  The phenotype is
    used for :meth:`get_action` inference.

    Parameters
    ----------
    genome:
        The CPPN genome (a ``neat.DefaultGenome``).
    config:
        The NEAT config used to evolve the CPPN.
    substrate:
        The geometric layout for phenotype construction.
    action_space:
        Optional gym/gymnasium ``Space`` for action decoding.
    masked:
        Forwarded to :func:`make_head`.
    activation:
        Phenotype activation function (default ``"sigmoid"``).
    weight_threshold:
        Minimum absolute CPPN output below which phenotype weights are
        zeroed.  Lower values let through more connections at the cost
        of noise.  Default 0.1.
    """

    def __init__(
        self,
        genome: neat.DefaultGenome,
        config: neat.Config,
        substrate: Substrate,
        *,
        action_space: Any = None,
        masked: bool = False,
        activation: str = "sigmoid",
        weight_threshold: float = 0.1,
    ) -> None:
        self._genome = genome
        self._config = config
        self._substrate = substrate
        self._activation = activation
        self._masked = bool(masked)

        # Build CPPN from genome.
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        # Build phenotype from CPPN + substrate.
        self._network = create_phenotype_network(
            cppn,
            substrate,
            activation_function=activation,
            weight_threshold=weight_threshold,
        )
        self._activation_steps = max(
            1, len(getattr(substrate, "hidden_coordinates", []) or []) + 1
        )

        self._head: Optional[ActionHead] = (
            make_head(action_space, masked=self._masked)
            if action_space is not None
            else None
        )

    # PolicyProtocol

    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        *,
        action_mask: Optional[np.ndarray] = None,
    ) -> Any:
        # Heads are deterministic decoders; the kwarg exists for
        # PolicyProtocol compatibility (benchmark probes pass it).
        del deterministic
        obs = np.asarray(observation, dtype=np.float64)
        self._network.reset()
        raw = None
        for _ in range(self._activation_steps):
            raw = self._network.activate(obs)
        assert raw is not None
        raw_arr = np.asarray(raw, dtype=np.float32)

        head = self._head
        if head is not None:
            if action_mask is not None and isinstance(head, MaskedDiscreteHead):
                return MaskedDiscreteHead.decode(head, raw_arr, action_mask)
            return head.decode(raw_arr)
        return raw_arr

    def reset(self) -> None:
        self._network.reset()

    @property
    def genome(self) -> neat.DefaultGenome:
        return self._genome

    @property
    def substrate(self) -> Substrate:
        return self._substrate


class ESHyperNeatPolicy(PolicyProtocol):
    """Policy backed by an ES-HyperNEAT phenotype network.

    Like :class:`HyperNeatPolicy`, but the hidden nodes are discovered
    automatically via variance-based quadtree subdivision — no hidden
    coordinates need to be specified in the substrate.

    Parameters
    ----------
    genome:
        The CPPN genome.
    config:
        The NEAT config.
    substrate:
        Substrate with input + output coordinates (hidden_coordinates is
        ignored — ES-HyperNEAT discovers hidden nodes automatically).
    action_space:
        Optional gym action space.
    masked:
        Forwarded to :func:`make_head`.
    es_params:
        ES-HyperNEAT parameter dict.  See
        :data:`~._pureples.es_hyperneat.ES_HYPERNEAT_PARAMS_DEFAULTS`.
    """

    def __init__(
        self,
        genome: neat.DefaultGenome,
        config: neat.Config,
        substrate: Substrate,
        *,
        action_space: Any = None,
        masked: bool = False,
        es_params: Optional[dict] = None,
    ) -> None:
        self._genome = genome
        self._config = config
        self._substrate = substrate
        self._masked = bool(masked)
        self._es_params = dict(
            es_params if es_params is not None else ES_HYPERNEAT_PARAMS_DEFAULTS
        )

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        es_network = ESNetwork(substrate, cppn, self._es_params)
        self._network = es_network.create_phenotype_network()
        self._activation_steps = max(1, int(es_network.activations))

        self._head: Optional[ActionHead] = (
            make_head(action_space, masked=self._masked)
            if action_space is not None
            else None
        )

    # PolicyProtocol

    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        *,
        action_mask: Optional[np.ndarray] = None,
    ) -> Any:
        # Heads are deterministic decoders; the kwarg exists for
        # PolicyProtocol compatibility (benchmark probes pass it).
        del deterministic
        obs = np.asarray(observation, dtype=np.float64)
        self._network.reset()
        raw = None
        for _ in range(self._activation_steps):
            raw = self._network.activate(obs)
        assert raw is not None
        raw_arr = np.asarray(raw, dtype=np.float32)

        head = self._head
        if head is not None:
            if action_mask is not None and isinstance(head, MaskedDiscreteHead):
                return MaskedDiscreteHead.decode(head, raw_arr, action_mask)
            return head.decode(raw_arr)
        return raw_arr

    def reset(self) -> None:
        self._network.reset()

    @property
    def genome(self) -> neat.DefaultGenome:
        return self._genome

    @property
    def substrate(self) -> Substrate:
        return self._substrate

    def __repr__(self) -> str:
        n_in = len(self._substrate.input_coordinates)
        n_out = len(self._substrate.output_coordinates)
        n_evals = len(self._network.node_evals)
        # node_evals is list of (key, activation, aggregation, bias, response, links)
        n_nodes = len({entry[0] for entry in self._network.node_evals})
        return (
            f"ESHyperNeatPolicy(inputs={n_in}, outputs={n_out}, "
            f"phenotype_nodes={n_nodes}, node_evals={n_evals}, "
            f"activation_steps={self._activation_steps})"
        )
