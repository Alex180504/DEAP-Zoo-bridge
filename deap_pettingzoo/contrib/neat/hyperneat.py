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

from __future__ import annotations

from typing import Any, Optional

import numpy as np

import neat  # guarded by _require_neat

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.policies.action_heads import (
    ActionHead,
    make_head,
)

from ._pureples.hyperneat import create_phenotype_network
from ._pureples.es_hyperneat import ESNetwork, ES_HYPERNEAT_PARAMS_DEFAULTS
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
            cppn, substrate, activation_function=activation
        )

        self._head: Optional[ActionHead] = (
            make_head(action_space, masked=self._masked)
            if action_space is not None
            else None
        )

    # ------------------------------------------------------------------
    # PolicyProtocol
    # ------------------------------------------------------------------

    def get_action(
        self,
        observation: np.ndarray,
        *,
        action_mask: Optional[np.ndarray] = None,
    ) -> Any:
        obs = np.asarray(observation, dtype=np.float64)
        raw = self._network.activate(obs)
        raw_arr = np.asarray(raw, dtype=np.float32)

        if self._head is not None:
            if (
                action_mask is not None
                and getattr(self._head, "supports_mask", False)
            ):
                return self._head.decode(raw_arr, action_mask=action_mask)
            return self._head.decode(raw_arr)
        return raw_arr

    def reset(self) -> None:
        """No-op — the feed-forward phenotype has no hidden state."""
        pass

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

        self._head: Optional[ActionHead] = (
            make_head(action_space, masked=self._masked)
            if action_space is not None
            else None
        )

    # ------------------------------------------------------------------
    # PolicyProtocol
    # ------------------------------------------------------------------

    def get_action(
        self,
        observation: np.ndarray,
        *,
        action_mask: Optional[np.ndarray] = None,
    ) -> Any:
        obs = np.asarray(observation, dtype=np.float64)
        raw = self._network.activate(obs)
        raw_arr = np.asarray(raw, dtype=np.float32)

        if self._head is not None:
            if (
                action_mask is not None
                and getattr(self._head, "supports_mask", False)
            ):
                return self._head.decode(raw_arr, action_mask=action_mask)
            return self._head.decode(raw_arr)
        return raw_arr

    def reset(self) -> None:
        pass

    @property
    def genome(self) -> neat.DefaultGenome:
        return self._genome

    @property
    def substrate(self) -> Substrate:
        return self._substrate
