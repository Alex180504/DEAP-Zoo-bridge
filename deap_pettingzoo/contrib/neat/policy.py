"""NEAT policy adapter — wraps a ``neat.nn.FeedForwardNetwork``.

:class:`NeatPolicy` satisfies
:class:`~deap_pettingzoo.baselines.base.PolicyProtocol` and is the
bridge between neat-python's genome→network pipeline and the unified
evaluator API.  It supports both discrete and continuous action spaces
via the same :mod:`~deap_pettingzoo.policies.action_heads` machinery
used by :class:`~deap_pettingzoo.policies.FeedforwardNeuralPolicy`.

Parallel-evaluation pickling
----------------------------
``NeatPolicy`` is NOT directly picklable because
``neat.nn.FeedForwardNetwork`` is not fork-safe.  Instead, ship the
policy via :class:`~deap_pettingzoo.baselines.coevolution.pools.OpponentSnapshot`
with ``kind="neat"`` and payload
``{"genome": ..., "config_input_size": int, "config_output_size": int}``.
The worker calls :func:`rebuild_opponent_policy` which rebuilds the
``neat.Config`` via :func:`build_neat_config` and reconstructs
a ``NeatPolicy`` from the genome.
"""

from __future__ import annotations

from typing import Any, Optional

import neat  # guarded by _require_neat in __init__
import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.policies.action_heads import (
    ActionHead,
    make_head,
)


class NeatPolicy(PolicyProtocol):
    """Policy backed by a NEAT :class:`~neat.nn.FeedForwardNetwork`.

    The policy is stateless — :meth:`reset` is a no-op.

    Parameters
    ----------
    genome:
        A ``neat.DefaultGenome`` instance.  A fresh
        :class:`~neat.nn.FeedForwardNetwork` is created via
        ``FeedForwardNetwork.create(genome, config)`` and cached.
    config:
        The :class:`neat.Config` used to evolve *genome*.
    action_space:
        A gym/gymnasium ``Space`` used to build the :class:`ActionHead`.
        Pass ``None`` to skip head construction (raw network outputs).
    masked:
        Forwarded to :func:`~deap_pettingzoo.policies.action_heads.make_head`.
    """

    def __init__(
        self,
        genome: neat.DefaultGenome,
        config: neat.Config,
        *,
        action_space: Any = None,
        masked: bool = False,
    ) -> None:
        self._genome = genome
        self._config = config
        self._masked = bool(masked)

        self._network: Any = neat.nn.FeedForwardNetwork.create(genome, config)
        self._head: Optional[ActionHead] = (
            make_head(action_space, masked=self._masked)
            if action_space is not None
            else None
        )

    # PolicyProtocol

    def get_action(
        self,
        observation: np.ndarray,
        *,
        action_mask: Optional[np.ndarray] = None,
    ) -> Any:
        """Return an action for *observation*.

        Args:
            observation: 1-D array of shape ``(input_dim,)``.
            action_mask: Optional boolean mask forwarded to the head
                when the head supports masking.

        Returns:
            Decoded action (type depends on the configured head) or
            raw ``np.ndarray`` when no head was configured.
        """
        obs = np.asarray(observation, dtype=np.float64)
        raw = self._network.activate(obs)
        raw_arr = np.asarray(raw, dtype=np.float32)

        if self._head is not None:
            if action_mask is not None and getattr(self._head, "supports_mask", False):
                return self._head.decode(raw_arr, action_mask=action_mask)
            return self._head.decode(raw_arr)
        return raw_arr

    def reset(self) -> None:
        """No-op — the feed-forward policy has no hidden state."""
        pass

    # Utility properties

    @property
    def genome(self) -> neat.DefaultGenome:
        """The NEAT genome driving this policy (read-only)."""
        return self._genome

    @property
    def config(self) -> neat.Config:
        """The NEAT config (read-only)."""
        return self._config

    # Pickling helpers (used by OpponentSnapshot kind="neat")

    @classmethod
    def from_snapshot_payload(
        cls,
        genome: neat.DefaultGenome,
        config: neat.Config,
        *,
        action_space: Any = None,
        masked: bool = False,
    ) -> NeatPolicy:
        """Factory used by :func:`rebuild_opponent_policy` for kind ``"neat"``.

        This is the canonical way to revive a ``NeatPolicy`` on a
        worker process.  It wraps :meth:`__init__`.
        """
        return cls(genome, config, action_space=action_space, masked=masked)
