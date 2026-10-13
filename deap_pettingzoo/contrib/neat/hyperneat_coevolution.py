"""HyperNEAT and ES-HyperNEAT co-evolution side states.

:class:`HyperNeatSideState` and :class:`ESHyperNeatSideState` extend
:class:`NeatSideState`.  Everything that makes a co-evolution side — the
generation loop, the opponent pool, the hall-of-fame blending, the
learner hooks — is inherited; the subclasses differ only in how a CPPN
genome becomes an executable phenotype (substrate query vs ES-HyperNEAT
quadtree) and in the substrate they build at :meth:`initialise` time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import neat  # guarded by _require_neat
import numpy as np

from deap_pettingzoo.baselines.base import PolicyProtocol
from deap_pettingzoo.baselines.coevolution.pools import OpponentSnapshot
from deap_pettingzoo.baselines.coevolution.sides import Side

from ._pureples.es_hyperneat import ES_HYPERNEAT_PARAMS_DEFAULTS
from ._pureples.substrate import Substrate
from .coevolution import NeatSideState
from .config import build_cppn_config
from .hyperneat import ESHyperNeatPolicy, HyperNeatPolicy
from .substrate import substrate_for_env


class _CPPNSideState(NeatSideState):
    """Shared plumbing for the two CPPN-backed sides.

    Both evolve a 5-input / 1-output CPPN and express it through a
    substrate; only the expression step and the substrate layout differ.
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        self._stochastic_head = bool(cfg.get("stochastic_head", False))
        self._head_temperature = float(cfg.get("head_temperature", 1.0))
        self._substrate: Optional[Substrate] = None
        self._env_obs_size: int = 0
        self._env_act_size: int = 0
        self._head_rng: Optional[np.random.Generator] = None

    def initialise(
        self,
        env,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().initialise(env, rng=rng)
        # Seeded from the side's own RNG so a stochastic head replays
        # identically under a fixed trainer seed.
        self._head_rng = (
            np.random.default_rng(self._neat_seed) if self._stochastic_head else None
        )

    def _build_config(self, input_size: int, output_size: int) -> neat.Config:
        """CPPN config — the genome maps (x1, y1, x2, y2, bias) to one weight.

        The substrate is what carries the env dimensions, so the CPPN's own
        input/output sizes are fixed; ``input_size`` / ``output_size`` are
        consumed by :meth:`_build_substrate` instead.
        """
        self._env_obs_size = input_size
        self._env_act_size = output_size
        self._substrate = self._build_substrate(input_size, output_size)
        return build_cppn_config(
            input_size=5,
            output_size=1,
            pop_size=self.pop_size,
            seed=self._neat_seed,
            overrides=self._config_overrides(),
        )

    def _build_substrate(self, input_size: int, output_size: int) -> Substrate:
        raise NotImplementedError  # pragma: no cover - abstract


class HyperNeatSideState(_CPPNSideState):
    """HyperNEAT co-evolution side.

    Same as :class:`NeatSideState` but expresses the CPPN through a fixed
    substrate into a :class:`HyperNeatPolicy`.

    Extra ``optimizer_config`` keys (vs :class:`NeatSideState`):

    * ``substrate_hidden_layers`` — optional ``list[int]`` of hidden
      layer sizes for the HyperNEAT substrate.
    * ``phenotype_activation`` — activation for the phenotype network
      (default ``"sigmoid"``).
    * ``stochastic_head`` (bool, default ``False``) — decode discrete
      actions by softmax sampling during training rather than argmax.
    * ``head_temperature`` (float, default ``1.0``) — temperature for
      that head.
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        self._hidden_layers = cfg.get("substrate_hidden_layers", None)
        self._phenotype_activation = cfg.get("phenotype_activation", "sigmoid")

    def _build_substrate(self, input_size: int, output_size: int) -> Substrate:
        return substrate_for_env(
            input_size,
            output_size,
            layout="uniform",
            hidden_layers=self._hidden_layers,
        )

    def _build_phenotype(
        self,
        genome: neat.DefaultGenome,
        action_space: Any,
        *,
        stochastic: bool = True,
    ) -> PolicyProtocol:
        return HyperNeatPolicy(
            genome,
            self._neat_config,  # type: ignore[arg-type]
            self._substrate,  # type: ignore[arg-type]
            action_space=action_space,
            masked=self.masked,
            activation=self._phenotype_activation,
            stochastic=stochastic and self._stochastic_head,
            temperature=self._head_temperature,
            rng=self._head_rng,
        )

    def _phenotype_snapshot(self, genome: neat.DefaultGenome) -> OpponentSnapshot:
        return OpponentSnapshot(
            kind="hyperneat",
            payload={
                "genome": genome,
                "env_obs_size": self._env_obs_size,
                "env_act_size": self._env_act_size,
                "activation": self._phenotype_activation,
                # Without this the worker builds a substrate with no hidden
                # layers — a topologically different phenotype from the very
                # same genome.
                "hidden_layers": (
                    [int(h) for h in self._hidden_layers]
                    if self._hidden_layers
                    else None
                ),
                "masked": self.masked,
                "stochastic": self._stochastic_head,
                "temperature": self._head_temperature,
            },
        )


class ESHyperNeatSideState(_CPPNSideState):
    """ES-HyperNEAT co-evolution side.

    Same as :class:`NeatSideState` but expresses the CPPN through the
    ES-HyperNEAT quadtree into an :class:`ESHyperNeatPolicy`.

    Extra ``optimizer_config`` keys:

    * ``es_params`` — ES-HyperNEAT parameter dict (default
      :data:`~._pureples.es_hyperneat.ES_HYPERNEAT_PARAMS_DEFAULTS`).
    * ``stochastic_head`` / ``head_temperature`` — see
      :class:`HyperNeatSideState`.
    """

    def __init__(self, side: Side) -> None:
        super().__init__(side)
        cfg = side.optimizer_config
        self._es_params: Dict[str, Any] = dict(
            cfg.get("es_params", ES_HYPERNEAT_PARAMS_DEFAULTS)
        )

    def _build_substrate(self, input_size: int, output_size: int) -> Substrate:
        # No hidden coordinates — ES-HyperNEAT discovers them itself.
        return substrate_for_env(input_size, output_size, layout="uniform")

    def _build_phenotype(
        self,
        genome: neat.DefaultGenome,
        action_space: Any,
        *,
        stochastic: bool = True,
    ) -> PolicyProtocol:
        return ESHyperNeatPolicy(
            genome,
            self._neat_config,  # type: ignore[arg-type]
            self._substrate,  # type: ignore[arg-type]
            action_space=action_space,
            masked=self.masked,
            es_params=self._es_params,
            stochastic=stochastic and self._stochastic_head,
            temperature=self._head_temperature,
            rng=self._head_rng,
        )

    def _phenotype_snapshot(self, genome: neat.DefaultGenome) -> OpponentSnapshot:
        return OpponentSnapshot(
            kind="es_hyperneat",
            payload={
                "genome": genome,
                "env_obs_size": self._env_obs_size,
                "env_act_size": self._env_act_size,
                "es_params": self._es_params,
                "masked": self.masked,
                "stochastic": self._stochastic_head,
                "temperature": self._head_temperature,
            },
        )
