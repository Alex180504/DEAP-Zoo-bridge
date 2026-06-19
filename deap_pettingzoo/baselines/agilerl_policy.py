"""
Policy wrapper for trained AgileRL multi-agent algorithms.

AgileRL algorithms expose :meth:`get_action` that takes a dict mapping
``agent_id -> observation`` and returns per-agent actions. The unified
baseline API, by contrast, is a stateless-looking :class:`PolicyProtocol`
that receives one observation at a time with no agent id.

:class:`AgileRLPolicyWrapper` bridges the two by:

* tracking a rotation counter through ``possible_agents`` that is reset
  at the beginning of each episode (relies on :class:`UnifiedEvaluator`
  iterating ``env.possible_agents`` in a fixed order and calling
  ``get_action`` once per agent per env step), and
* maintaining a per-agent observation cache so that every call can pass
  the full ``{agent_id: obs}`` dict required by multi-agent AgileRL
  algorithms. Since MADDPG/MATD3/IPPO use decentralized actors, the
  returned action for the current agent depends only on that agent's
  obs; stale entries for the other agents are harmless and get refreshed
  as the rotation progresses through the step.

:class:`FrozenRLPolicyWrapper` is a CPU-only, picklable inference-only
snapshot created by :meth:`AgileRLPolicyWrapper.make_frozen_cpu_snapshot`.
It holds deepcopied CPU actor ``nn.Module`` objects and implements the
same :class:`PolicyProtocol` interface without any optimizer, replay
buffer, or CUDA state, making it safe to ship across ``ProcessPoolExecutor``
spawn boundaries for parallel co-evolution evaluation.
"""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _ensure_agilerl_imports() -> None:
    """Guarantee ``agilerl.networks`` is fully loaded in *this* process.

    In spawn-mode ``ProcessPoolExecutor`` workers the Python interpreter
    starts fresh.  If an earlier import attempt (e.g. the grouping patch in
    ``agilerl_trainer``) was interrupted, ``torch._inductor`` may be left in
    ``sys.modules`` as a *partially initialised* module — the
    ``custom_graph_pass`` submodule attribute has not been set yet.  Any
    subsequent attempt to import ``agilerl.networks`` (which drills down
    through ``torch._dynamo`` → ``torch._inductor.config``) will then fail
    with::

        AttributeError: partially initialised module 'torch._inductor'
            has no attribute 'custom_graph_pass'

    The fix: detect the partial state, evict *all* ``torch._inductor``
    entries from ``sys.modules`` so they can be re-imported cleanly, then
    import ``agilerl.networks``.  Evicting ``torch._inductor`` modules is
    safe in a brand-new worker subprocess — there are no compiled graphs or
    cached state to lose.
    """
    import sys

    if "agilerl.networks" in sys.modules:
        return  # already fully loaded — fast path

    # Detect a partially-initialised torch._inductor (missing the
    # 'custom_graph_pass' attribute that config.py needs at annotation time).
    _ind = sys.modules.get("torch._inductor")
    if _ind is not None and not hasattr(_ind, "custom_graph_pass"):
        # Evict all stale torch._inductor entries so the next import starts
        # from a clean slate.  Also evict torch._dynamo because it may hold
        # references into the broken torch._inductor module object.
        _stale = [
            k for k in list(sys.modules)
            if k.startswith("torch._inductor") or k.startswith("torch._dynamo")
        ]
        for k in _stale:
            del sys.modules[k]

    try:
        import agilerl.networks  # noqa: F401
    except Exception:
        pass  # best-effort; pickle.loads will raise a clear error if needed


class AgileRLPolicyWrapper:
    """Adapt a trained AgileRL multi-agent algorithm to :class:`PolicyProtocol`.

    Args:
        agent: Trained AgileRL algorithm (e.g. ``MADDPG``, ``MATD3``,
            ``IPPO``) exposing a ``get_action(obs_dict, training=...)``.
        agent_ids: PettingZoo ``possible_agents`` list in evaluation
            order. ``get_action`` dispatches through this list round-robin.
        algorithm_name: Optional human-readable label (used only for
            ``repr``).
    """

    def __init__(
        self,
        agent: Any,
        agent_ids: Iterable[str],
        algorithm_name: Optional[str] = None,
    ) -> None:
        self._agent = agent
        self._agent_ids = list(agent_ids)
        if not self._agent_ids:
            raise ValueError("agent_ids must be non-empty")
        self._call_index = 0
        self._obs_cache: Dict[str, np.ndarray] = {}
        self._algorithm_name = algorithm_name or type(agent).__name__

    # Policy protocol

    def get_action(self, observation: np.ndarray, **_: Any) -> Any:
        """Return a deterministic action for the next agent in rotation.

        Accepts and ignores extra keyword arguments (e.g. ``deterministic``)
        so it can be dropped into evaluators that pass such flags.
        """
        agent_id = self._agent_ids[self._call_index % len(self._agent_ids)]
        self._call_index += 1

        self._obs_cache[agent_id] = np.asarray(observation)
        obs_dict = self._build_full_obs_dict()

        try:
            result = self._agent.get_action(obs_dict, training=False)
        except TypeError:
            # On-policy algorithms may use a different kwarg name.
            result = self._agent.get_action(obs_dict)

        actions = result[0] if isinstance(result, tuple) else result
        action = np.asarray(actions[agent_id])

        # Strip leading vectorization dims of size 1.
        while action.ndim > 0 and action.shape[0] == 1:
            action = action[0]
        if action.shape == ():
            return action.item()
        return action

    def reset(self) -> None:
        """Reset the agent rotation counter and obs cache for a new episode."""
        self._call_index = 0
        self._obs_cache = {}

    # Internal helpers

    def _build_full_obs_dict(self) -> Dict[str, np.ndarray]:
        """Return ``{agent_id: obs}`` with zero-filled placeholders for any
        agent whose observation has not yet been seen this episode.

        Required because AgileRL's multi-agent algorithms expect every
        agent they were constructed with to be keyed in the input dict
        — not just the subset this wrapper dispatches for. Placeholders
        only matter for decentralized-actor algorithms in the warm-up
        phase before the first full rotation completes; thereafter
        every slot holds the most recent real observation for that
        agent.
        """
        full: Dict[str, np.ndarray] = {}
        fallback_shape = next(iter(self._obs_cache.values())).shape
        fallback_dtype = next(iter(self._obs_cache.values())).dtype
        # Use the underlying algorithm's full agent roster so subsets still satisfy MADDPG/MATD3 "every agent keyed" precondition.
        for aid in self._full_agent_roster():
            if aid in self._obs_cache:
                full[aid] = self._obs_cache[aid]
            else:
                full[aid] = self._zero_obs_for(aid, fallback_shape, fallback_dtype)
        return full

    def _full_agent_roster(self) -> list:
        """Best-effort list of every agent the underlying algorithm
        expects in its input dict. Falls back to this wrapper's own
        agent_ids if nothing else is exposed.
        """
        roster = getattr(self._agent, "agent_ids", None)
        if isinstance(roster, (list, tuple)):
            return list(roster)
        spaces = getattr(self._agent, "observation_spaces", None) or getattr(
            self._agent, "possible_observation_spaces", None
        )
        if isinstance(spaces, dict):
            return list(spaces.keys())
        return list(self._agent_ids)

    def _zero_obs_for(
        self, agent_id: str, fallback_shape, fallback_dtype
    ) -> np.ndarray:
        """Best-effort zero observation for ``agent_id``.

        Uses the wrapped algorithm's declared observation spaces when
        available (dict in older AgileRL, list aligned with ``agent_ids``
        in >=2.6); otherwise falls back to the shape/dtype of an
        already-observed agent.
        """
        spaces = getattr(self._agent, "observation_spaces", None) or getattr(
            self._agent, "possible_observation_spaces", None
        )
        space = None
        if isinstance(spaces, dict) and agent_id in spaces:
            space = spaces[agent_id]
        elif isinstance(spaces, (list, tuple)):
            roster = getattr(self._agent, "agent_ids", None)
            if roster is not None and agent_id in list(roster):
                idx = list(roster).index(agent_id)
                if idx < len(spaces):
                    space = spaces[idx]
        if space is not None:
            shape = getattr(space, "shape", fallback_shape) or fallback_shape
            dtype = getattr(space, "dtype", fallback_dtype) or fallback_dtype
            return np.zeros(shape, dtype=dtype)
        return np.zeros(fallback_shape, dtype=fallback_dtype)

    # Introspection helpers

    @property
    def agent_ids(self) -> list:
        return list(self._agent_ids)

    @property
    def underlying_agent(self) -> Any:
        """The wrapped AgileRL algorithm instance."""
        return self._agent

    @property
    def algorithm_name(self) -> str:
        return self._algorithm_name

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"AgileRLPolicyWrapper(algorithm={self._algorithm_name!r}, "
            f"agent_ids={self._agent_ids!r})"
        )

    # Snapshot

    def make_frozen_cpu_snapshot(self) -> "FrozenRLPolicyWrapper":
        """Create a picklable, CPU-only inference snapshot of this policy.

        Deep-copies every actor ``nn.Module`` from the underlying AgileRL
        agent, moves them to CPU, and sets them to ``eval()`` mode with
        gradient computation disabled.  The resulting
        :class:`FrozenRLPolicyWrapper` is safe to pickle across
        ``ProcessPoolExecutor`` spawn boundaries — it carries no
        optimizer state, replay buffer, or CUDA device references.

        Supports both *list-of-actors* layouts (``agent.actors``) and
        *single-actor* layouts (``agent.actor``).  Falls back gracefully
        when neither attribute is present.

        Returns:
            FrozenRLPolicyWrapper — a self-contained CPU inference policy.

        Raises:
            ImportError: if PyTorch is not installed.
            RuntimeError: if actor extraction or the deepcopy/cpu move fails.
        """
        import copy

        try:
            import torch  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "make_frozen_cpu_snapshot requires PyTorch. "
                "Install with: pip install torch"
            ) from exc

        actors_raw = getattr(self._agent, "actors", None)
        if hasattr(actors_raw, "values"):
            # ModuleDict or plain dict — extract the module values, not keys.
            actors: List[Any] = list(actors_raw.values())
        elif isinstance(actors_raw, (list, tuple)) and len(actors_raw) > 0:
            actors = list(actors_raw)
        else:
            single = getattr(self._agent, "actor", None)
            actors = [single] if single is not None else []
        if not actors:
            raise RuntimeError(
                f"make_frozen_cpu_snapshot: {type(self._agent).__name__!r} "
                "exposes neither 'actors' (non-empty list/dict) nor 'actor' "
                "(single module). Cannot extract actor networks."
            )

        try:
            frozen: List[Any] = []
            for a in actors:
                cpu_actor = copy.deepcopy(a).cpu()
                cpu_actor.eval()
                for p in cpu_actor.parameters():
                    p.requires_grad_(False)
                frozen.append(cpu_actor)
        except Exception as exc:
            raise RuntimeError(
                f"make_frozen_cpu_snapshot: failed to deepcopy/move actors "
                f"from {type(self._agent).__name__!r}: {exc}"
            ) from exc

        is_discrete = bool(getattr(self._agent, "discrete_actions", False))

        # Per-slot MultiDiscrete nvecs (aligned to agent_ids) so the frozen
        # wrapper can shape MultiDiscrete actions correctly. None for slots
        # whose action space is not MultiDiscrete.
        multidiscrete_nvecs = None
        spaces_map = getattr(self._agent, "possible_action_spaces", None)
        if isinstance(spaces_map, dict):
            slots = [
                tuple(int(x) for x in spaces_map[a].nvec)
                if a in spaces_map and hasattr(spaces_map[a], "nvec")
                else None
                for a in self._agent_ids
            ]
            if any(s is not None for s in slots):
                multidiscrete_nvecs = slots

        return FrozenRLPolicyWrapper(
            actors=frozen,
            agent_ids=self._agent_ids,
            full_roster=self._full_agent_roster(),
            algorithm_name=self._algorithm_name,
            is_discrete=is_discrete,
            multidiscrete_nvecs=multidiscrete_nvecs,
        )


class FrozenRLPolicyWrapper:
    """Inference-only RL policy — CPU-resident and picklable.

    Created by :meth:`AgileRLPolicyWrapper.make_frozen_cpu_snapshot`.
    Holds deepcopied CPU ``torch.nn.Module`` actor objects with no
    optimizer, replay buffer, or CUDA state, making it safe to pickle
    across ``ProcessPoolExecutor`` spawn boundaries.

    Implements the same round-robin / obs-cache :class:`PolicyProtocol`
    interface as :class:`AgileRLPolicyWrapper` so it can be used as a
    drop-in opponent policy in both serial and parallel evaluation paths.

    Args:
        actors: List of CPU-resident, eval-mode actor ``nn.Module`` objects
            (one per agent slot, as indexed by ``agent_ids``).
        agent_ids: The agent IDs this wrapper dispatches for (rotation order).
        full_roster: Full agent roster expected by the actor modules
            (may be equal to or a superset of ``agent_ids``).
        algorithm_name: Human-readable label (e.g. ``"matd3"``).
        is_discrete: If ``True``, a bare-tensor (deterministic) actor output
            is treated as logits over a discrete action space and the argmax
            is returned.  If ``False`` (default), the raw actor output is
            returned for continuous spaces.  Ignored when the actor returns a
            distribution tuple (stochastic actors already sample the action).
        multidiscrete_nvecs: Optional per-slot ``nvec`` tuples (aligned to
            ``agent_ids``); ``None`` for non-MultiDiscrete slots.  Used to
            shape MultiDiscrete actions as an integer array of length
            ``len(nvec)``.
    """

    def __init__(
        self,
        actors: List[Any],
        agent_ids: List[str],
        full_roster: List[str],
        algorithm_name: str,
        is_discrete: bool = False,
        multidiscrete_nvecs: "Optional[List[Optional[tuple]]]" = None,
    ) -> None:
        self._actors = list(actors)
        self._agent_ids = list(agent_ids)
        self._full_roster = list(full_roster)
        self._algorithm_name = algorithm_name
        self._is_discrete = is_discrete
        self._md_nvecs = list(multidiscrete_nvecs) if multidiscrete_nvecs else None
        self._call_index = 0
        self._obs_cache: Dict[str, np.ndarray] = {}

    # Policy protocol

    def get_action(self, observation: np.ndarray, **_: Any) -> Any:
        """Return a greedy action for the next agent in rotation.

        Accepts and ignores extra keyword arguments for drop-in
        compatibility with evaluators that pass flags like
        ``deterministic=True``.
        """
        import torch  # type: ignore[import]

        agent_id = self._agent_ids[self._call_index % len(self._agent_ids)]
        self._call_index += 1
        self._obs_cache[agent_id] = np.asarray(observation, dtype=np.float32)

        actor_idx = self._agent_ids.index(agent_id) % len(self._actors)
        actor = self._actors[actor_idx]

        obs_t = torch.from_numpy(self._obs_cache[agent_id]).unsqueeze(0)  # (1, obs_dim)
        with torch.no_grad():
            output = actor(obs_t)  # (1, action_dim) or (1, n_actions)

        slot_nvec = (
            self._md_nvecs[self._agent_ids.index(agent_id)]
            if self._md_nvecs is not None
            else None
        )

        # Stochastic actors (IPPO/PPO) return ``(action, log_prob, entropy)``
        # with the action already sampled and in the env's action space — an
        # int index (Discrete), an integer vector (MultiDiscrete), or a float
        # vector (Box). No argmax/rescale is needed in that case.
        if isinstance(output, tuple):
            sampled = output[0].squeeze(0).detach().cpu().numpy()
            if slot_nvec is not None:
                return np.asarray(sampled, dtype=np.int64).reshape(len(slot_nvec))
            if sampled.shape == ():
                return int(sampled) if self._is_discrete else float(sampled)
            return sampled

        # Rescale continuous actions from the network's output range to the
        # declared action-space bounds.  AgileRL's DeterministicActor stores
        # action_low / action_high / output_activation for this purpose;
        # mirroring the logic in DeterministicActor.rescale_action so that
        # FrozenRLPolicyWrapper produces actions that are valid in the env.
        if not self._is_discrete:
            action_low = getattr(actor, "action_low", None)
            action_high = getattr(actor, "action_high", None)
            output_activation = getattr(actor, "output_activation", None)
            if action_low is not None and action_high is not None and output_activation is not None:
                # Output bounds per activation: Tanh->[-1,1], Sigmoid/GumbelSoftmax->[0,1]
                _OUT_BOUNDS: Dict[str, tuple] = {
                    "Tanh": (-1.0, 1.0),
                    "Sigmoid": (0.0, 1.0),
                    "GumbelSoftmax": (0.0, 1.0),
                }
                bounds = _OUT_BOUNDS.get(output_activation)
                if bounds is not None:
                    min_out, max_out = bounds
                    low_t = action_low.to(output.device)
                    high_t = action_high.to(output.device)
                    output = low_t + (high_t - low_t) * (
                        (output - min_out) / (max_out - min_out)
                    )

        out_np: np.ndarray = output.squeeze(0).numpy()

        if self._is_discrete:
            return int(np.argmax(out_np))

        # Continuous: strip any remaining leading size-1 dims.
        while out_np.ndim > 0 and out_np.shape[0] == 1:
            out_np = out_np[0]
        if out_np.shape == ():
            return out_np.item()
        return out_np

    def reset(self) -> None:
        """Reset rotation counter and observation cache for a new episode."""
        self._call_index = 0
        self._obs_cache = {}

    # Pickle support — EvolvableMLP uses instance-bound methods that
    # standard pickle cannot serialize.  We serialise each actor with
    # cloudpickle (which handles closures / bound methods) and store the
    # resulting bytes in the state dict.  cloudpickle output is always
    # readable by standard pickle.loads, so the bytes survive
    # multiprocessing's ForkingPickler transparently.

    def __getstate__(self) -> dict:
        import io
        try:
            import cloudpickle as _cp  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "FrozenRLPolicyWrapper pickling requires cloudpickle. "
                "Install with: pip install cloudpickle"
            ) from exc

        serialized: List[bytes] = []
        for actor in self._actors:
            buf = io.BytesIO()
            _cp.dump(actor, buf)
            serialized.append(buf.getvalue())

        state = {k: v for k, v in self.__dict__.items() if k != "_actors"}
        state["_serialized_actors"] = serialized
        return state

    def __setstate__(self, state: dict) -> None:
        import pickle

        _ensure_agilerl_imports()

        serialized: List[bytes] = state.pop("_serialized_actors", [])
        self.__dict__.update(state)
        self._actors = [pickle.loads(b) for b in serialized]

    # Introspection

    @property
    def agent_ids(self) -> List[str]:
        return list(self._agent_ids)

    @property
    def algorithm_name(self) -> str:
        return self._algorithm_name

    @property
    def is_discrete(self) -> bool:
        return self._is_discrete

    def __repr__(self) -> str:
        return (
            f"FrozenRLPolicyWrapper({self._algorithm_name!r}, "
            f"agents={self._agent_ids}, discrete={self._is_discrete})"
        )
