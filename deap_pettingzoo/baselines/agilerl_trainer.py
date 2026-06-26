"""
AgileRL-backed trainers for the unified baseline API.

This module defines an abstract :class:`AgileRLTrainer` that orchestrates
multi-agent training through AgileRL's in-process APIs, plus four
concrete subclasses:

- :class:`MADDPGTrainer`       - Multi-Agent DDPG (off-policy)
- :class:`MATD3Trainer`        - Multi-Agent Twin-Delayed DDPG (off-policy)
- :class:`IPPOTrainer`         - Independent PPO (on-policy)
- :class:`GroupedIPPOTrainer`  - Prefix-grouped IPPO (on-policy)

The dependency on ``agilerl`` is imported lazily inside :meth:`train` so
that ``import deap_pettingzoo.baselines`` does not require the optional
``rl-baselines`` extra.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import time
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    ProgressCallback,
    ProgressEvent,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.baselines.agilerl_adapter import (
    build_parallel_env,
    build_vec_parallel_env,
    extract_space_metadata,
    translate_budget_to_env_steps,
)
from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper

logger = logging.getLogger(__name__)


# AgileRL bug-fix: misaligned state/advantage flattening in grouped IPPO

# AgileRL's ``IPPO._learn_individual`` flattens the rollout buffer using two
# functions with *different* memory layouts:
#
#   • ``vectorize_experiences_by_agent(rewards/log_probs/dones/values, dim=1)``
#     stacks agents on dim-1 then flattens ⟹ order (T, A, E) ⟹ flat index
#     k = t·(A·E) + a·E + e   (time outermost)
#
#   • ``concatenate_experiences_into_batches(states/actions)``
#     does ``torch.cat(dim=0)`` ⟹ order (A, T, E, obs) ⟹ flat index
#     k = a·(T·E) + t·E + e   (agent outermost)
#
# Example: with A=5 agents the two orderings agree only for the first E=8
# elements; every subsequent (state, advantage) pair is mismatched.
# The policy gradient is pure noise and the agent cannot learn.
#
# Fix: replace ``concatenate_experiences_into_batches`` with a version that
# uses ``torch.stack(dim=1)`` → reshape, matching the advantage layout.
# The patch is applied **lazily** the first time a ``GroupedIPPOTrainer`` is
# built (see ``_build_agent``), not at import time, and it fails loudly if the
# AgileRL function it targets has been renamed or re-signatured by a version
# bump — so a drifting dependency surfaces immediately instead of silently
# letting grouped IPPO train without the fix.

#: Name of the AgileRL function we monkey-patch.
_AGILERL_TARGET_NAME = "concatenate_experiences_into_batches"
#: Expected leading positional parameter names of that function. Verified
#: against AgileRL 2.6.1; a mismatch raises (and the regression test in
#: tests/test_agilerl_grouping_patch.py asserts the same contract).
_AGILERL_TARGET_PARAMS = ("experiences", "space", "actions")
#: AgileRL version range this patch was written and tested against
#: (see ``agilerl~=2.6`` in pyproject.toml).
_AGILERL_TESTED_VERSION = "2.6.x"


def _patched_concatenate_experiences_into_batches(experiences, space, actions=False):
    """Drop-in replacement for ``agilerl.utils.algo_utils.concatenate_experiences_into_batches``.

    Stacks agent tensors on dim-1 (time outermost) so the flat index order
    matches ``vectorize_experiences_by_agent(dim=1)`` used for advantages.
    """
    try:
        import torch
        import numpy as _np
        import agilerl.utils.algo_utils as _au

        tensors = [_au.experience_to_tensors(experiences[aid], space, actions)
                   for aid in experiences]

        def _stack_and_flatten(tlist, sp):
            import gymnasium.spaces as _spaces
            if isinstance(tlist[0], dict):
                return {k: _stack_and_flatten([t[k] for t in tlist], sp[k])
                        for k in tlist[0]}
            if isinstance(tlist[0], tuple):
                return tuple(_stack_and_flatten([t[i] for t in tlist], sp[i])
                             for i in range(len(tlist[0])))
            # tlist: A tensors of shape (T, E?, *features)
            # stack dim=1 → (T, A, E?, *features), matching advantage layout.
            stacked = torch.stack(tlist, dim=1)
            reshaped = stacked.reshape(-1, *sp.shape)
            for squeeze_dim in [0, -1]:
                if reshaped.size(squeeze_dim) == 1:
                    reshaped = reshaped.squeeze(squeeze_dim)
            return reshaped

        return _stack_and_flatten(tensors, space)
    except Exception as exc:
        # Runtime safety net: never crash a long training run if the
        # corrected flattening hits an unexpected layout. Emit a loud,
        # *capturable* warning (not a bare print) so the degraded state is
        # visible in logs/tests; grouped IPPO may not learn this update.
        warnings.warn(
            "AgileRL grouping patch failed at run time "
            f"({type(exc).__name__}: {exc}); falling back to the original "
            "flattening function. Grouped IPPO may not learn this update.",
            RuntimeWarning,
            stacklevel=2,
        )
        import agilerl.utils.algo_utils as _au
        return _au._orig_concatenate_experiences_into_batches(experiences, space, actions)


def _apply_agilerl_grouping_patch() -> None:
    """Idempotently monkey-patch AgileRL's flattening function.

    Called lazily from :meth:`GroupedIPPOTrainer._build_agent` rather than at
    import time, so importing this module never has side effects on AgileRL.
    Installs the patch on first call and is a no-op on subsequent calls.

    Raises:
        RuntimeError: If ``agilerl`` is not installed, or if the target
            function is missing / has an unexpected signature — i.e. the
            installed AgileRL drifted from the tested
            :data:`_AGILERL_TESTED_VERSION` series. Failing loudly here is
            deliberate: a silently-unpatched grouped IPPO "runs but does not
            learn", which is worse than a clear error.
    """
    try:
        import agilerl.utils.algo_utils as _au
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "GroupedIPPOTrainer requires the optional 'agilerl' dependency. "
            "Install the 'rl-baselines' extra."
        ) from exc

    if getattr(_au, "_grouping_patch_applied", False):
        return

    target = getattr(_au, _AGILERL_TARGET_NAME, None)
    if not callable(target):
        raise RuntimeError(
            f"AgileRL grouping-patch target '{_AGILERL_TARGET_NAME}' is missing "
            "from agilerl.utils.algo_utils. The installed AgileRL has drifted "
            f"from the tested {_AGILERL_TESTED_VERSION} series; update the patch "
            "in deap_pettingzoo/baselines/agilerl_trainer.py."
        )
    actual_params = tuple(inspect.signature(target).parameters)
    if actual_params[: len(_AGILERL_TARGET_PARAMS)] != _AGILERL_TARGET_PARAMS:
        raise RuntimeError(
            f"AgileRL '{_AGILERL_TARGET_NAME}' signature changed: expected "
            f"leading parameters {_AGILERL_TARGET_PARAMS}, got {actual_params}. "
            f"The installed AgileRL has drifted from the tested "
            f"{_AGILERL_TESTED_VERSION} series; update the grouping patch in "
            "deap_pettingzoo/baselines/agilerl_trainer.py."
        )

    _au._orig_concatenate_experiences_into_batches = target
    _au.concatenate_experiences_into_batches = (
        _patched_concatenate_experiences_into_batches
    )
    # Also patch the name already bound inside the IPPO module.
    try:
        import agilerl.algorithms.ippo as _ippo_mod
        _ippo_mod.concatenate_experiences_into_batches = (
            _patched_concatenate_experiences_into_batches
        )
    except ImportError:  # pragma: no cover - IPPO module always present in 2.6
        pass
    _au._grouping_patch_applied = True


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AgileRLTrainer(BaselineTrainer):
    """Abstract :class:`BaselineTrainer` backed by AgileRL.

    Subclasses must set:

    - :attr:`ALGORITHM_NAME`  - short label (e.g. ``'maddpg'``).
    - :attr:`ALGORITHM_IMPORT`  - ``(module_path, class_name)`` tuple
      used for the lazy import inside :meth:`train`.
    - :attr:`ON_POLICY`  - ``True`` for PPO-family algorithms,
      ``False`` for DDPG-family.
    - :meth:`get_capabilities`  - per-algorithm capability flags.
    
    Notes:
        AgileRL training is rollout-based, not generation-based, so the
        default cadence is the largest sensible bucket. Currently we only
        emit a single end-of-training rollout regardless of cadence. 
        The ClassVar value is kept for API parity with other trainers.
    """

    ALGORITHM_NAME: str = "agilerl"
    ALGORITHM_IMPORT: Tuple[str, str] = ("agilerl", "")
    ON_POLICY: bool = False

    # Default hyperparameters; subclasses may override.
    DEFAULT_PARAMS: Dict[str, Any] = {
        "batch_size": 64,
        "memory_size": 10_000,
        "learning_delay": 500,
        # Off-policy learning frequency: call agent.learn() once every ``learn_step`` env steps after ``learning_delay``.
        "learn_step": 100,
        "rollout_steps": 256,
        "device": "cpu",
    }

    @staticmethod
    def describe_hyperparameters():
        """Declarative hyperparameter overlay consumed by the catalog.

        The four AgileRL subclasses share one hyperparameter surface. Only
        ``device`` / ``verbose`` are constructor arguments
        (:class:`~deap_pettingzoo.catalog.schema.ParamTarget.TRAINER`); every
        other knob is merged over :attr:`DEFAULT_PARAMS` and passed through
        ``algorithm_params``. Off-policy-only knobs (``memory_size``,
        ``learning_delay``, ``learn_step``) and the on-policy ``rollout_steps``
        are both surfaced; the inactive ones are ignored by the running
        algorithm. Optimiser passthroughs (``lr_actor`` / ``lr_critic`` /
        ``gamma`` / ``tau``) default to AgileRL's own values
        (1e-3 / 1e-2 / 0.95 / 0.01).
        """
        from deap_pettingzoo.catalog.schema import (
            ParamSpec,
            ParamTarget,
            ParamType,
        )

        I, F, B, E = (
            ParamType.INT,
            ParamType.FLOAT,
            ParamType.BOOL,
            ParamType.ENUM,
        )
        TR, AP = ParamTarget.TRAINER, ParamTarget.ALGORITHM_PARAMS
        return (
            ParamSpec("batch_size", I, 64, minimum=1, step=1, target=AP,
                      group="Learning",
                      help="Mini-batch size for gradient updates."),
            ParamSpec("num_envs", I, 1, minimum=1, step=1, target=AP,
                      group="Parallelism",
                      help="Vectorised environments for rollout collection."),
            ParamSpec("memory_size", I, 10_000, minimum=1, step=1000, target=AP,
                      group="Learning", advanced=True,
                      help="Replay-buffer capacity (off-policy only)."),
            ParamSpec("learning_delay", I, 500, minimum=0, step=1, target=AP,
                      group="Learning", advanced=True,
                      help="Env steps before learning begins (off-policy)."),
            ParamSpec("learn_step", I, 100, minimum=1, step=1, target=AP,
                      group="Learning", advanced=True,
                      help="Env steps between learn() calls (off-policy)."),
            ParamSpec("rollout_steps", I, 256, minimum=1, step=1, target=AP,
                      group="Learning", advanced=True,
                      help="Steps collected per update (on-policy / IPPO)."),
            ParamSpec("lr_actor", F, 1e-3, minimum=0.0, step=1e-4, target=AP,
                      group="Learning", advanced=True,
                      help="Actor learning rate (AgileRL default: 1e-3)."),
            ParamSpec("lr_critic", F, 1e-2, minimum=0.0, step=1e-4, target=AP,
                      group="Learning", advanced=True,
                      help="Critic learning rate (AgileRL default: 1e-2)."),
            ParamSpec("gamma", F, 0.95, minimum=0.0, maximum=1.0, step=0.01, target=AP,
                      group="Learning", advanced=True,
                      help="Discount factor (AgileRL default: 0.95)."),
            ParamSpec("tau", F, 0.01, minimum=0.0, maximum=1.0, step=0.001, target=AP,
                      group="Learning", advanced=True,
                      help="Target-network soft-update rate (AgileRL default: 0.01)."),
            ParamSpec("device", E, "cpu", target=TR, group="Runtime",
                      choices=("cpu", "cuda"),
                      help="Torch device for training."),
            ParamSpec("verbose", B, False, target=TR, group="Runtime",
                      advanced=True,
                      help="Print per-interval training progress."),
            ParamSpec("seed", I, None, minimum=0, step=1, target=AP,
                      group="Run", nullable=True,
                      help="Random seed for this run (None = unseeded)."),
            ParamSpec("max_steps_per_episode", I, None, minimum=1, step=1,
                      target=AP, group="Run", nullable=True,
                      help="Hard cap on env steps per evaluation episode."),
        )

    def __init__(self, device: str = "cpu", verbose: bool = False) -> None:
        # Fall back to CPU when a CUDA device is requested but unavailable
        # (e.g. the CPU-only torch build in the service container) so AgileRL
        # doesn't crash with "Torch not compiled with CUDA enabled".
        from deap_pettingzoo.utils.device_check import resolve_device

        self.device = resolve_device(device)
        self.verbose = verbose

    # BaselineTrainer interface

    def train(
        self,
        config: TrainingConfig,
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """Train an AgileRL algorithm under the given budget.

        Raises:
            ImportError: If the optional ``rl-baselines`` extra is not
                installed.
            ValueError: If the target environment is a turn-based AEC
                game (``api='aec_masked'`` / ``'aec_sequential'``) and
                this algorithm does not support turn-based training
                (e.g. MADDPG / MATD3), or if ``config.env_name`` is not
                registered in the environment registry (propagated from
                :func:`~deap_pettingzoo.baselines.agilerl_adapter.build_parallel_env`).
            TypeError: If the registered factory's ``api`` declaration is
                inconsistent (propagated from
                :func:`~deap_pettingzoo.baselines.agilerl_adapter.build_parallel_env`).
        """
        # Fast-fail on turn-based envs for algos that cannot handle them.
        if not self.get_capabilities().get("supports_turn_based", True):
            try:
                from deap_pettingzoo.environments.registry import get_env_spec
                spec_api = get_env_spec(config.env_name).api
            except Exception:  # pragma: no cover - registry lookup fallback
                spec_api = None
            if spec_api in ("aec_masked", "aec_sequential"):
                raise ValueError(
                    f"Algorithm '{self.ALGORITHM_NAME}' cannot train on "
                    f"turn-based AEC env '{config.env_name}' "
                    f"(api={spec_api!r}). The replay buffer / critic "
                    "structure assumes simultaneous-step execution. "
                    "Use EA, CMA-ES, or IPPO for turn-based games."
                )

        alg_cls = self._import_algorithm()

        params = dict(self.DEFAULT_PARAMS)
        params.update(config.algorithm_params or {})
        seed = int(params.pop("seed", 0))
        self._seed_everything(seed)

        # ``num_envs`` controls vectorized rollout collection.
        num_envs = int(params.pop("num_envs", 1))
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs!r}")

        if num_envs > 1:
            env = build_vec_parallel_env(
                config.env_name,
                env_config=config.env_config or None,
                num_envs=num_envs,
            )
        else:
            env = build_parallel_env(
                config.env_name, env_config=config.env_config or None
            )
        obs_spaces, act_spaces, agent_ids = extract_space_metadata(env)

        # Split params: constructor vs. loop knobs.
        # Note: batch_size and learn_step are accepted by both on-policy
        # (IPPO) and off-policy (MADDPG/MATD3) constructors, so they
        # go to ctor_kwargs AND remain in params for the training loop.
        loop_keys = {
            "memory_size",
            "learning_delay",
            "rollout_steps",
            "max_steps_per_episode",
        }
        ctor_kwargs = {k: v for k, v in params.items() if k not in loop_keys}
        ctor_kwargs["device"] = self.device  # Always use trainer's device, not DEFAULT_PARAMS "cpu"

        agent = self._build_agent(
            alg_cls, obs_spaces, act_spaces, agent_ids, ctor_kwargs
        )

        # Build the policy wrapper *before* training so that mid-training
        # recording can call into the live agent (the wrapper holds a
        # reference to ``agent`` and observes its weight updates).
        policy = AgileRLPolicyWrapper(agent, agent_ids, self.ALGORITHM_NAME)

        budget_steps = self._budget_to_steps(config)

        # Persist results via ExperimentLogger unless ``config.output_dir is None``.
        # NOTE: A single ``ExperimentLogger`` instance is built here and reused for
        # both end-of-training and mid-training frame recording as well as for
        # metrics/summary persistence so that all artefacts (frames/,
        # metrics.csv, summary.json) land under the same ``run_dir``.
        from deap_pettingzoo.baselines._logging_helpers import (
            build_logger,
            write_summary,
        )
        from deap_pettingzoo.baselines.recording import RecordingController

        with build_logger(config, algorithm=self.ALGORITHM_NAME) as log:
            run_dir_str = (
                str(log.run_dir) if config.output_dir is not None else None
            )

            # Unified champion-rollout recording. The cadence is decided by
            # the controller from the run's RecordingMode on the shared
            # env-step axis (RL has no generations). The champion is the live
            # ``policy`` wrapper around ``agent`` (weights update in place).
            # Use the env's natural episode length for the recording rollout,
            # NOT max_steps_per_episode from algorithm_params — that's a
            # training knob (episode-reset frequency) that users often set
            # to small values for off-policy efficiency.
            _env_cfg = config.env_config or {}
            _rec_max_steps = int(
                config.recording.max_frames_per_episode
                or _env_cfg.get("max_cycles")
                or _env_cfg.get("max_time_steps")
                or 500
            )
            self._recorder = RecordingController(
                config, log,
                max_steps=_rec_max_steps,
                seed=seed,
            )

            def _champion():
                return policy

            start = time.time()
            if self.ON_POLICY:
                metrics = self._run_on_policy(
                    env, agent, agent_ids, budget_steps, params, seed,
                    champion=_champion,
                    progress_callback=progress_callback,
                )
            else:
                metrics = self._run_off_policy(
                    env, agent, agent_ids, budget_steps, params, seed,
                    champion=_champion,
                    progress_callback=progress_callback,
                )
            training_time = time.time() - start
            try:
                env.close()
            except SystemExit:
                # Some third-party envs (e.g. marl-aquarium) call sys.exit() from close().
                # Suppress to keep the process alive.
                pass
            except Exception:  # pragma: no cover
                pass

            # Always emit one end-of-training rollout — guarantees at least one
            # artefact even on tight budgets where no mid-run checkpoint fired.
            self._recorder.finalize(
                _champion, iteration=budget_steps, env_steps=budget_steps
            )
            self._recorder.close()

            ep_rewards = list(metrics.get("episode_rewards", []) or [])
            for idx, r in enumerate(ep_rewards):
                log.row(
                    "episodes",
                    {
                        "episode_idx": idx,
                        "seed": seed,
                        "reward": float(r),
                        "length": -1,
                        "success": -1,
                    },
                )
            for entry in metrics.get("metrics_history", []):
                log.row("metrics", entry)
            from deap_pettingzoo.utils.logging import save_agilerl_agent
            save_agilerl_agent(
                log,
                agent,
                extra={
                    "algorithm": self.ALGORITHM_NAME,
                    "env_name": config.env_name,
                    "agent_ids": list(agent_ids),
                },
            )
            write_summary(
                log,
                {
                    "algorithm": self.ALGORITHM_NAME,
                    "num_episodes": int(metrics.get("num_episodes", len(ep_rewards))),
                    "env_steps": int(metrics.get("env_steps", 0)),
                    "mean_episode_reward": (
                        float(np.mean(ep_rewards)) if ep_rewards else None
                    ),
                    "mean_loss": metrics.get("mean_loss"),
                    "on_policy": self.ON_POLICY,
                },
                training_time_s=training_time,
            )

        return TrainingResult(
            algorithm=self.ALGORITHM_NAME,
            policy=policy,
            metrics=metrics,
            metadata={
                "env_name": config.env_name,
                "budget_mode": config.budget_mode.value,
                "budget_value": config.budget_value,
                "agent_ids": agent_ids,
                "on_policy": self.ON_POLICY,
                "algorithm_params": params,
                "run_dir": run_dir_str,
            },
            seed=seed,
            training_time=training_time,
        )

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        return mode in {BudgetMode.ENV_STEPS, BudgetMode.EVAL_EPISODES}

    def get_capabilities(self) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    # Hooks / helpers

    def _import_algorithm(self):
        """Lazily import the AgileRL algorithm class."""
        module_path, class_name = self.ALGORITHM_IMPORT
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                f"AgileRL is required for {self.ALGORITHM_NAME.upper()} "
                "training. Install with: "
                "pip install 'deap-pettingzoo[rl-baselines]'"
            ) from exc
        return getattr(module, class_name)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        from deap_pettingzoo.utils.rng import seed_everything as _se
        _se(seed)

    def _budget_to_steps(self, config: TrainingConfig) -> int:
        if config.budget_mode == BudgetMode.WALL_CLOCK:
            # Soft cap: rely on a large step budget and let the caller's timer stop us; we return a very large number here.
            return 10**9
        return translate_budget_to_env_steps(config)

    def _build_agent(
        self,
        alg_cls,
        obs_spaces,
        act_spaces,
        agent_ids: List[str],
        ctor_kwargs: Dict[str, Any],
    ):
        """Instantiate the AgileRL algorithm. Subclasses may override
        to inject algorithm-specific defaults (e.g. a centralized
        critic configuration)."""
        return alg_cls(
            observation_spaces=obs_spaces,
            action_spaces=act_spaces,
            agent_ids=agent_ids,
            **ctor_kwargs,
        )

    @staticmethod
    def _unwrap_actions_for_env(actions: Dict[str, Any]) -> Dict[str, Any]:
        """Strip a leading vectorization dim of size 1 from each action.

        AgileRL always returns batched actions (shape ``(1, ...)`` for a
        single non-vectorized env). PettingZoo's parallel API expects
        unbatched actions (a scalar for ``Discrete`` spaces, a 1-D array
        for ``Box`` spaces), so we squeeze the leading dim per-agent.
        """
        unwrapped: Dict[str, Any] = {}
        for agent_id, act in actions.items():
            arr = np.asarray(act)
            if arr.ndim > 0 and arr.shape[0] == 1:
                arr = arr[0]
                # Discrete actions should be Python ints for PettingZoo.
                if arr.shape == ():
                    arr = arr.item()
            unwrapped[agent_id] = arr
        return unwrapped

    @staticmethod
    def _mean_loss(loss: Any) -> Optional[float]:
        """Reduce AgileRL's heterogeneous ``learn()`` return to a scalar.

        AgileRL returns one of:
          * ``None`` (some on-policy paths),
          * ``float`` / ``torch.Tensor`` scalar,
          * ``dict[str, float]`` (per-agent total loss),
          * ``dict[str, tuple[float, float, ...]]`` (per-agent
            ``(actor_loss, critic_loss, ...)`` — current MADDPG path).

        Returns the mean across all leaf scalars, or ``None`` if the
        payload is empty / non-numeric.
        """
        if loss is None:
            return None

        scalars: List[float] = []

        def _flatten(node: Any) -> None:
            if node is None:
                return
            if hasattr(node, "item") and not isinstance(node, (bytes, str)):
                # Tensor / 0-d ndarray.
                try:
                    scalars.append(float(node.item()))
                    return
                except Exception:  # pragma: no cover
                    pass
            if isinstance(node, (int, float)):
                scalars.append(float(node))
                return
            if isinstance(node, dict):
                for v in node.values():
                    _flatten(v)
                return
            if isinstance(node, (list, tuple)):
                for v in node:
                    _flatten(v)
                return
            # Fallback: best-effort cast.
            try:
                scalars.append(float(node))
            except (TypeError, ValueError):
                return

        _flatten(loss)
        if not scalars:
            return None
        return float(np.mean(scalars))

    @staticmethod
    def _nan_placeholder_for_space(space: Any) -> Any:
        """Return a float NaN placeholder matching a Gymnasium space shape."""
        try:
            from gymnasium import spaces
        except ImportError:  # pragma: no cover
            spaces = None

        if spaces is not None and isinstance(space, spaces.Dict):
            return {
                key: AgileRLTrainer._nan_placeholder_for_space(subspace)
                for key, subspace in space.spaces.items()
            }
        if spaces is not None and isinstance(space, spaces.Tuple):
            return tuple(
                AgileRLTrainer._nan_placeholder_for_space(subspace)
                for subspace in space.spaces
            )

        shape = tuple(getattr(space, "shape", ()) or ())
        if not shape:
            return np.array([np.nan], dtype=np.float32)
        return np.full(shape, np.nan, dtype=np.float32)

    @staticmethod
    def _done_array(
        terminated: Any,
        truncated: Any,
        *,
        is_vec: bool,
        num_envs: int,
    ) -> np.ndarray:
        """Convert termination/truncation payloads to AgileRL-style done arrays."""
        if is_vec:
            term = np.asarray(terminated, dtype=np.float32).reshape(num_envs)
            trunc = np.asarray(truncated, dtype=np.float32).reshape(num_envs)
            valid_mask = ~(np.isnan(term) | np.isnan(trunc))
            done = np.full(num_envs, np.nan, dtype=np.float32)
            done[valid_mask] = np.logical_or(
                term[valid_mask].astype(bool),
                trunc[valid_mask].astype(bool),
            ).astype(np.float32)
            return done

        return np.array([float(bool(terminated) or bool(truncated))], dtype=np.float32)

    @classmethod
    def _build_done_dict(
        cls,
        terminations: Dict[str, Any],
        truncations: Dict[str, Any],
        agent_ids: List[str],
        *,
        is_vec: bool,
        num_envs: int,
    ) -> Dict[str, np.ndarray]:
        """Build per-agent done arrays, preserving NaN inactive markers for vec envs."""
        if is_vec:
            missing_term: Any = np.full(num_envs, np.nan, dtype=np.float32)
            missing_trunc: Any = np.full(num_envs, np.nan, dtype=np.float32)
        else:
            missing_term = True
            missing_trunc = False

        return {
            agent_id: cls._done_array(
                terminations.get(agent_id, missing_term),
                truncations.get(agent_id, missing_trunc),
                is_vec=is_vec,
                num_envs=num_envs,
            )
            for agent_id in agent_ids
        }

    @staticmethod
    def _completed_env_mask(
        done_dict: Dict[str, np.ndarray],
        agent_ids: List[str],
        num_envs: int,
    ) -> np.ndarray:
        """Return the per-env episode-complete mask (all agents done or inactive)."""
        if not agent_ids:
            return np.ones(num_envs, dtype=bool)

        stacked = np.stack(
            [np.asarray(done_dict[agent_id], dtype=np.float32).reshape(num_envs)
             for agent_id in agent_ids],
            axis=0,
        )
        return np.all(np.isnan(stacked) | stacked.astype(bool), axis=0)

    # Training loops

    def _run_off_policy(
        self,
        env,
        agent,
        agent_ids: List[str],
        budget_steps: int,
        params: Dict[str, Any],
        seed: int,
        *,
        champion: Optional[Any] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Hand-rolled off-policy loop using AgileRL's replay buffer.

        Note:
            AgileRL's ``get_action`` returns a ``(actions, raw_actions)``
            tuple. The two dicts serve different purposes:

            * ``actions`` — sent to ``env.step``. Discrete spaces expect
              Python ints for a single env, while vec envs expect batched
              ``(num_envs, ...)`` actions.
            * ``raw_actions`` — stored in the replay buffer. These must
              keep their leading vec axis in the vectorized path because
              ``MultiAgentReplayBuffer.save_to_memory(...,
              is_vectorised=True)`` reorganizes one transition per worker.

            For the single-env path we still squeeze the leading size-1
            axis before storing raw actions; otherwise ``MADDPG.learn()`` /
            ``MATD3.learn()`` see 3-D actions next to 2-D observations.
        """
        from agilerl.components.multi_agent_replay_buffer import (  # noqa: WPS433
            MultiAgentReplayBuffer,
        )

        memory = MultiAgentReplayBuffer(
            memory_size=int(params["memory_size"]),
            field_names=["obs", "action", "reward", "next_obs", "done"],
            agent_ids=agent_ids,
            device=self.device,
        )
        batch_size = int(params["batch_size"])
        learning_delay = int(params["learning_delay"])
        learn_step = max(1, int(params.get("learn_step", 1)))
        num_envs = int(getattr(env, "num_envs", 1))
        is_vec = num_envs > 1
        progress_interval = max(1, budget_steps // 300) if self.verbose else 0
        progress_emit_interval = max(1, budget_steps // 100)
        loop_start = time.time()

        obs, info = env.reset(seed=seed)
        episode_rewards: List[float] = []
        ep_reward = np.zeros(num_envs, dtype=np.float64)
        losses: List[float] = []
        steps_done = 0
        metrics_history: List[Dict[str, Any]] = []

        def _expand_placeholder(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: _expand_placeholder(subvalue)
                    for key, subvalue in value.items()
                }
            if isinstance(value, tuple):
                return tuple(_expand_placeholder(subvalue) for subvalue in value)

            arr = np.asarray(value)
            return np.repeat(arr[None, ...], num_envs, axis=0)

        def _placeholder_for_space(space: Any) -> Any:
            placeholder = self._nan_placeholder_for_space(space)
            if not is_vec:
                return placeholder
            return _expand_placeholder(placeholder)

        def _complete_obs_dict(observation_dict: Dict[str, Any]) -> Dict[str, Any]:
            return {
                agent_id: observation_dict.get(
                    agent_id,
                    _placeholder_for_space(env.observation_space(agent_id)),
                )
                for agent_id in agent_ids
            }

        def _complete_action_dict(action_dict: Dict[str, Any]) -> Dict[str, Any]:
            completed: Dict[str, Any] = {}
            for agent_id in agent_ids:
                if agent_id in action_dict:
                    arr = np.asarray(action_dict[agent_id])
                    if not is_vec and arr.ndim > 0 and arr.shape[0] == 1:
                        arr = arr[0]
                    completed[agent_id] = arr
                else:
                    completed[agent_id] = _placeholder_for_space(
                        env.action_space(agent_id)
                    )
            return completed

        def _complete_reward_dict(reward_dict: Dict[str, Any]) -> Dict[str, Any]:
            if is_vec:
                return {
                    agent_id: np.asarray(
                        reward_dict.get(
                            agent_id,
                            np.full(num_envs, np.nan, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    ).reshape(num_envs)
                    for agent_id in agent_ids
                }

            return {
                agent_id: float(reward_dict[agent_id])
                if agent_id in reward_dict else np.nan
                for agent_id in agent_ids
            }

        def _sum_rewards_per_env(reward_dict: Dict[str, Any]) -> np.ndarray:
            if not is_vec:
                return np.array(
                    [float(np.nansum(list(reward_dict.values())))],
                    dtype=np.float64,
                )

            total = np.zeros(num_envs, dtype=np.float64)
            for agent_id in agent_ids:
                rewards_a = np.asarray(
                    reward_dict.get(
                        agent_id,
                        np.full(num_envs, np.nan, dtype=np.float32),
                    ),
                    dtype=np.float64,
                ).reshape(num_envs)
                total += np.where(np.isnan(rewards_a), 0.0, rewards_a)
            return total

        while steps_done < budget_steps:
            prev_steps = steps_done
            action_result = agent.get_action(obs, infos=info)
            actions, raw_actions = (
                action_result if isinstance(action_result, tuple) else (action_result, action_result)
            )
            env_actions = actions if is_vec else self._unwrap_actions_for_env(actions)
            next_obs, reward, termination, truncation, info = env.step(env_actions)

            full_obs = _complete_obs_dict(obs)
            full_actions = _complete_action_dict(raw_actions)
            full_reward = _complete_reward_dict(reward)
            full_next_obs = _complete_obs_dict(next_obs)
            next_done = self._build_done_dict(
                termination,
                truncation,
                agent_ids,
                is_vec=is_vec,
                num_envs=num_envs,
            )

            memory.save_to_memory(
                full_obs,
                full_actions,
                full_reward,
                full_next_obs,
                next_done,
                is_vectorised=is_vec,
            )

            ep_reward += _sum_rewards_per_env(full_reward)
            steps_done += num_envs if is_vec else 1

            if len(memory) >= batch_size:
                learn_calls_this_step = 0
                for env_step in range(prev_steps, steps_done):
                    if (
                        env_step >= learning_delay
                        and (env_step - learning_delay) % learn_step == 0
                    ):
                        learn_calls_this_step += 1

                for _ in range(learn_calls_this_step):
                    experiences = memory.sample(batch_size)
                    try:
                        loss = agent.learn(experiences)
                        mean_loss = self._mean_loss(loss)
                        if mean_loss is not None:
                            losses.append(mean_loss)
                    except Exception as exc:  # pragma: no cover - version drift
                        logger.warning(
                            "[%s] learn() failed: %s",
                            self.ALGORITHM_NAME, exc, exc_info=True,
                        )

            env_done = self._completed_env_mask(next_done, agent_ids, num_envs)
            for i, finished in enumerate(env_done):
                if finished:
                    episode_rewards.append(float(ep_reward[i]))
                    ep_reward[i] = 0.0

            if not is_vec and bool(env_done[0]):
                obs, info = env.reset()
            else:
                obs = next_obs

            progress_steps = min(steps_done, budget_steps)
            if progress_interval and (
                progress_steps // progress_interval > prev_steps // progress_interval
                or progress_steps >= budget_steps
            ):
                elapsed = time.time() - loop_start
                pct = 100.0 * progress_steps / budget_steps
                mean_r = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
                last_r = episode_rewards[-1] if episode_rewards else float("nan")
                print(
                    f"[{self.ALGORITHM_NAME}] step {progress_steps}/{budget_steps} "
                    f"({pct:5.1f}%) | episodes={len(episode_rewards)} | "
                    f"mean_rew={mean_r:.2f} | last_reward={last_r:.2f} | "
                    f"learn_calls={len(losses)} | "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            # Fire the unified progress+recording hook on the emit cadence,
            # independent of whether a progress_callback is set (recording must
            # still happen on a headless run). _on_progress no-ops the emit when
            # the callback is None and lets the controller decide on recording.
            if (
                progress_steps // progress_emit_interval
                > prev_steps // progress_emit_interval
                or progress_steps >= budget_steps
            ):
                elapsed = time.time() - loop_start
                _emit_mean = (
                    float(np.mean(episode_rewards[-10:]))
                    if episode_rewards else None
                )
                _emit_best = (
                    float(max(episode_rewards)) if episode_rewards else None
                )
                _window = episode_rewards[-10:] if episode_rewards else []
                metrics_history.append({
                    "iteration": len(metrics_history),
                    "env_steps": int(progress_steps),
                    "wall_time_s": elapsed,
                    "mean": _emit_mean,
                    "std": float(np.std(_window)) if _window else None,
                    "min": float(min(_window)) if _window else None,
                    "max": _emit_best,
                    "mean_loss": float(losses[-1]) if losses else None,
                })
                self._on_progress(
                    progress_callback,
                    ProgressEvent(
                        unit="env_step",
                        current=int(progress_steps),
                        total=int(budget_steps),
                        env_steps=int(progress_steps),
                        best_fitness=_emit_best,
                        mean_fitness=_emit_mean,
                        wall_time_s=elapsed,
                    ),
                    champion=champion,
                )

        if not episode_rewards:
            episode_rewards.append(float(ep_reward.sum()))
        else:
            for i in range(num_envs):
                if abs(ep_reward[i]) > 0.0:
                    episode_rewards.append(float(ep_reward[i]))

        return {
            "episode_rewards": episode_rewards,
            "num_episodes": len(episode_rewards),
            "env_steps": min(steps_done, budget_steps),
            "mean_loss": float(np.mean(losses)) if losses else None,
            "metrics_history": metrics_history,
        }

    def _run_on_policy(
        self,
        env,
        agent,
        agent_ids: List[str],
        budget_steps: int,
        params: Dict[str, Any],
        seed: int,
        *,
        champion: Optional[Any] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """On-policy rollout loop for IPPO.

        Accumulates ``(states, actions, log_probs, rewards, dones, values)``
        per agent over ``learn_step`` env steps, then calls
        ``agent.learn(experiences)`` with the properly structured tuple.
        This matches the data-flow in AgileRL's official
        ``train_multi_agent_on_policy`` reference implementation.

        Supports both a single PettingZoo parallel env (``num_envs=1``)
        and an :class:`AsyncPettingZooVecEnv` (``num_envs>1``). The vec
        path:

        * Skips the manual ``[None, ...]`` expansion (vec env outputs
          already carry a leading ``num_envs`` axis);
        * Skips the manual ``env.reset()`` on episode end (vec env
          workers auto-reset finished sub-envs internally);
        * Tracks per-sub-env episode-reward accumulators and emits one
          ``episode_rewards`` entry per finished sub-env episode.
        """
        # Rollout chunk size: prefer user-specified learn_step, then agent attribute.
        learn_step = int(params.get("learn_step", getattr(agent, "learn_step", 256)))

        # Vec env detection. AsyncPettingZooVecEnv exposes ``num_envs``;
        # a raw PettingZoo parallel env does not.
        num_envs = int(getattr(env, "num_envs", 1))
        is_vec = num_envs > 1

        episode_rewards: List[float] = []
        losses: List[float] = []
        steps_done = 0
        step_size = num_envs if is_vec else 1
        metrics_history: List[Dict[str, Any]] = []

        # Per-(sub-)env running episode-reward accumulator. For the
        # single-env case this is a scalar in a length-1 array; for the
        # vec case it tracks each sub-env independently so we can emit
        # an episode reward each time a sub-env terminates.
        ep_reward = np.zeros(num_envs, dtype=np.float64)

        obs, info = env.reset(seed=seed)
        # Per-agent done flags: shape ``(num_envs,)`` — zero at episode 0.
        done: Dict[str, Any] = {
            a: np.zeros(num_envs, dtype=np.float32) for a in agent_ids
        }

        # Helper: ensure obs has a leading num_envs axis. Vec envs
        # already provide it; single envs return ``(obs_dim,)``.
        def _batch_obs(o: Any) -> np.ndarray:
            arr = np.asarray(o)
            return arr if is_vec else arr[None, ...]

        # Grouped IPPO expects a fixed ``(num_envs, ...)`` batch per agent.
        # When PettingZoo workers mark an agent inactive they emit all-NaN
        # placeholders; pass zeros to ``get_action`` and the stored state tensor
        # while preserving NaN rewards/dones to mark inactivity downstream.
        def _sanitize_obs(o: Any) -> np.ndarray:
            arr = np.asarray(o, dtype=np.float32)
            if not np.isnan(arr).any():
                return arr

            clean = np.array(arr, copy=True)
            if clean.ndim <= 1:
                if np.isnan(clean).all():
                    clean[...] = 0.0
                return clean

            inactive_mask = np.isnan(clean).all(axis=tuple(range(1, clean.ndim)))
            clean[inactive_mask] = 0.0
            return clean

        # Helper: per-sub-env reward summed across agents, shape (num_envs,).
        def _sum_rewards_per_env(reward_dict: Dict[str, Any]) -> np.ndarray:
            if not is_vec:
                return np.array(
                    [float(np.nansum([reward_dict.get(a, np.nan) for a in agent_ids]))],
                    dtype=np.float64,
                )

            total = np.zeros(num_envs, dtype=np.float64)
            for a in agent_ids:
                rewards_a = np.asarray(
                    reward_dict.get(a, np.full(num_envs, np.nan, dtype=np.float32)),
                    dtype=np.float64,
                ).reshape(num_envs)
                total += np.where(np.isnan(rewards_a), 0.0, rewards_a)
            return total

        print(
            f"[{self.ALGORITHM_NAME}] Training: {budget_steps:,} steps, "
            f"learn_step={learn_step}, agents={len(agent_ids)}, "
            f"num_envs={num_envs}"
        )

        on_policy_start = time.time()
        while steps_done < budget_steps:
            remaining_steps = budget_steps - steps_done
            collect_target = min(learn_step, remaining_steps)
            collect_n = max(1, (collect_target + step_size - 1) // step_size)

            # Per-step rollout buffers (lists of per-agent values).
            states:       Dict[str, List] = {a: [] for a in agent_ids}
            actions_buf:  Dict[str, List] = {a: [] for a in agent_ids}
            log_probs_buf: Dict[str, List] = {a: [] for a in agent_ids}
            rewards_buf:  Dict[str, List] = {a: [] for a in agent_ids}
            dones_buf:    Dict[str, List] = {a: [] for a in agent_ids}
            values_buf:   Dict[str, List] = {a: [] for a in agent_ids}

            for _ in range(collect_n):
                action_obs = {
                    a: _sanitize_obs(obs_a) for a, obs_a in obs.items()
                }
                action_result = agent.get_action(action_obs, infos=info)
                # AgileRL's IPPO/PPO contract is a 4-tuple
                # ``(action, log_prob, entropy, value)``.  Anything else is
                # an API breakage — silently substituting zeros for log_prob
                # turns the PPO importance ratio into ``exp(new_logp)`` which
                # is unbounded and destroys training.  Fail loud instead.
                if not (isinstance(action_result, tuple) and len(action_result) >= 4):
                    raise RuntimeError(
                        f"[{self.ALGORITHM_NAME}] expected agent.get_action to "
                        f"return (action, log_prob, entropy, value); got "
                        f"{type(action_result).__name__} of length "
                        f"{len(action_result) if isinstance(action_result, tuple) else 'N/A'}"
                    )
                action, log_prob, _, value = (
                    action_result[0], action_result[1],
                    action_result[2], action_result[3],
                )

                if is_vec:
                    # Vec env's ``step`` accepts a dict of
                    # shape-``(num_envs, ...)`` arrays per agent and
                    # returns batched results. Pass actions through
                    # unchanged.
                    next_obs, reward, termination, truncation, info = env.step(action)
                else:
                    # Strip leading (1, ...) batch dim for raw parallel envs.
                    env_actions = self._unwrap_actions_for_env(action)
                    next_obs, reward, termination, truncation, info = env.step(env_actions)

                # Per-sub-env reward / done bookkeeping.
                step_rew = _sum_rewards_per_env(reward)
                ep_reward += step_rew
                next_done = self._build_done_dict(
                    termination,
                    truncation,
                    agent_ids,
                    is_vec=is_vec,
                    num_envs=num_envs,
                )
                env_done = self._completed_env_mask(next_done, agent_ids, num_envs)

                # Emit completed-episode rewards. For the vec path
                # ``env_done[i]`` triggers the worker's auto-reset; for
                # the single-env path we manually reset below.
                for i, finished in enumerate(env_done):
                    if finished:
                        episode_rewards.append(float(ep_reward[i]))
                        ep_reward[i] = 0.0

                # AgileRL's ``learn()`` expects every per-step tensor to
                # carry an explicit vectorization axis of size
                # ``num_envs``. Vec envs already provide this; single
                # envs need ``[None, ...]`` to inject a singleton axis.
                for a, obs_a in obs.items():
                    states[a].append(_batch_obs(_sanitize_obs(obs_a)))

                    # Action / log_prob / value already carry the
                    # ``num_envs`` axis from AgileRL's get_action.
                    act_a = action.get(a)
                    actions_buf[a].append(
                        np.asarray(act_a) if act_a is not None
                        else np.array([np.nan], dtype=np.float32)
                    )
                    lp_a = log_prob.get(a)
                    log_probs_buf[a].append(
                        np.asarray(lp_a) if lp_a is not None
                        else np.array([np.nan], dtype=np.float32)
                    )
                    v_a = value.get(a)
                    values_buf[a].append(
                        np.asarray(v_a) if v_a is not None
                        else np.array([np.nan], dtype=np.float32)
                    )

                    if is_vec:
                        rewards_buf[a].append(
                            np.asarray(reward.get(a, np.full(num_envs, np.nan, dtype=np.float32)),
                                       dtype=np.float32).reshape(num_envs)
                        )
                    else:
                        rewards_buf[a].append(
                            np.array([float(reward.get(a, np.nan))], dtype=np.float32)
                        )
                    dones_buf[a].append(np.asarray(done[a]))

                obs = next_obs
                done = next_done
                steps_done += step_size

                # For the single-env path, manually auto-reset at episode
                # end. Vec env workers auto-reset internally; the single-env
                # path mirrors AgileRL's reference loop and explicitly resets
                # the environment plus the carried ``done`` flags.
                if not is_vec and bool(env_done[0]):
                    obs, info = env.reset()
                    done = {
                        a: np.zeros(num_envs, dtype=np.float32)
                        for a in agent_ids
                    }

            # Bootstrap obs for GAE: AgileRL's learn() expects the same
            # ``(num_envs, obs_dim)`` shape as the stored states.
            bootstrap_obs = {
                a: _batch_obs(_sanitize_obs(obs_a))
                for a, obs_a in obs.items()
            }

            # Learn from the collected rollout.
            experiences = (
                states, actions_buf, log_probs_buf,
                rewards_buf, dones_buf, values_buf,
                bootstrap_obs, done,
            )
            try:
                loss = agent.learn(experiences)
                mean_loss = self._mean_loss(loss)
                if mean_loss is not None:
                    losses.append(mean_loss)
            except Exception as exc:
                logger.warning(
                    "[%s] learn() failed: %s",
                    self.ALGORITHM_NAME, exc, exc_info=True,
                )

            progress_steps = min(steps_done, budget_steps)
            elapsed_op = time.time() - on_policy_start
            loss_str = f" | mean_loss={losses[-1]:.4f}" if losses else ""
            print(
                f"[{self.ALGORITHM_NAME}] {progress_steps:,}/{budget_steps:,} steps"
                f" | episodes={len(episode_rewards)}"
                f" | mean_rew={np.mean(episode_rewards[-10:]) if episode_rewards else 0.0:.2f}"
                + loss_str
            )

            # Fire the unified progress+recording hook on every learn-iteration
            # boundary, independent of whether a progress_callback is set
            # (recording must still happen on a headless run). _on_progress
            # no-ops the emit when the callback is None.
            _emit_mean = (
                float(np.mean(episode_rewards[-10:]))
                if episode_rewards else None
            )
            _emit_best = (
                float(max(episode_rewards)) if episode_rewards else None
            )
            _window = episode_rewards[-10:] if episode_rewards else []
            metrics_history.append({
                "iteration": len(metrics_history),
                "env_steps": int(progress_steps),
                "wall_time_s": elapsed_op,
                "mean": _emit_mean,
                "std": float(np.std(_window)) if _window else None,
                "min": float(min(_window)) if _window else None,
                "max": _emit_best,
                "mean_loss": float(losses[-1]) if losses else None,
            })
            self._on_progress(
                progress_callback,
                ProgressEvent(
                    unit="env_step",
                    current=int(progress_steps),
                    total=int(budget_steps),
                    env_steps=int(progress_steps),
                    best_fitness=_emit_best,
                    mean_fitness=_emit_mean,
                    wall_time_s=elapsed_op,
                ),
                champion=champion,
            )

        # Flush any in-progress sub-env episodes (e.g. budget exhausted
        # mid-episode) so callers see the partial reward rather than
        # silently dropping it. Only emit non-zero accumulators or if no
        # episode has been emitted yet (preserves prior single-env
        # behaviour).
        if not episode_rewards:
            episode_rewards.append(float(ep_reward.sum()))
        else:
            for i in range(num_envs):
                if abs(ep_reward[i]) > 0.0:
                    episode_rewards.append(float(ep_reward[i]))

        return {
            "episode_rewards": episode_rewards,
            "num_episodes": len(episode_rewards),
            "env_steps": min(steps_done, budget_steps),
            "mean_loss": float(np.mean(losses)) if losses else None,
            "metrics_history": metrics_history,
        }


# Concrete trainers

class MADDPGTrainer(AgileRLTrainer):
    """Multi-Agent DDPG via AgileRL (off-policy, decentralized actors,
    centralized critics)."""

    ALGORITHM_NAME = "maddpg"
    ALGORITHM_IMPORT = ("agilerl.algorithms.maddpg", "MADDPG")
    ON_POLICY = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            # AgileRL's DeterministicActor supports only Box/Discrete, so
            # MADDPG cannot train MultiDiscrete envs (gated by compatibility).
            "multidiscrete_action": False,
            "needs_global_state": False,
            "on_policy": False,
            # MADDPG's centralized-critic replay buffer assumes every
            # agent acts every step — turn-based AEC is incompatible.
            "supports_turn_based": False,
        }


class MATD3Trainer(AgileRLTrainer):
    """Multi-Agent Twin-Delayed DDPG via AgileRL (off-policy)."""

    ALGORITHM_NAME = "matd3"
    ALGORITHM_IMPORT = ("agilerl.algorithms.matd3", "MATD3")
    ON_POLICY = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            # Same DeterministicActor limitation as MADDPG: no MultiDiscrete.
            "multidiscrete_action": False,
            "needs_global_state": False,
            "on_policy": False,
            # Same centralized-critic simultaneity assumption as MADDPG.
            "supports_turn_based": False,
        }


class IPPOTrainer(AgileRLTrainer):
    """Independent PPO via AgileRL (on-policy, decentralized)."""

    ALGORITHM_NAME = "ippo"
    ALGORITHM_IMPORT = ("agilerl.algorithms.ippo", "IPPO")
    ON_POLICY = True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            # AgileRL's StochasticActor (PPO) natively supports MultiDiscrete.
            "multidiscrete_action": True,
            "needs_global_state": False,
            "on_policy": True,
            "supports_turn_based": True,
        }


class GroupedIPPOTrainer(IPPOTrainer):
    """IPPO with AgileRL's prefix-based agent grouping.

    AgileRL 2.6+ exposes a grouping API on :class:`agilerl.algorithms.IPPO`:
    agents whose IDs share a common prefix (split on the **final** ``_``
    before any trailing digit block) share one actor/critic network. Classic
    PettingZoo naming — ``adversary_0``, ``adversary_1``, ``agent_0`` — is
    already compatible; agents in a group **must** have identical
    observation and action spaces.

    This subclass validates the grouping **up-front** and fails loudly
    when the environment's agents cannot be sensibly grouped (mixed
    prefixes with no structure, or same-prefix agents with mismatched
    spaces). No silent fallback to per-agent networks.

    Notes:
        * Grouping reduces parameter count and typically speeds training
          when roles are symmetric (e.g. predator–prey pursuit).
        * Grouping is derived from the environment's ``possible_agents``
          naming, following AgileRL's default behaviour.
    """

    ALGORITHM_NAME = "ippo-grouped"
    ON_POLICY = True

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "discrete_action": True,
            "continuous_action": True,
            # AgileRL's StochasticActor (PPO) natively supports MultiDiscrete.
            "multidiscrete_action": True,
            "needs_global_state": False,
            "on_policy": True,
            "supports_turn_based": True,
        }

    @staticmethod
    def _group_prefix(agent_id: str) -> str:
        """Return the prefix before a trailing numeric suffix in ``agent_id``.

        Mirrors AgileRL's grouping convention for ids like ``agent_0`` and
        ``agent_1``. Agents without an underscore, or whose final underscore
        is not followed by only digits, are treated as singleton groups
        (prefix = full id) so non-indexed names like ``team_blue`` /
        ``team_red`` are not collapsed into a single ``team`` group.
        """
        if "_" not in agent_id:
            return agent_id
        prefix, suffix = agent_id.rsplit("_", 1)
        if suffix.isdigit():
            return prefix
        return agent_id

    @classmethod
    def _validate_grouping(
        cls,
        agent_ids: List[str],
        obs_spaces,
        act_spaces,
    ) -> Dict[str, List[str]]:
        """Group ``agent_ids`` by prefix; fail loudly on inconsistencies.

        Raises:
            ValueError: If the environment yields only singleton groups
                (grouping has no effect — user probably wants plain
                ``IPPOTrainer``), or if any group contains agents with
                mismatched observation or action spaces.

        Returns:
            ``{prefix: [agent_id, ...]}`` mapping, preserving insertion
            order of the original ``agent_ids`` list.
        """
        groups: Dict[str, List[str]] = {}
        for aid in agent_ids:
            groups.setdefault(cls._group_prefix(aid), []).append(aid)

        if all(len(members) == 1 for members in groups.values()):
            raise ValueError(
                f"GroupedIPPOTrainer: agent_ids={agent_ids!r} yield only "
                "singleton groups. Use IPPOTrainer instead, or rename the "
                "environment's agents so multiple share a prefix "
                "(e.g. 'adversary_0', 'adversary_1')."
            )

        # Spaces within a group must be identical.
        def _space_key(sp):
            return repr(sp)

        for prefix, members in groups.items():
            if len(members) < 2:
                continue
            ref_aid = members[0]
            ref_obs = obs_spaces[ref_aid]
            ref_act = act_spaces[ref_aid]
            for aid in members[1:]:
                if _space_key(obs_spaces[aid]) != _space_key(ref_obs):
                    raise ValueError(
                        "GroupedIPPOTrainer: agents in group "
                        f"{prefix!r} have mismatched observation spaces. "
                        f"{ref_aid}: {ref_obs!r}; {aid}: {obs_spaces[aid]!r}."
                    )
                if _space_key(act_spaces[aid]) != _space_key(ref_act):
                    raise ValueError(
                        "GroupedIPPOTrainer: agents in group "
                        f"{prefix!r} have mismatched action spaces. "
                        f"{ref_aid}: {ref_act!r}; {aid}: {act_spaces[aid]!r}."
                    )
        return groups

    def _build_agent(
        self,
        alg_cls,
        obs_spaces,
        act_spaces,
        agent_ids: List[str],
        ctor_kwargs: Dict[str, Any],
    ):
        """Validate grouping then delegate to AgileRL's IPPO constructor.

        AgileRL's IPPO detects grouping from ``agent_ids`` internally —
        no extra kwarg required — so all we do here is the pre-flight
        consistency check.
        """
        # Apply the grouped-IPPO advantage/state flattening fix exactly when
        # grouped IPPO is actually used. Raises loudly if AgileRL drifted.
        _apply_agilerl_grouping_patch()
        validation_obs_spaces = (
            dict(zip(agent_ids, obs_spaces))
            if isinstance(obs_spaces, (list, tuple))
            else obs_spaces
        )
        validation_act_spaces = (
            dict(zip(agent_ids, act_spaces))
            if isinstance(act_spaces, (list, tuple))
            else act_spaces
        )
        self._validate_grouping(
            agent_ids,
            validation_obs_spaces,
            validation_act_spaces,
        )
        return alg_cls(
            observation_spaces=obs_spaces,
            action_spaces=act_spaces,
            agent_ids=agent_ids,
            **ctor_kwargs,
        )
