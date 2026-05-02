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
import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
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

    _default_recording_every_n: ClassVar[int] = 10

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

    def __init__(self, device: str = "cpu", verbose: bool = False) -> None:
        self.device = device
        self.verbose = verbose

    # BaselineTrainer interface

    def train(self, config: TrainingConfig) -> TrainingResult:
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

        # ``num_envs`` controls vectorized rollout collection for
        # on-policy trainers. Off-policy trainers ignore it and always
        # use a single env. Pop it from ``params`` before splitting into
        # ctor/loop kwargs because it is a trainer-level concern, not
        # an algorithm constructor argument.
        num_envs = int(params.pop("num_envs", 1))
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs!r}")
        if num_envs > 1 and not self.ON_POLICY:
            # Off-policy replay buffers in AgileRL are built around a
            # single env; transparently coerce to 1 with a warning.
            if self.verbose:
                print(
                    f"[{self.ALGORITHM_NAME}] num_envs={num_envs} ignored: "
                    "off-policy trainers only support num_envs=1."
                )
            num_envs = 1

        if self.ON_POLICY and num_envs > 1:
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
        from deap_pettingzoo.utils.logging.experiment_logger import NoOpLogger

        recording_cfg = config.recording
        with build_logger(config, algorithm=self.ALGORITHM_NAME) as log:
            recorder = None
            record_every_n_steps: Optional[int] = None
            record_state = {"episode_idx": 0}

            # Frame-recording setup. Recording is best-effort: errors are
            # caught (and a traceback is printed) so they cannot break
            # training.
            if recording_cfg.enabled and not isinstance(log, NoOpLogger):
                try:
                    recorder = log.get_frame_recorder(
                        fps=recording_cfg.fps,
                        fmt=recording_cfg.fmt,
                        max_frames_per_episode=recording_cfg.max_frames_per_episode,
                        max_dim=recording_cfg.max_dim,
                        episode_prefix=recording_cfg.episode_prefix or "train_",
                    )
                except Exception:  # pragma: no cover - best-effort
                    import traceback as _tb
                    _tb.print_exc()
                    recorder = None

            if recorder is not None and recorder.enabled:
                # Resolve the mid-training cadence. ``every_n_steps`` (env
                # steps) wins outright; otherwise ``every_n_generations`` is
                # interpreted as "every N learn iterations" and converted to
                # env steps via ``learn_step``. ``None`` on both → no
                # mid-training recording (final rollout still emitted).
                if recording_cfg.every_n_steps is not None:
                    record_every_n_steps = int(recording_cfg.every_n_steps)
                elif recording_cfg.every_n_generations is not None:
                    learn_step = int(
                        params.get(
                            "learn_step", self.DEFAULT_PARAMS.get("learn_step", 1)
                        )
                    )
                    record_every_n_steps = max(
                        1, int(recording_cfg.every_n_generations) * learn_step
                    )

            from deap_pettingzoo.baselines.unified_evaluator import (
                UnifiedEvaluator,
            )

            def _record_episode() -> None:
                """Run one champion-rollout episode through ``recorder``.

                Reuses the already-built ``recorder`` (and therefore the
                same ``run_dir``). The episode index increments per
                successful call so artefacts are named
                ``{prefix}episode_NNNN.{mp4,gif}`` with monotonically
                increasing N.
                """
                if recorder is None or not recorder.enabled:
                    return
                ep_idx = record_state["episode_idx"]
                record_state["episode_idx"] = ep_idx + 1
                try:
                    rec_eval = UnifiedEvaluator(
                        max_steps=int(params.get("max_steps_per_episode", 100)),
                        frame_recorder=recorder,
                        record_episode_offset=ep_idx,
                    )
                    try:
                        rec_eval.evaluate(
                            policy=policy,
                            env_name=config.env_name,
                            num_episodes=1,
                            seed=seed + ep_idx,
                            env_config=dict(config.env_config)
                            if config.env_config
                            else None,
                        )
                    finally:
                        rec_eval.close()
                except Exception:  # pragma: no cover - best-effort
                    import traceback as _tb
                    _tb.print_exc()

            start = time.time()
            if self.ON_POLICY:
                metrics = self._run_on_policy(
                    env, agent, agent_ids, budget_steps, params, seed,
                    record_callback=_record_episode if recorder is not None else None,
                    record_every_n_steps=record_every_n_steps,
                )
            else:
                metrics = self._run_off_policy(
                    env, agent, agent_ids, budget_steps, params, seed,
                    record_callback=_record_episode if recorder is not None else None,
                    record_every_n_steps=record_every_n_steps,
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

            # Always emit one end-of-training rollout when recording is
            # enabled, regardless of mid-training cadence — guarantees at
            # least one mp4 even on tight budgets where the cadence never
            # fires.
            if recorder is not None and recorder.enabled:
                _record_episode()

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
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:  # pragma: no cover
            pass

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
        record_callback: Optional[Any] = None,
        record_every_n_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Hand-rolled off-policy loop using AgileRL's replay buffer.

        Note:
            AgileRL's ``get_action`` returns a ``(actions, raw_actions)``
            tuple.  The two dicts serve different purposes and must each
            be squeezed independently:

            * ``actions`` — sent to ``env.step``.  Discrete spaces expect
              Python ints; Box spaces expect 1-D arrays.
            * ``raw_actions`` — stored in the replay buffer.  Always the
              continuous representation the critic was built for.

            Storing ``raw_actions`` with its leading vec-dim intact
            (shape ``(1, act_dim)``) next to unbatched obs
            (shape ``(features,)``) caused a 2-vs-3 ndim mismatch inside
            ``MADDPG.learn()``.  Using ``actions`` for the buffer instead
            would silently break discrete envs where ``actions`` are scalar
            ints but the critic expects the continuous representation.
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
        progress_interval = max(1, budget_steps // 20) if self.verbose else 0
        loop_start = time.time()

        obs, info = env.reset(seed=seed)
        episode_rewards: List[float] = []
        ep_reward = 0.0
        ep_len = 0
        losses: List[float] = []

        for step in range(budget_steps):
            action_result = agent.get_action(obs, infos=info)
            actions, raw_actions = (
                action_result if isinstance(action_result, tuple) else (action_result, action_result)
            )
            env_actions = self._unwrap_actions_for_env(actions)
            buffer_actions: Dict[str, Any] = {}
            for aid, act in raw_actions.items():
                arr = np.asarray(act)
                if arr.ndim > 0 and arr.shape[0] == 1:
                    arr = arr[0]
                buffer_actions[aid] = arr
            next_obs, reward, termination, truncation, info = env.step(env_actions)

            memory.save_to_memory(
                obs, buffer_actions, reward, next_obs, termination
            )

            ep_reward += float(sum(reward.values()))
            ep_len += 1

            if (
                step >= learning_delay
                and len(memory) >= batch_size
                and (step - learning_delay) % learn_step == 0
            ):
                experiences = memory.sample(batch_size)
                try:
                    loss = agent.learn(experiences)
                    mean_loss = self._mean_loss(loss)
                    if mean_loss is not None:
                        losses.append(mean_loss)
                except Exception as exc:  # pragma: no cover - version drift
                    if self.verbose:
                        print(f"[{self.ALGORITHM_NAME}] learn() failed: {exc}")

            done = any(termination.values()) or any(truncation.values())
            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_len = 0
                obs, info = env.reset()
            else:
                obs = next_obs

            if progress_interval and (step + 1) % progress_interval == 0:
                elapsed = time.time() - loop_start
                pct = 100.0 * (step + 1) / budget_steps
                last_r = episode_rewards[-1] if episode_rewards else float("nan")
                print(
                    f"[{self.ALGORITHM_NAME}] step {step + 1}/{budget_steps} "
                    f"({pct:5.1f}%) | episodes={len(episode_rewards)} | "
                    f"last_reward={last_r:.2f} | learn_calls={len(losses)} | "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

            # Mid-training champion-rollout recording. Triggered every
            # ``record_every_n_steps`` env steps when both a callback and a
            # cadence are set; the callback is responsible for catching
            # its own errors (see ``_record_episode`` in ``train()``).
            if (
                record_callback is not None
                and record_every_n_steps
                and (step + 1) % record_every_n_steps == 0
            ):
                record_callback()

        if ep_len > 0:
            episode_rewards.append(ep_reward)

        return {
            "episode_rewards": episode_rewards,
            "num_episodes": len(episode_rewards),
            "env_steps": budget_steps,
            "mean_loss": float(np.mean(losses)) if losses else None,
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
        record_callback: Optional[Any] = None,
        record_every_n_steps: Optional[int] = None,
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

        # Per-(sub-)env running episode-reward accumulator. For the
        # single-env case this is a scalar in a length-1 array; for the
        # vec case it tracks each sub-env independently so we can emit
        # an episode reward each time a sub-env terminates.
        ep_reward = np.zeros(num_envs, dtype=np.float64)

        obs, info = env.reset(seed=seed)
        # Per-agent done flags: shape ``(num_envs,)`` — zero at episode 0.
        done: Dict[str, Any] = {a: np.zeros(num_envs, dtype=np.int8) for a in agent_ids}

        # Helper: ensure obs has a leading num_envs axis. Vec envs
        # already provide it; single envs return ``(obs_dim,)``.
        def _batch_obs(o: Any) -> np.ndarray:
            arr = np.asarray(o)
            return arr if is_vec else arr[None, ...]

        # Helper: per-sub-env reward summed across agents, shape (num_envs,).
        def _sum_rewards_per_env(reward_dict: Dict[str, Any]) -> np.ndarray:
            total = np.zeros(num_envs, dtype=np.float64)
            for a in agent_ids:
                if a not in reward_dict:
                    continue
                r = reward_dict[a]
                if is_vec:
                    total += np.asarray(r, dtype=np.float64).reshape(num_envs)
                else:
                    total += float(r)
            return total

        # Helper: per-sub-env "any agent done" bool, shape (num_envs,).
        def _env_done_mask(term: Dict[str, Any], trunc: Dict[str, Any]) -> np.ndarray:
            mask = np.zeros(num_envs, dtype=bool)
            for a in agent_ids:
                t = term.get(a, False)
                u = trunc.get(a, False)
                if is_vec:
                    mask |= np.asarray(t, dtype=bool).reshape(num_envs)
                    mask |= np.asarray(u, dtype=bool).reshape(num_envs)
                else:
                    mask |= bool(t) or bool(u)
            return mask

        print(
            f"[{self.ALGORITHM_NAME}] Training: {budget_steps:,} steps, "
            f"learn_step={learn_step}, agents={len(agent_ids)}, "
            f"num_envs={num_envs}"
        )

        while steps_done < budget_steps:
            collect_n = min(learn_step, budget_steps - steps_done)

            # Per-step rollout buffers (lists of per-agent values).
            states:       Dict[str, List] = {a: [] for a in agent_ids}
            actions_buf:  Dict[str, List] = {a: [] for a in agent_ids}
            log_probs_buf: Dict[str, List] = {a: [] for a in agent_ids}
            rewards_buf:  Dict[str, List] = {a: [] for a in agent_ids}
            dones_buf:    Dict[str, List] = {a: [] for a in agent_ids}
            values_buf:   Dict[str, List] = {a: [] for a in agent_ids}

            for _ in range(collect_n):
                action_result = agent.get_action(obs, infos=info)
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
                env_done = _env_done_mask(termination, truncation)

                # Emit completed-episode rewards. For the vec path
                # ``env_done[i]`` triggers the worker's auto-reset; for
                # the single-env path we manually reset below.
                for i, finished in enumerate(env_done):
                    if finished:
                        episode_rewards.append(float(ep_reward[i]))
                        ep_reward[i] = 0.0

                # next_done per agent, shape (num_envs,) int8.
                next_done: Dict[str, Any] = {}
                for a in agent_ids:
                    t = termination.get(a, False)
                    u = truncation.get(a, False)
                    if is_vec:
                        nd = (np.asarray(t, dtype=bool).reshape(num_envs)
                              | np.asarray(u, dtype=bool).reshape(num_envs))
                    else:
                        nd = np.array([int(bool(t) or bool(u))], dtype=np.int8)
                    next_done[a] = nd.astype(np.int8)

                # AgileRL's ``learn()`` expects every per-step tensor to
                # carry an explicit vectorization axis of size
                # ``num_envs``. Vec envs already provide this; single
                # envs need ``[None, ...]`` to inject a singleton axis.
                obs_fallback = (
                    np.zeros_like(next(iter(obs.values()))) if obs else None
                )
                for a in agent_ids:
                    obs_a = obs.get(a, obs_fallback)
                    states[a].append(_batch_obs(obs_a))

                    # Action / log_prob / value already carry the
                    # ``num_envs`` axis from AgileRL's get_action.
                    act_a = action.get(a)
                    actions_buf[a].append(
                        np.asarray(act_a) if act_a is not None
                        else np.zeros((num_envs,))
                    )
                    lp_a = log_prob.get(a)
                    log_probs_buf[a].append(
                        np.asarray(lp_a) if lp_a is not None
                        else np.zeros((num_envs,))
                    )
                    v_a = value.get(a)
                    values_buf[a].append(
                        np.asarray(v_a) if v_a is not None
                        else np.zeros((num_envs,))
                    )

                    if is_vec:
                        rewards_buf[a].append(
                            np.asarray(reward.get(a, np.zeros(num_envs)),
                                       dtype=np.float32).reshape(num_envs)
                        )
                    else:
                        rewards_buf[a].append(
                            np.array([float(reward.get(a, 0.0))], dtype=np.float32)
                        )
                    dones_buf[a].append(np.asarray(done[a]))

                obs = next_obs
                done = next_done
                steps_done += 1

                # For the single-env path, manually auto-reset at episode
                # end. The just-stored ``dones_buf[T]`` is the *previous*
                # step's done; the next iteration stores ``done =
                # next_done`` (terminal flag = 1) marking the new-episode
                # start. Zeroing here would lose the boundary and let GAE
                # bootstrap rewards across episode resets.
                #
                # The vec-env path skips this block: AsyncPettingZooVecEnv
                # workers reset terminated sub-envs internally and
                # ``next_obs`` already reflects the new episode's first
                # state.
                if not is_vec and bool(env_done[0]):
                    obs, info = env.reset()

            # Bootstrap obs for GAE: AgileRL's learn() expects the same
            # ``(num_envs, obs_dim)`` shape as the stored states.
            bootstrap_obs = {}
            for a in agent_ids:
                obs_a = obs.get(
                    a, np.zeros_like(next(iter(obs.values())))
                ) if obs else None
                bootstrap_obs[a] = _batch_obs(obs_a)

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
                print(f"[{self.ALGORITHM_NAME}] learn() failed: {exc}")

            loss_str = f" | mean_loss={losses[-1]:.4f}" if losses else ""
            print(
                f"[{self.ALGORITHM_NAME}] {steps_done:,}/{budget_steps:,} steps"
                f" | episodes={len(episode_rewards)}"
                f" | mean_rew={np.mean(episode_rewards[-10:]) if episode_rewards else 0.0:.2f}"
                + loss_str
            )

            # Mid-training champion-rollout recording. ``steps_done`` is
            # checked at learn-iteration boundaries (rollout chunks of
            # ``learn_step``) so for on-policy trainers the cadence is
            # quantised to the rollout size: a 256-step learn_step with
            # ``record_every_n_steps=200`` records once per learn iteration
            # rather than mid-rollout. This keeps the recording episode
            # off-thread from the on-policy data collection.
            if (
                record_callback is not None
                and record_every_n_steps
                and steps_done % record_every_n_steps < learn_step
                and (steps_done // record_every_n_steps)
                > ((steps_done - learn_step) // record_every_n_steps)
            ):
                record_callback()

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
            "env_steps": steps_done,
            "mean_loss": float(np.mean(losses)) if losses else None,
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
