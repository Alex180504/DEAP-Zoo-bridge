"""Unified champion-rollout recording for all trainers.

Historically every trainer re-implemented champion recording: building a
:class:`~deap_pettingzoo.utils.logging.replay.FrameRecorder`, constructing a
:class:`~deap_pettingzoo.baselines.unified_evaluator.UnifiedEvaluator`, deciding
"record now?" (in per-generation *or* per-env-step units, which are not
comparable across EA and RL), tracking episode offsets, writing sidecar
metadata, and swallowing errors. :class:`RecordingController` centralises all of
that behind one object so each trainer has a single record call site with
identical, algorithm-agnostic cadence semantics.

Cadence is expressed on the normalized progress axis shared by every trainer
(:attr:`~deap_pettingzoo.baselines.base.ProgressEvent.fraction` and
``.env_steps``), selected by :class:`~deap_pettingzoo.baselines.base.RecordingMode`:

* ``COUNT`` — ``max_recordings`` videos spread evenly over ``fraction`` (default).
* ``EVERY`` — one per generation/iteration, no cap.
* ``INTERVAL`` — one every ``every_n_steps`` env steps.
* ``FINAL`` — only the end-of-run rollout (via :meth:`finalize`).

Usage from a trainer::

    self._recorder = RecordingController(config, log, max_steps=ep_len, seed=seed)
    ...                                   # per iteration:
    self._on_progress(cb, event, champion=lambda: GenomePolicyWrapper(...))
    ...                                   # once at the end:
    self._recorder.finalize(lambda: champ_policy, iteration=g, env_steps=es)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, List, Mapping, Optional, Union

from deap_pettingzoo.baselines.base import ProgressEvent, RecordingMode
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator
from deap_pettingzoo.utils.logging import NoOpLogger

logger = logging.getLogger(__name__)

#: Default number of evenly-spread captures in ``COUNT`` mode when the user
#: leaves ``max_recordings`` unset.
DEFAULT_COUNT = 10

#: A champion provider returns the policy to roll out — either a single policy
#: broadcast to all agents, or a per-agent ``{agent_id: policy}`` map for
#: heterogeneous rollouts (co-evolution). ``None`` skips the recording.
ChampionProvider = Callable[[], Union[Any, Mapping[str, Any], None]]


class RecordingController:
    """Owns the recorder, evaluator, cadence schedule, and error handling."""

    def __init__(
        self,
        config: Any,
        experiment_logger: Any,
        *,
        max_steps: int,
        seed: Optional[int] = None,
        episode_prefix: str = "train_",
    ) -> None:
        cfg = config.recording
        self.mode: RecordingMode = cfg.mode
        self._env_name: str = config.env_name
        self._env_config = dict(config.env_config) if config.env_config else None
        self._seed = seed
        self.recorder = None
        self.evaluator: Optional[UnifiedEvaluator] = None
        self.enabled = False
        # Dedup key (iteration, env_steps) of the most recent capture so
        # finalize() doesn't double-record a champion observe() already caught.
        self._last_key: Optional[tuple] = None
        self._warned = False

        if not bool(cfg.enabled) or isinstance(experiment_logger, NoOpLogger):
            return
        try:
            recorder = experiment_logger.get_frame_recorder(
                fps=cfg.fps,
                fmt=cfg.fmt,
                max_frames_per_episode=cfg.max_frames_per_episode,
                max_dim=cfg.max_dim,
                episode_prefix=cfg.episode_prefix or episode_prefix,
            )
        except Exception:  # pragma: no cover - best-effort
            logger.warning(
                "RecordingController: failed to build frame recorder; "
                "recording disabled for this run.",
                exc_info=True,
            )
            return
        if recorder is None or not getattr(recorder, "enabled", False):
            return

        self.recorder = recorder
        self.evaluator = UnifiedEvaluator(max_steps=max_steps, frame_recorder=recorder)
        self.enabled = True

        # Resolve the cadence schedule.
        self._fraction_points: List[float] = []  # COUNT: ascending in [0, 1]
        self._interval: Optional[int] = None  # INTERVAL: fixed env-step stride
        self._next_step: Optional[int] = None  # INTERVAL: next env-step threshold
        if self.mode is RecordingMode.COUNT:
            n = int(cfg.max_recordings) if cfg.max_recordings else DEFAULT_COUNT
            if n <= 1:
                self._fraction_points = [1.0]
            else:
                self._fraction_points = [i / (n - 1) for i in range(n)]
        elif self.mode is RecordingMode.INTERVAL:
            self._interval = max(1, int(cfg.every_n_steps))
            self._next_step = self._interval

    # -- per-iteration hook -------------------------------------------------

    def observe(
        self,
        event: ProgressEvent,
        champion: ChampionProvider,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record a champion rollout if ``event`` crosses the next checkpoint.

        ``champion`` is evaluated lazily — only when a recording is actually
        due — so trainers pay the rollout cost zero times on non-recording
        iterations.
        """
        if not self.enabled or not self._is_due(event):
            return
        self._record(champion, iteration=event.current, env_steps=event.env_steps,
                     extra=extra)

    def finalize(
        self,
        champion: ChampionProvider,
        *,
        iteration: int,
        env_steps: int,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Always emit the end-of-run rollout (unless already captured here).

        Covers trainers whose loop can't be hooked mid-run (QDax, the
        DEAP-internal EA strategies) and guarantees at least one artefact even
        on tight budgets where no mid-run checkpoint fired.
        """
        if not self.enabled:
            return
        if self._last_key == (int(iteration), int(env_steps)):
            return
        self._record(champion, iteration=iteration, env_steps=env_steps, extra=extra)

    # -- internals ----------------------------------------------------------

    def _is_due(self, event: ProgressEvent) -> bool:
        """Decide — per :attr:`mode` — whether ``event`` warrants a recording."""
        if self.mode is RecordingMode.EVERY:
            return True
        if self.mode is RecordingMode.FINAL:
            return False
        if self.mode is RecordingMode.COUNT:
            crossed = False
            while self._fraction_points and event.fraction >= self._fraction_points[0]:
                self._fraction_points.pop(0)
                crossed = True
            return crossed
        if self.mode is RecordingMode.INTERVAL:
            if self._next_step is None or event.env_steps < self._next_step:
                return False
            # Advance the threshold past the current step count by the fixed
            # interval (record once per crossing even if many steps elapsed).
            while self._next_step <= event.env_steps:
                self._next_step += self._interval
            return True
        return False

    def _record(
        self,
        champion: ChampionProvider,
        *,
        iteration: int,
        env_steps: int,
        extra: Optional[Mapping[str, Any]],
    ) -> None:
        """Run one champion rollout through the evaluator (best-effort)."""
        try:
            pol = champion()
        except Exception as exc:  # pragma: no cover - best-effort
            self._note_failure(exc, "building the champion policy")
            return
        if pol is None:
            return
        single = None
        policy_map = None
        if isinstance(pol, Mapping):
            policy_map = dict(pol)
        else:
            single = pol
        try:
            self.evaluator.evaluate(
                policy=single,
                env_name=self._env_name,
                num_episodes=1,
                seed=self._seed,
                env_config=self._env_config,
                iteration=int(iteration),
                env_steps=int(env_steps),
                record_extra=extra,
                policy_map=policy_map,
            )
            self._last_key = (int(iteration), int(env_steps))
        except Exception as exc:  # pragma: no cover - best-effort
            self._note_failure(exc, "rendering the champion rollout")

    def _note_failure(self, exc: Exception, while_doing: str) -> None:
        """Report a best-effort recording failure once, without spamming.

        The full traceback is always logged at DEBUG (visible via
        ``DEAP_LOG_LEVEL=debug``); a single headline ``RuntimeWarning`` is
        emitted the first time so it surfaces in the service log (the backend
        routes ``warnings.warn`` to the ``py.warnings`` logger at WARNING via
        ``logging.captureWarnings``).
        """
        logger.debug("RecordingController: failed while %s", while_doing, exc_info=True)
        if not self._warned:
            warnings.warn(
                f"Frame recording failed while {while_doing} "
                f"({type(exc).__name__}: {exc}); continuing without further "
                "recordings.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned = True

    def close(self) -> None:
        if self.evaluator is not None:
            try:
                self.evaluator.close()
            except Exception:  # pragma: no cover
                pass
