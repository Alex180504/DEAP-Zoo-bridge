"""Unified periodic champion probing for all trainers.

:class:`ProbeController` centralises the off-budget held-out evaluation
configured by :class:`~deap_pettingzoo.baselines.base.ProbeConfig` behind one
object, mirroring :class:`~deap_pettingzoo.baselines.recording.RecordingController`
so each trainer has a single probe call site with identical cadence semantics.

Every probe rolls the frozen champion for ``n_episodes`` episodes on a
dedicated env instance (never counted against the training budget) and
appends one row to the run's ``probe.csv``:

``iteration, env_steps, probe_reward_mean, probe_reward_std,
probe_episodes[, probe_<info_key>_mean, probe_<info_key>_std, ...]``

The row schema is fixed per run (it depends only on ``info_keys``), which the
strict CSV sink requires. Episode seeds are common across probes of one run
(common random numbers), so successive checkpoints of a learning curve differ
by policy, not by episode draw.

Usage from a trainer::

    self._prober = ProbeController(config, log, max_steps=ep_len, seed=seed)
    ...                                   # per iteration (via _on_progress):
    self._on_progress(cb, event, champion=lambda: GenomePolicyWrapper(...))
    ...                                   # once at the end:
    self._prober.finalize(lambda: champ_policy, iteration=g, env_steps=es)
    self._prober.close()

Trainers whose logger only exists after training (QDax) construct the
controller with ``experiment_logger=None`` (rows buffer in memory), and call
:meth:`flush_to` once the logger is available.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np

from deap_pettingzoo.baselines.base import ProgressEvent
from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator
from deap_pettingzoo.utils.logging import NoOpLogger

logger = logging.getLogger(__name__)

#: A champion provider returns the policy to probe: either a single policy
#: broadcast to all agents, or a per-agent ``{agent_id: policy}`` map.
#: ``None`` skips the probe.
ChampionProvider = Callable[[], Union[Any, Mapping[str, Any], None]]


class ProbeController:
    """Owns the probe evaluator, cadence schedule, buffering, and error handling."""

    def __init__(
        self,
        config: Any,
        experiment_logger: Any = None,
        *,
        max_steps: int,
        seed: Optional[int] = None,
        table: str = "probe",
    ) -> None:
        cfg = config.probe
        self.enabled = False
        self._log: Any = None
        self._buffer: List[Dict[str, Any]] = []
        self._table = table
        self._last_key: Optional[tuple] = None
        self._warned = False

        if not bool(cfg.enabled):
            return
        if isinstance(experiment_logger, NoOpLogger):
            # No sink will ever exist for this run; probing would be wasted work.
            return

        self._cfg = cfg
        self._log = experiment_logger  # None = buffer until flush_to()
        self._env_name: str = cfg.env_name or config.env_name
        if cfg.env_config is not None:
            self._env_config: Optional[Dict[str, Any]] = dict(cfg.env_config)
        else:
            self._env_config = dict(config.env_config) if config.env_config else None
        self._seed = cfg.seed if cfg.seed is not None else seed
        self.evaluator = UnifiedEvaluator(
            max_steps=max_steps, info_keys=cfg.info_keys
        )
        # First probe fires on the first progress event (baseline point);
        # subsequent probes every `every_n_steps` env steps.
        self._interval = max(1, int(cfg.every_n_steps))
        self._next_step = 0
        self.enabled = True

    # Per-iteration hook
    def observe(
        self,
        event: ProgressEvent,
        champion: ChampionProvider,
        *,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Probe the champion if ``event`` crosses the next step threshold.

        ``champion`` is evaluated lazily (only when a probe is actually due),
        so trainers pay nothing on non-probe iterations. ``extra`` is accepted
        for call-site symmetry with the recording controller but is not
        written into probe rows (their schema must stay fixed per run).
        """
        del extra
        if not self.enabled or not self._is_due(event):
            return
        self._probe(champion, iteration=event.current, env_steps=event.env_steps)

    def finalize(
        self,
        champion: ChampionProvider,
        *,
        iteration: int,
        env_steps: int,
    ) -> None:
        """Guarantee an end-of-run probe (unless one already landed here)."""
        if not self.enabled:
            return
        if self._last_key == (int(iteration), int(env_steps)):
            return
        self._probe(champion, iteration=iteration, env_steps=env_steps)

    def flush_to(self, experiment_logger: Any) -> None:
        """Attach a (late-built) logger and write any buffered rows to it."""
        if isinstance(experiment_logger, NoOpLogger) or experiment_logger is None:
            self._buffer.clear()
            return
        self._log = experiment_logger
        for row in self._buffer:
            self._write_row(row)
        self._buffer.clear()

    # Business logic

    def _is_due(self, event: ProgressEvent) -> bool:
        if event.env_steps < self._next_step:
            return False
        # Advance the threshold past the current step count (probe once per
        # crossing even if many steps elapsed since the previous event).
        while self._next_step <= event.env_steps:
            self._next_step += self._interval
        return True

    def _probe(
        self,
        champion: ChampionProvider,
        *,
        iteration: int,
        env_steps: int,
    ) -> None:
        """Run one held-out evaluation of the champion (best-effort)."""
        try:
            pol = champion()
        except Exception as exc:  # noqa: BLE001 - probe must never crash training
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
            result = self.evaluator.evaluate(
                policy=single,
                env_name=self._env_name,
                num_episodes=int(self._cfg.n_episodes),
                seed=self._seed,
                env_config=self._env_config,
                policy_map=policy_map,
            )
        except Exception as exc:  # noqa: BLE001 - probe must never crash training
            self._note_failure(exc, "rolling probe episodes")
            return

        row: Dict[str, Any] = {
            "iteration": int(iteration),
            "env_steps": int(env_steps),
            "probe_reward_mean": float(result.mean_reward),
            "probe_reward_std": float(result.std_reward),
            "probe_episodes": int(result.num_episodes),
        }
        for key in self._cfg.info_keys:
            values = result.info_metrics.get(key, [])
            row[f"probe_{key}_mean"] = float(np.mean(values)) if values else float("nan")
            row[f"probe_{key}_std"] = float(np.std(values)) if values else float("nan")
        self._last_key = (int(iteration), int(env_steps))
        if self._log is None:
            self._buffer.append(row)
        else:
            self._write_row(row)

    def _write_row(self, row: Dict[str, Any]) -> None:
        try:
            self._log.row(self._table, row)
        except Exception as exc:  # noqa: BLE001 - best-effort persistence
            self._note_failure(exc, "writing the probe row")

    def _note_failure(self, exc: Exception, while_doing: str) -> None:
        """Report a best-effort probe failure once, without spamming."""
        logger.debug("ProbeController: failed while %s", while_doing, exc_info=True)
        if not self._warned:
            warnings.warn(
                f"Champion probing failed while {while_doing} "
                f"({type(exc).__name__}: {exc}); continuing without further "
                "probes.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned = True

    def close(self) -> None:
        if self.enabled and self.evaluator is not None:
            try:
                self.evaluator.close()
            except Exception:  # pragma: no cover
                pass
