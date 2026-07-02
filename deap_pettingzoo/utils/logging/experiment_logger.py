"""High-level :class:`ExperimentLogger` wiring sinks and the manifest.

Trainers should construct a logger via :meth:`ExperimentLogger.from_config`
or :meth:`ExperimentLogger.create`. When ``output_dir is None`` a
:class:`NoOpLogger` is returned, so calling code can use the same
``with logger:`` pattern unconditionally.
"""

from __future__ import annotations

import time
import uuid
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping, Optional, Sequence, Type

from deap_pettingzoo.utils.logging.manifest import (
    RunManifest,
    capture_environment_metadata,
)
from deap_pettingzoo.utils.logging.replay import FrameRecorder
from deap_pettingzoo.utils.logging.sinks import (
    CsvScalarSink,
    JsonManifestSink,
    MultiSink,
    Sink,
    TensorBoardSink,
)


def _safe_token(text: str) -> str:
    """Sanitise a string into a path-safe token."""
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in text)


def _default_run_id(algorithm: str, env_name: str, seed: Optional[int]) -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    short = uuid.uuid4().hex[:6]
    seed_token = "noseed" if seed is None else f"s{seed}"
    return _safe_token(f"{algorithm}_{env_name}_{seed_token}_{ts}_{short}")


class ExperimentLogger(AbstractContextManager["ExperimentLogger"]):
    """Top-level logger threaded through trainers.

    Construct via :meth:`from_config` or :meth:`create`. Use as a
    context manager — :meth:`__exit__` finalises the manifest and
    closes every sink.

    All ``log_*`` methods are no-ops when the underlying sink stack is
    empty, so the same calling code works whether or not ``output_dir``
    is set.
    """

    def __init__(
        self,
        run_dir: Path,
        manifest: RunManifest,
        sink: Sink,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.manifest = manifest
        self._sink = sink
        self._t0: Optional[float] = None
        self._closed = False
        self._frame_recorder: Optional[FrameRecorder] = None

    # -- factory -----------------------------------------------------

    @classmethod
    def create(
        cls,
        output_dir: Optional[Path],
        *,
        algorithm: str,
        env_name: str,
        seed: Optional[int] = None,
        env_config: Optional[Mapping[str, Any]] = None,
        budget_mode: Optional[str] = None,
        budget_value: Optional[int] = None,
        hyperparams: Optional[Mapping[str, Any]] = None,
        notes: str = "",
        run_id: Optional[str] = None,
        enable_tensorboard: bool = False,
        enable_parquet: bool = False,
        extra_sinks: Sequence[Sink] = (),
    ) -> "ExperimentLogger":
        """Build an :class:`ExperimentLogger` (or :class:`NoOpLogger`).

        When ``output_dir is None`` returns a :class:`NoOpLogger` whose
        methods are all no-ops; this keeps the no-output code path
        identical to the with-output path on the trainer side.

        Args:
            output_dir: Parent directory for the run. The actual run
                directory is ``output_dir / run_id``. If ``None``, a
                :class:`NoOpLogger` is returned.
            algorithm: Algorithm short name (``"ea"``, ``"maddpg"``, ...).
            env_name: Registered PettingZoo env name.
            seed: Master seed (recorded in the manifest).
            env_config: Env config dict.
            budget_mode: ``"env_steps" | "eval_episodes" | "generations"``.
            budget_value: Budget amount in matching units.
            hyperparams: Trainer-specific hyperparameters.
            notes: Free-form notes.
            run_id: Optional explicit run id; auto-generated otherwise.
            enable_tensorboard: Add :class:`TensorBoardSink` (lazy import).
            enable_parquet: Add :class:`ParquetTrajectorySink`.
            extra_sinks: User-supplied sinks appended to the stack.
        """
        if output_dir is None:
            return NoOpLogger.instance()

        rid = run_id or _default_run_id(algorithm, env_name, seed)
        run_dir = Path(output_dir) / rid
        run_dir.mkdir(parents=True, exist_ok=True)

        sinks: list[Sink] = [JsonManifestSink(), CsvScalarSink()]
        if enable_parquet:
            from deap_pettingzoo.utils.logging.sinks import ParquetTrajectorySink

            sinks.append(ParquetTrajectorySink())
        if enable_tensorboard:
            sinks.append(TensorBoardSink())
        sinks.extend(extra_sinks)
        sink = MultiSink(sinks)
        sink.open(run_dir)

        manifest = RunManifest(
            run_id=rid,
            algorithm=algorithm,
            env_name=env_name,
            env_config=dict(env_config or {}),
            seed=seed,
            budget_mode=budget_mode,
            budget_value=budget_value,
            hyperparams=dict(hyperparams or {}),
            notes=notes,
            started_at=RunManifest.utc_now_iso(),
            environment=capture_environment_metadata(),
        )
        manifest.write(run_dir / "manifest.json")
        return cls(run_dir=run_dir, manifest=manifest, sink=sink)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        algorithm: str,
        run_id: Optional[str] = None,
        enable_tensorboard: Optional[bool] = None,
        enable_parquet: Optional[bool] = None,
        notes: str = "",
        extra_sinks: Sequence[Sink] = (),
    ) -> "ExperimentLogger":
        """Build a logger from a :class:`TrainingConfig`-shaped object.

        Reads ``output_dir``, ``env_name``, ``env_config``,
        ``budget_mode``, ``budget_value``, ``algorithm_params`` from
        ``config``. ``algorithm_params`` may contain
        ``logging_tensorboard`` / ``logging_parquet`` boolean overrides
        which are honored when the explicit kwargs are ``None``.
        """
        algorithm_params = getattr(config, "algorithm_params", {}) or {}
        # Fall back to the legacy
        # ``algorithm_params['seed']`` escape hatch for config-shaped objects
        # that don't declare the attribute (e.g. NEAT / QDax raw dicts).
        seed = getattr(config, "seed", None)
        if seed is None:
            seed = algorithm_params.get("seed")
        if enable_tensorboard is None:
            enable_tensorboard = bool(algorithm_params.get("logging_tensorboard", False))
        if enable_parquet is None:
            enable_parquet = bool(algorithm_params.get("logging_parquet", False))
        budget_mode = getattr(config, "budget_mode", None)
        if hasattr(budget_mode, "value"):
            budget_mode_str: Optional[str] = budget_mode.value
        elif budget_mode is not None:
            budget_mode_str = str(budget_mode)
        else:
            budget_mode_str = None
        return cls.create(
            output_dir=getattr(config, "output_dir", None),
            algorithm=algorithm,
            env_name=getattr(config, "env_name", "unknown_env"),
            seed=seed,
            env_config=getattr(config, "env_config", None),
            budget_mode=budget_mode_str,
            budget_value=getattr(config, "budget_value", None),
            hyperparams=algorithm_params,
            notes=notes,
            run_id=run_id,
            enable_tensorboard=enable_tensorboard,
            enable_parquet=enable_parquet,
            extra_sinks=extra_sinks,
        )

    # -- context manager --------------------------------------------

    def __enter__(self) -> "ExperimentLogger":
        self._t0 = time.time()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    # -- core API ---------------------------------------------------

    def row(self, table: str, row: Mapping[str, Any]) -> None:
        """Append ``row`` to the named table (CSV / TB / Parquet sinks)."""
        if self._closed:
            return
        self._sink.log_row(table, row)

    def scalar(self, name: str, value: float, step: int) -> None:
        if self._closed:
            return
        self._sink.log_scalar(name, value, step)

    def artifact(self, name: str, path: Path) -> None:
        if self._closed:
            return
        self._sink.log_artifact(name, path)
        self.manifest.artifacts[name] = str(Path(path).relative_to(self.run_dir)) \
            if Path(path).is_absolute() and Path(path).is_relative_to(self.run_dir) \
            else str(path)

    def summary(self, summary: Mapping[str, Any]) -> None:
        """Persist the final summary block to ``summary.json``."""
        self._sink.log_row("summary", summary)
        self.manifest.final_summary = dict(summary)

    # -- frame recorder hook ----------------------------------------

    def get_frame_recorder(
        self,
        *,
        fps: int = 20,
        fmt: str = "mp4",
        max_frames_per_episode: Optional[int] = None,
        max_dim: Optional[int] = None,
        episode_prefix: str = "",
    ) -> FrameRecorder:
        """Lazily build and cache a :class:`FrameRecorder` rooted at the run dir.

        ``episode_prefix`` is woven into output filenames as
        ``frames/{prefix}episode_NNNN.{ext}``. Trainers should pass
        ``"train_"`` to keep training captures distinct from future
        live-rollout captures sharing the same ``frames/`` directory.
        """
        if self._frame_recorder is None:
            self._frame_recorder = FrameRecorder(
                self.run_dir,
                fps=fps,
                fmt=fmt,
                max_frames_per_episode=max_frames_per_episode,
                max_dim=max_dim,
                episode_prefix=episode_prefix,
            )
        return self._frame_recorder

    # -- finalisation -----------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._t0 is not None:
            self.manifest.wall_time_s = time.time() - self._t0
        self.manifest.finished_at = RunManifest.utc_now_iso()
        try:
            self.manifest.write(self.run_dir / "manifest.json")
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            self._sink.close()
        except Exception:  # pragma: no cover
            pass


class NoOpLogger(ExperimentLogger):
    """Drop-in :class:`ExperimentLogger` whose every method is a no-op.

    Construct via :meth:`instance` (singleton) or by calling
    :meth:`ExperimentLogger.create` with ``output_dir=None``.
    """

    _singleton: Optional["NoOpLogger"] = None

    def __init__(self) -> None:  # noqa: D401 - intentional override
        # Skip super().__init__ to avoid touching the filesystem.
        from deap_pettingzoo.utils.logging.sinks import NoOpSink

        self.run_dir = Path()  # placeholder, never written to
        self.manifest = RunManifest(
            run_id="noop",
            algorithm="noop",
            env_name="noop",
        )
        self._sink = NoOpSink()
        self._t0 = None
        self._closed = False
        self._frame_recorder = None

    @classmethod
    def instance(cls) -> "NoOpLogger":
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def __enter__(self) -> "NoOpLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def row(self, table: str, row: Mapping[str, Any]) -> None:  # type: ignore[override]
        return None

    def scalar(self, name: str, value: float, step: int) -> None:  # type: ignore[override]
        return None

    def artifact(self, name: str, path: Path) -> None:  # type: ignore[override]
        return None

    def summary(self, summary: Mapping[str, Any]) -> None:  # type: ignore[override]
        return None

    def get_frame_recorder(self, **kwargs: Any) -> FrameRecorder:  # type: ignore[override]
        # Return a recorder whose run_dir is a temp-y placeholder and that is
        # disabled by default. Callers can flip ``enabled=True`` at their own
        # risk if they really want frames without an output_dir.
        rec = FrameRecorder(Path("."), **kwargs)
        rec.enabled = False
        return rec

    def close(self) -> None:  # type: ignore[override]
        self._closed = True
