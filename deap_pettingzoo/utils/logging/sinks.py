"""Sink implementations for the logging subsystem.

A :class:`Sink` consumes logging events and persists them. Sinks
compose via :class:`MultiSink`; trainers always interact with sinks
through :class:`~deap_pettingzoo.utils.logging.ExperimentLogger`.

All sinks are designed to be safe to ``close()`` multiple times and
to silently swallow late writes after close (so a trainer can reuse
the logger across an evaluation phase that follows training).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence


class Sink(Protocol):
    """Minimal sink contract used by :class:`ExperimentLogger`.

    A sink need only implement the methods it supports — the default
    implementations are no-ops. ``ExperimentLogger`` calls every method
    on every sink and the sink decides what to persist.
    """

    def open(self, run_dir: Path) -> None: ...

    def log_row(self, table: str, row: Mapping[str, Any]) -> None: ...

    def log_scalar(self, name: str, value: float, step: int) -> None: ...

    def log_artifact(self, name: str, path: Path) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class _BaseSink:
    """Convenience base providing no-op defaults for the :class:`Sink` API."""

    def open(self, run_dir: Path) -> None:  # pragma: no cover - no-op default
        pass

    def log_row(self, table: str, row: Mapping[str, Any]) -> None:  # pragma: no cover
        pass

    def log_scalar(
        self, name: str, value: float, step: int
    ) -> None:  # pragma: no cover
        pass

    def log_artifact(self, name: str, path: Path) -> None:  # pragma: no cover
        pass

    def flush(self) -> None:  # pragma: no cover
        pass

    def close(self) -> None:  # pragma: no cover
        pass


class NoOpSink(_BaseSink):
    """Sink that drops every event. Used by :class:`NoOpLogger`."""


class MultiSink(_BaseSink):
    """Fan out every event to a list of child sinks."""

    def __init__(self, sinks: Sequence[Sink]) -> None:
        self._sinks: List[Sink] = list(sinks)
        self._closed = False

    def add(self, sink: Sink) -> None:
        self._sinks.append(sink)

    def open(self, run_dir: Path) -> None:
        for s in self._sinks:
            s.open(run_dir)

    def log_row(self, table: str, row: Mapping[str, Any]) -> None:
        if self._closed:
            return
        for s in self._sinks:
            s.log_row(table, row)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if self._closed:
            return
        for s in self._sinks:
            s.log_scalar(name, value, step)

    def log_artifact(self, name: str, path: Path) -> None:
        if self._closed:
            return
        for s in self._sinks:
            s.log_artifact(name, path)

    def flush(self) -> None:
        for s in self._sinks:
            s.flush()

    def close(self) -> None:
        if self._closed:
            return
        for s in self._sinks:
            try:
                s.close()
            except Exception:  # pragma: no cover - best-effort close
                pass
        self._closed = True


class JsonManifestSink(_BaseSink):
    """Persists named JSON artefacts via :meth:`log_row` to ``<run_dir>/<table>.json``.

    Used for one-shot writes such as the final summary. Each
    ``log_row(table, row)`` call **overwrites** the file — the row is
    treated as the full content. Only tables in :attr:`tables` are
    persisted; unknown tables are ignored (so streaming tables like
    ``metrics`` / ``episodes`` flow only to :class:`CsvScalarSink`).
    """

    DEFAULT_TABLES: tuple[str, ...] = ("summary",)

    def __init__(self, tables: Optional[Sequence[str]] = None) -> None:
        self._run_dir: Optional[Path] = None
        self.tables = tuple(tables) if tables is not None else self.DEFAULT_TABLES

    def open(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def log_row(self, table: str, row: Mapping[str, Any]) -> None:
        if self._run_dir is None or table not in self.tables:
            return
        path = self._run_dir / f"{table}.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(dict(row), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        tmp.replace(path)


class CsvScalarSink(_BaseSink):
    """Append-only CSV writer keyed by ``table`` name.

    Each ``log_row(table, row)`` call appends ``row`` as a CSV line in
    ``<run_dir>/<table>.csv``. The header is taken from the first row's
    keys; subsequent rows reuse that header. Unknown keys raise
    :class:`ValueError` to catch schema drift early.
    """

    def __init__(self) -> None:
        self._run_dir: Optional[Path] = None
        self._writers: Dict[str, csv.DictWriter] = {}
        self._files: Dict[str, Any] = {}
        self._headers: Dict[str, List[str]] = {}

    def open(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_writer(self, table: str, fieldnames: Iterable[str]) -> csv.DictWriter:
        if table in self._writers:
            return self._writers[table]
        if self._run_dir is None:
            raise RuntimeError("CsvScalarSink: open() not called")
        path = self._run_dir / f"{table}.csv"
        # Newline='' per csv module docs to avoid blank lines on Windows.
        f = path.open("w", newline="", encoding="utf-8")
        header = list(fieldnames)
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        self._files[table] = f
        self._writers[table] = writer
        self._headers[table] = header
        return writer

    def log_row(self, table: str, row: Mapping[str, Any]) -> None:
        if self._run_dir is None:
            return
        if table in self._writers:
            header = self._headers[table]
            unknown = [k for k in row.keys() if k not in header]
            if unknown:
                raise ValueError(
                    f"CsvScalarSink: row for table {table!r} has unknown "
                    f"columns {unknown}; first-row header was {header}. "
                    "Keep row schemas stable or use a fresh table name."
                )
            # Pad missing keys with empty string for CSV compatibility.
            self._writers[table].writerow({k: row.get(k, "") for k in header})
        else:
            writer = self._ensure_writer(table, row.keys())
            writer.writerow(dict(row))
        # Best-effort flush so partial runs are inspectable.
        f = self._files.get(table)
        if f is not None:
            f.flush()

    def flush(self) -> None:
        for f in self._files.values():
            try:
                f.flush()
            except Exception:  # pragma: no cover
                pass

    def close(self) -> None:
        for f in self._files.values():
            try:
                f.close()
            except Exception:  # pragma: no cover
                pass
        self._files.clear()
        self._writers.clear()


class ParquetTrajectorySink(_BaseSink):
    """Buffered Parquet writer for high-volume per-step rollout data.

    Buffers rows in memory; flushes to ``<run_dir>/<table>.parquet``
    on :meth:`close` (or every ``flush_every`` rows if set).
    Requires ``pyarrow`` (install via the ``[logging]`` extra).
    """

    def __init__(self, *, flush_every: Optional[int] = None) -> None:
        self._run_dir: Optional[Path] = None
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._flush_every = flush_every
        self._writers: Dict[str, Any] = {}
        self._schemas: Dict[str, Any] = {}

    def open(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._run_dir.mkdir(parents=True, exist_ok=True)

    def log_row(self, table: str, row: Mapping[str, Any]) -> None:
        if self._run_dir is None:
            return
        self._buffers.setdefault(table, []).append(dict(row))
        if self._flush_every and len(self._buffers[table]) >= self._flush_every:
            self._flush_table(table)

    def _flush_table(self, table: str) -> None:
        rows = self._buffers.get(table)
        if not rows:
            return
        if self._run_dir is None:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "ParquetTrajectorySink requires pyarrow. Install with "
                "`pip install 'deap-pettingzoo[logging]'`."
            ) from exc

        path = self._run_dir / f"{table}.parquet"
        table_obj = pa.Table.from_pylist(rows)
        if table not in self._writers:
            self._writers[table] = pq.ParquetWriter(path, table_obj.schema)
            self._schemas[table] = table_obj.schema
        # Re-cast against the established schema to avoid type drift.
        try:
            table_obj = table_obj.cast(self._schemas[table])
        except Exception:  # pragma: no cover - schema drift surfaces on close
            pass
        self._writers[table].write_table(table_obj)
        rows.clear()

    def flush(self) -> None:
        for table in tuple(self._buffers):
            self._flush_table(table)

    def close(self) -> None:
        self.flush()
        for w in self._writers.values():
            try:
                w.close()
            except Exception:  # pragma: no cover
                pass
        self._writers.clear()


class TensorBoardSink(_BaseSink):
    """Mirror scalar events to TensorBoard event files.

    Writes to ``<run_dir>/tb/``. Requires ``tensorboard`` (install via
    the ``[tensorboard]`` extra). Lazy import — sink construction does
    not import tensorboard.
    """

    def __init__(self) -> None:
        self._run_dir: Optional[Path] = None
        self._writer: Any = None

    def open(self, run_dir: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "TensorBoardSink requires tensorboard + torch. Install via "
                "`pip install 'deap-pettingzoo[tensorboard]'` or "
                "`pip install 'deap-pettingzoo[rl-baselines]'`."
            ) from exc
        self._run_dir = Path(run_dir)
        tb_dir = self._run_dir / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(tb_dir))

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if self._writer is None:
            return
        try:
            self._writer.add_scalar(name, float(value), int(step))
        except Exception:  # pragma: no cover - defensive
            pass

    def log_row(self, table: str, row: Mapping[str, Any]) -> None:
        # Auto-promote numeric row fields to scalars under "<table>/<key>".
        if self._writer is None:
            return
        step = row.get("iteration", row.get("step", 0))
        try:
            step_i = int(step)
        except (TypeError, ValueError):
            step_i = 0
        for k, v in row.items():
            if k in ("iteration", "step"):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.log_scalar(f"{table}/{k}", float(v), step_i)

    def flush(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception:  # pragma: no cover
                pass

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # pragma: no cover
                pass
            self._writer = None
