"""SQLite persistence layer for the experiment queue.

Uses the standard-library :mod:`sqlite3` in WAL (Write-Ahead Logging) mode so
the dispatcher (single writer) and the API handlers (multiple readers) can
access the database concurrently without serialising.

All disk I/O is performed through :func:`asyncio.to_thread` so the event loop
never blocks. The database file is created next to the results root; callers
can override the path via the ``DEAP_SERVICE_DB`` environment variable.

Schema
------
experiments (id TEXT PK, label TEXT, spec_json TEXT,
    status TEXT CHECK IN {queued,running,completed,failed,cancelled},
    created_at REAL, started_at REAL, finished_at REAL,
    run_dirs_json TEXT, error TEXT, pid INTEGER,
    progress_json TEXT, enqueue_seq INTEGER)

queue_state (id INTEGER PK CHECK(id=1), paused INTEGER DEFAULT 0,
    worker_limit INTEGER DEFAULT 1)
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default database location — users can redirect via env var.
_DEFAULT_DB_DIR = os.environ.get("DEAP_RESULTS_ROOT", str(Path("results")))
_DEFAULT_DB_PATH = os.environ.get(
    "DEAP_SERVICE_DB",
    str(Path(_DEFAULT_DB_DIR) / "_service.db"),
)


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def _connect(path: str = _DEFAULT_DB_PATH, *, read_only: bool = False) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro" if read_only else str(path)
    conn = sqlite3.connect(
        uri if read_only else str(path),
        uri=True if read_only else False,
        check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id            TEXT PRIMARY KEY,
            label         TEXT NOT NULL DEFAULT '',
            spec_json     TEXT NOT NULL,
            status        TEXT NOT NULL DEFAULT 'queued'
                          CHECK (status IN ('queued','running','completed',
                                            'failed','cancelled')),
            created_at    REAL NOT NULL,
            started_at    REAL,
            finished_at   REAL,
            run_dirs_json TEXT,
            error         TEXT,
            pid           INTEGER,
            progress_json TEXT,
            enqueue_seq   INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_status     ON experiments(status);
        CREATE INDEX IF NOT EXISTS idx_enqueue_seq ON experiments(enqueue_seq);

        CREATE TABLE IF NOT EXISTS queue_state (
            id            INTEGER PRIMARY KEY CHECK (id = 1),
            paused        INTEGER NOT NULL DEFAULT 0,
            worker_limit  INTEGER NOT NULL DEFAULT 1
        );
        INSERT OR IGNORE INTO queue_state (id, paused, worker_limit) VALUES (1, 0, 1);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Async CRUD
# ---------------------------------------------------------------------------


async def _run_sync(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


# -- experiments -----------------------------------------------------------


async def insert_experiment(
    spec_json: str,
    label: str = "",
    path: Optional[str] = None,
) -> str:
    """Insert a new queued experiment; return its UUID id."""
    exp_id = uuid.uuid4().hex[:16]
    now = _now()

    def _insert():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(enqueue_seq), 0) + 1 FROM experiments"
            ).fetchone()
            next_seq = row[0]
            conn.execute(
                """INSERT INTO experiments
                   (id, label, spec_json, status, created_at, enqueue_seq)
                   VALUES (?, ?, ?, 'queued', ?, ?)""",
                (exp_id, label, spec_json, now, next_seq),
            )
            conn.commit()
            return exp_id
        finally:
            conn.close()

    return await _run_sync(_insert)


async def count_queued_before(exp_id: str, path: Optional[str] = None) -> int:
    """Return the number of queued experiments with smaller enqueue_seq."""

    def _count():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            row = conn.execute(
                "SELECT enqueue_seq FROM experiments WHERE id = ?", (exp_id,)
            ).fetchone()
            if row is None:
                return -1
            seq = row["enqueue_seq"]
            row2 = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE status = 'queued' "
                "AND enqueue_seq < ?",
                (seq,),
            ).fetchone()
            return row2[0]
        finally:
            conn.close()

    return await _run_sync(_count)


async def list_experiments(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    def _list():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            query = (
                "SELECT id, label, spec_json, status, created_at, started_at, "
                "finished_at, run_dirs_json, error, pid, progress_json, enqueue_seq "
                "FROM experiments"
            )
            params: tuple = ()
            if status:
                query += " WHERE status = ?"
                params = (status,)
            query += " ORDER BY enqueue_seq ASC LIMIT ? OFFSET ?"
            params = params + (limit, offset)
            rows = conn.execute(query, params).fetchall()
            return [_row_to_dict(r) for r in rows]
        finally:
            conn.close()

    return await _run_sync(_list)


async def get_experiment(exp_id: str, path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    def _get():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            row = conn.execute(
                "SELECT id, label, spec_json, status, created_at, started_at, "
                "finished_at, run_dirs_json, error, pid, progress_json, enqueue_seq "
                "FROM experiments WHERE id = ?",
                (exp_id,),
            ).fetchone()
            return _row_to_dict(row) if row else None
        finally:
            conn.close()

    return await _run_sync(_get)


async def set_running(exp_id: str, pid: int, path: Optional[str] = None) -> None:
    now = _now()

    def _set():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            conn.execute(
                "UPDATE experiments SET status='running', started_at=?, pid=? "
                "WHERE id=?",
                (now, pid, exp_id),
            )
            conn.commit()
        finally:
            conn.close()

    await _run_sync(_set)


async def set_completed(
    exp_id: str,
    run_dirs: List[str],
    path: Optional[str] = None,
) -> None:
    now = _now()

    def _set():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            conn.execute(
                "UPDATE experiments SET status='completed', finished_at=?, "
                "run_dirs_json=? WHERE id=?",
                (now, json.dumps(run_dirs), exp_id),
            )
            conn.commit()
        finally:
            conn.close()

    await _run_sync(_set)


async def set_failed(
    exp_id: str,
    error: str,
    run_dirs: Optional[List[str]] = None,
    path: Optional[str] = None,
) -> None:
    now = _now()
    dirs_json = json.dumps(run_dirs or [])

    def _set():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            conn.execute(
                "UPDATE experiments SET status='failed', finished_at=?, "
                "error=?, run_dirs_json=? WHERE id=?",
                (now, error, dirs_json, exp_id),
            )
            conn.commit()
        finally:
            conn.close()

    await _run_sync(_set)


async def set_cancelled(exp_id: str, path: Optional[str] = None) -> None:
    now = _now()

    def _set():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            conn.execute(
                "UPDATE experiments SET status='cancelled', finished_at=? WHERE id=?",
                (now, exp_id),
            )
            conn.commit()
        finally:
            conn.close()

    await _run_sync(_set)


async def set_interrupted_to_failed(path: Optional[str] = None) -> None:
    """Crash recovery: any running experiment -> failed('interrupted')."""
    now = _now()

    def _set():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            conn.execute(
                "UPDATE experiments SET status='failed', finished_at=?, "
                "error='Server interrupted while experiment was running' "
                "WHERE status='running'",
                (now,),
            )
            conn.commit()
        finally:
            conn.close()

    await _run_sync(_set)


async def delete_experiment(exp_id: str, path: Optional[str] = None) -> bool:
    def _del():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            cur = conn.execute(
                "DELETE FROM experiments WHERE id=? AND status NOT IN ('running')",
                (exp_id,),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    return await _run_sync(_del)


async def dequeue_next(path: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """Pop the next queued experiment (id, spec_json) or None.

    Only dequeues when the queue is not paused and no experiment is running.
    """

    def _pop():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            qs = conn.execute("SELECT paused FROM queue_state WHERE id=1").fetchone()
            if qs and qs["paused"]:
                return None
            running = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE status='running'"
            ).fetchone()[0]
            if running > 0:
                return None
            row = conn.execute(
                "SELECT id, spec_json FROM experiments WHERE status='queued' "
                "ORDER BY enqueue_seq ASC LIMIT 1"
            ).fetchone()
            conn.commit()
            return (row["id"], row["spec_json"]) if row else None
        finally:
            conn.close()

    return await _run_sync(_pop)


# -- queue_state -----------------------------------------------------------


async def get_queue_state(path: Optional[str] = None) -> Dict[str, Any]:
    def _get():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            qs = conn.execute(
                "SELECT paused, worker_limit FROM queue_state WHERE id=1"
            ).fetchone()
            running_id = conn.execute(
                "SELECT id FROM experiments WHERE status='running' LIMIT 1"
            ).fetchone()
            queued = conn.execute(
                "SELECT id FROM experiments WHERE status='queued' ORDER BY enqueue_seq ASC"
            ).fetchall()
            return {
                "paused": bool(qs["paused"]) if qs else False,
                "running_id": running_id["id"] if running_id else None,
                "queued_ids": [r["id"] for r in queued],
                "worker_limit": qs["worker_limit"] if qs else 1,
            }
        finally:
            conn.close()

    return await _run_sync(_get)


async def set_paused(paused: bool, path: Optional[str] = None) -> None:
    def _set():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            conn.execute(
                "UPDATE queue_state SET paused=? WHERE id=1", (int(paused),)
            )
            conn.commit()
        finally:
            conn.close()

    await _run_sync(_set)


async def get_pid(exp_id: str, path: Optional[str] = None) -> Optional[int]:
    def _get():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            row = conn.execute(
                "SELECT pid FROM experiments WHERE id=?", (exp_id,)
            ).fetchone()
            return row["pid"] if row else None
        finally:
            conn.close()

    return await _run_sync(_get)


async def update_progress(
    exp_id: str,
    progress_json: str,
    path: Optional[str] = None,
) -> None:
    def _upd():
        conn = _connect(path or _DEFAULT_DB_PATH)
        try:
            conn.execute(
                "UPDATE experiments SET progress_json=? WHERE id=?",
                (progress_json, exp_id),
            )
            conn.commit()
        finally:
            conn.close()

    await _run_sync(_upd)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    spec = {}
    try:
        spec = json.loads(row["spec_json"])
    except (json.JSONDecodeError, TypeError):
        pass
    run_dirs = []
    try:
        run_dirs = json.loads(row["run_dirs_json"] or "[]")
    except (json.JSONDecodeError, TypeError):
        pass
    progress = None
    try:
        progress_raw = row["progress_json"]
        if progress_raw:
            progress = json.loads(progress_raw)
    except (json.JSONDecodeError, TypeError):
        pass
    # Position: count of queued items ahead of this one in enqueue order.
    position = 0
    return {
        "id": row["id"],
        "label": row["label"] or "",
        "status": row["status"],
        "spec": spec,
        "created_at": _iso(row["created_at"]),
        "started_at": _iso(row["started_at"]),
        "finished_at": _iso(row["finished_at"]),
        "run_dirs": run_dirs,
        "error": row["error"],
        "pid": row["pid"],
        "position": position,
        "progress": progress,
    }


def _iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
