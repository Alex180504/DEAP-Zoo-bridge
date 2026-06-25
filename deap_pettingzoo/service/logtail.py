"""Shared log-tailing helper.

Used by ``GET /experiments/{id}/stdout`` and ``GET /logs`` in
:mod:`deap_pettingzoo.service.app`, and by the dependency-install job status
endpoint in :mod:`deap_pettingzoo.service.install_jobs`, so the three share one
forward-streaming / last-N-lines implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def read_tail(
    log_path: Path,
    offset: Optional[int],
    limit: Optional[int],
    lines: Optional[int],
) -> Tuple[str, int, bool]:
    """Read a chunk from a log file, returning ``(text, next_offset, eof)``.

    With ``lines`` set, returns the last N lines via a byte-seek from EOF
    (dropping a partial first line); otherwise returns up to ``limit`` bytes
    from ``offset``.  Missing or unreadable files yield ``("", 0, True)``.
    """
    if not log_path.exists():
        return "", 0, True

    text = ""
    next_offset = offset or 0
    eof = False
    try:
        if lines is not None:
            # Read at most lines*300 bytes from the end; if that chunk starts
            # mid-line we drop the partial first line (unless at file start).
            with open(log_path, "rb") as fh:
                fh.seek(0, 2)
                file_size = fh.tell()
                read_start = max(0, file_size - lines * 300)
                fh.seek(read_start)
                raw_bytes = fh.read()
            all_lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
            if read_start > 0 and len(all_lines) > 1:
                all_lines = all_lines[1:]
            text = "\n".join(all_lines[-lines:])
            next_offset = file_size
            eof = True
        else:
            with open(log_path, "rb") as fh:
                start = offset if offset is not None else 0
                fh.seek(0, 2)  # end
                file_size = fh.tell()
                if start >= file_size:
                    text = ""
                    next_offset = file_size
                    eof = True
                else:
                    fh.seek(start)
                    chunk_size = min(limit or 65536, file_size - start)
                    text = fh.read(chunk_size).decode("utf-8", errors="replace")
                    next_offset = start + chunk_size
                    eof = next_offset >= file_size
    except OSError:
        return "", 0, True
    return text, next_offset, eof
