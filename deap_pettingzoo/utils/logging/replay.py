"""Frame recording for environment replay.

:class:`FrameRecorder` consumes ``rgb_array`` frames returned by
``env.render()`` and writes a video (mp4) or animated GIF artefact to
``<run_dir>/frames/episode_<n>.{mp4,gif}``. This is opt-in to keep
training loops fast.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


class FrameRecorder:
    """Lightweight rgb_array → mp4/gif recorder.

    Lifecycle:

    1. ``recorder = FrameRecorder(run_dir, fps=20, fmt="mp4")``
    2. ``recorder.start_episode(n)`` at episode begin
    3. ``recorder.add_frame(env.render())`` per env step (best-effort;
       silently skips when ``frame is None``)
    4. ``recorder.end_episode()`` writes the artefact and returns its path

    Format selection: ``mp4`` requires ``imageio[ffmpeg]``; ``gif`` works
    with the base ``imageio`` install. If the requested format is not
    available the recorder transparently falls back to ``gif`` and emits
    a single warning.

    Parameters
    ----------
    run_dir:
        Output root; frames are written to ``run_dir/frames/``.
    fps:
        Playback frame rate written to the video container.
    fmt:
        ``"mp4"`` (requires ``imageio[ffmpeg]``) or ``"gif"``.
    max_frames_per_episode:
        If set, ``add_frame`` silently drops frames once this many have
        been buffered, capping both memory use and encode time.
    max_dim:
        If set, each frame is downscaled via numpy index subsampling so
        that its larger dimension is at most ``max_dim`` pixels before
        buffering.  Useful for large envs (e.g. 6000×6000) that would
        exceed H.264 macroblock limits or exhaust RAM.
        No external dependency — pure numpy.
    episode_prefix:
        String prepended to each output filename, e.g. ``"ea_"`` →
        ``frames/ea_episode_0001.mp4``.

    Use ``recorder.enabled = False`` to short-circuit recording without
    tearing the recorder down.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        fps: int = 20,
        fmt: str = "mp4",
        max_frames_per_episode: Optional[int] = None,
        max_dim: Optional[int] = None,
        episode_prefix: str = "",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.fps = int(fps)
        self.fmt = str(fmt).lower().lstrip(".")
        if self.fmt not in {"mp4", "gif"}:
            raise ValueError(f"FrameRecorder: fmt must be 'mp4' or 'gif', got {fmt!r}")
        self.max_frames_per_episode = max_frames_per_episode
        self.max_dim = int(max_dim) if max_dim is not None else None
        self.episode_prefix = str(episode_prefix)
        self.enabled: bool = True
        self._frames: List[np.ndarray] = []
        self._episode_idx: Optional[int] = None
        self._frames_dir = self.run_dir / "frames"
        self._warned_fallback = False
        # Sidecar metadata: every successfully written artefact appends an
        # entry to ``self._recordings`` and the list is mirrored to
        # ``frames/recordings.json`` so the UI can associate a recording
        # with the training iteration/generation it captured.
        self._recordings: List[Dict[str, Any]] = []
        self._pending_meta: Dict[str, Any] = {}
        self._t0 = time.time()

    # -- lifecycle ---------------------------------------------------

    def start_episode(
        self,
        episode_idx: int,
        *,
        iteration: Optional[int] = None,
        env_steps: Optional[int] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Begin buffering frames for a new episode.

        ``iteration``/``env_steps``/``extra`` are optional metadata stashed
        for the upcoming :meth:`end_episode` call: they let the recorder
        record *which* training iteration/generation this artefact captured
        in the ``frames/recordings.json`` sidecar. Omitting them still
        produces a coherent sidecar entry (filename + episode index only).
        """
        self._episode_idx = int(episode_idx)
        self._frames = []
        self._pending_meta = {
            "iteration": int(iteration) if iteration is not None else None,
            "env_steps": int(env_steps) if env_steps is not None else None,
            "extra": dict(extra) if extra else None,
        }

    def add_frame(self, frame: Any) -> None:
        if not self.enabled or self._episode_idx is None or frame is None:
            return
        if (
            self.max_frames_per_episode is not None
            and len(self._frames) >= self.max_frames_per_episode
        ):
            return
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            return  # silently skip malformed frames
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if self.max_dim is not None:
            h, w = arr.shape[:2]
            if max(h, w) > self.max_dim:
                scale = self.max_dim / max(h, w)
                # Floor each side at 2px: H.264 needs even, >=2 dimensions,
                # so a too-small max_dim (e.g. 1) would otherwise yield a
                # 1x1 frame the encoder rejects ("width not divisible by 2"),
                # producing a silent 0-frame mp4.
                nh = max(2, int(h * scale))
                nw = max(2, int(w * scale))
                ri = np.linspace(0, h - 1, nh, dtype=int)
                ci = np.linspace(0, w - 1, nw, dtype=int)
                arr = arr[np.ix_(ri, ci)]
        # H.264/yuv420p requires even width and height; trim 1 px if needed.
        # Guard against trimming a 1-pixel dimension to zero.
        if arr.shape[0] % 2 != 0 and arr.shape[0] > 1:
            arr = arr[:-1]
        if arr.shape[1] % 2 != 0 and arr.shape[1] > 1:
            arr = arr[:, :-1]
        self._frames.append(arr)

    def end_episode(self) -> Optional[Path]:
        if not self.enabled or self._episode_idx is None or not self._frames:
            self._frames = []
            self._episode_idx = None
            return None
        try:
            import imageio.v2 as imageio  # type: ignore
        except ImportError:
            if not self._warned_fallback:
                import warnings

                warnings.warn(
                    "FrameRecorder: `imageio` is not installed, so no replay "
                    "videos will be written. Install the logging extra "
                    "(`pip install deap-pettingzoo[logging]`) to enable "
                    "frame recording.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_fallback = True
            self._frames = []
            self._episode_idx = None
            self._pending_meta = {}
            return None

        self._frames_dir.mkdir(parents=True, exist_ok=True)
        ep = self._episode_idx
        fmt = self.fmt
        path = self._frames_dir / f"{self.episode_prefix}episode_{ep:04d}.{fmt}"

        try:
            if fmt == "mp4":
                imageio.mimsave(
                    path, self._frames, fps=self.fps, macro_block_size=1, quality=9
                )
            else:
                # imageio.v2 gif uses 'duration' in seconds per frame.
                imageio.mimsave(path, self._frames, duration=1.0 / max(self.fps, 1))
        except Exception:
            # Fallback to GIF on ffmpeg-related failures.
            if fmt == "mp4":
                if not self._warned_fallback:
                    import warnings

                    warnings.warn(
                        "FrameRecorder: mp4 write failed (ffmpeg missing?). "
                        "Falling back to GIF for this and subsequent episodes.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_fallback = True
                self.fmt = "gif"
                path = self._frames_dir / f"{self.episode_prefix}episode_{ep:04d}.gif"
                try:
                    imageio.mimsave(path, self._frames, duration=1.0 / max(self.fps, 1))
                except Exception:
                    path = None  # type: ignore[assignment]
            else:
                path = None  # type: ignore[assignment]

        if path is not None:
            self._append_recording(path, ep)

        self._frames = []
        self._episode_idx = None
        self._pending_meta = {}
        return path

    # -- sidecar metadata --------------------------------------------

    def _append_recording(self, path: Path, episode_idx: int) -> None:
        """Record ``path`` in ``frames/recordings.json`` (best-effort).

        Associates the just-written artefact with the iteration/generation
        metadata stashed by :meth:`start_episode`. Any I/O error is
        swallowed so recording can never break a training run.
        """
        meta = self._pending_meta or {}
        entry: Dict[str, Any] = {
            "file": path.name,
            "episode_idx": int(episode_idx),
            "iteration": meta.get("iteration"),
            "env_steps": meta.get("env_steps"),
            "wall_time": round(time.time() - self._t0, 3),
        }
        if meta.get("extra"):
            entry["extra"] = meta["extra"]
        self._recordings.append(entry)
        try:
            tmp = self._frames_dir / "recordings.json.tmp"
            tmp.write_text(json.dumps(self._recordings, indent=2), encoding="utf-8")
            tmp.replace(self._frames_dir / "recordings.json")
        except Exception:  # pragma: no cover - best-effort sidecar
            pass
