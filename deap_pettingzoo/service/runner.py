"""Subprocess entry-point for executing an experiment from the queue.

Launched by :class:`~deap_pettingzoo.service.dispatcher.Dispatcher` via
:class:`subprocess.Popen`. The runner:

1. Loads an :class:`~deap_pettingzoo.baselines.experiment_spec.ExperimentSpec`
   from the ``--spec-file`` argument.
2. Re-validates the spec (defense in depth).
3. Calls :func:`~deap_pettingzoo.baselines.experiment_spec.build_configs`.
4. Iterates each :class:`SubRun`, instantiating the trainer and calling
   :meth:`~BaselineTrainer.train` with a
   :class:`ProgressWriter` callback that atomically writes
   ``<experiment-dir>/_progress.json``.
5. Writes ``<experiment-dir>/_result.json`` with status, discovered run dirs,
   final metrics, and any error.
6. Exits 0 on completion, non-zero on failure.

Stdout and stderr are inherited from the parent process; the dispatcher
redirects them to ``_stdout.log`` when spawning the subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# ProgressWriter
# ---------------------------------------------------------------------------


class ProgressWriter:
    """Wraps the trainer's ``progress_callback`` to enrich events and
    atomically persist the latest snapshot to ``_progress.json``.

    Written via ``tmp + os.replace`` so readers (the Queue page polling
    ``GET /experiments/{id}/progress``) never see a half-written file.

    Attributes:
        cb: The callable to pass as ``progress_callback`` to
            :meth:`BaselineTrainer.train`.
    """

    def __init__(
        self,
        experiment_dir: Path,
    ) -> None:
        self._dir = experiment_dir
        self._path = experiment_dir / "_progress.json"
        self._tmp = experiment_dir / "_progress.tmp"

    def __call__(self, event: Any) -> None:
        """Progress-callback protocol — receive one :class:`ProgressEvent`."""
        d = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        try:
            txt = json.dumps(d)
            self._tmp.write_text(txt, encoding="utf-8")
            os.replace(str(self._tmp), str(self._path))
        except OSError:
            pass  # best-effort


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def _load_spec(spec_file: Path) -> Any:
    spec_text = spec_file.read_text(encoding="utf-8")
    spec_data = json.loads(spec_text)
    from deap_pettingzoo.baselines.experiment_spec import ExperimentSpec
    return ExperimentSpec.from_dict(spec_data)


def _write_result(
    experiment_dir: Path,
    status: str,
    run_dirs: List[str],
    final_metrics: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "status": status,
        "run_dirs": run_dirs,
        "finished_at": time.time(),
    }
    if final_metrics is not None:
        payload["final_metrics"] = final_metrics
    if error is not None:
        payload["error"] = error
    result_path = experiment_dir / "_result.json"
    tmp_path = experiment_dir / "_result.tmp"
    try:
        txt = json.dumps(payload)
        tmp_path.write_text(txt, encoding="utf-8")
        os.replace(str(tmp_path), str(result_path))
    except OSError:  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_experiment(
    spec_file: Path,
    experiment_dir: Path,
) -> None:
    """Execute an experiment from a spec file. Exits the process on failure.

    Called from ``main()`` (CLI) and also from :func:`run_experiment_in_process`
    (test helper). Raises nothing on success; raises on unrecoverable error.
    """
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load spec
    try:
        spec = _load_spec(spec_file)
    except Exception as exc:
        _write_result(
            experiment_dir,
            "failed",
            [],
            error=f"Failed to parse spec: {exc}\n{traceback.format_exc()}",
        )
        sys.exit(1)

    # 2. Re-validate (defense in depth)
    from deap_pettingzoo.baselines.experiment_spec import validate_spec
    report = validate_spec(spec)
    if not report.ok:
        errors = [
            {"path": i.path, "code": i.code, "message": i.message}
            for i in report.errors
        ]
        _write_result(
            experiment_dir,
            "failed",
            [],
            error="Validation failed",
            final_metrics={"validation_errors": errors},
        )
        sys.exit(1)

    # 3. Build configs
    try:
        plan = build_configs(spec)
    except Exception as exc:
        _write_result(
            experiment_dir,
            "failed",
            [],
            error=f"Failed to build experiment configs: {exc}\n{traceback.format_exc()}",
        )
        sys.exit(1)

    # 4. Execute sub-runs
    from deap_pettingzoo.baselines.comparison import get_trainer

    progress_writer = ProgressWriter(experiment_dir)
    run_dirs: List[str] = []
    final_metrics: Dict[str, Any] = {}
    total_runs = len(plan.sub_runs)

    for sub in plan.sub_runs:
        label = f"[{sub.run_index + 1}/{total_runs}] {sub.algorithm} seed={sub.seed}"
        print(f"--> Starting {label}", flush=True)

        # Wire per-sub-run progress enrichment.
        def _make_cb(sr=sub):
            run_label = f"{sr.algorithm} s{sr.seed}"
            def cb(event):
                e = event.to_dict() if hasattr(event, "to_dict") else dict(event)
                e.setdefault("run_index", sr.run_index)
                e.setdefault("total_runs", total_runs)
                e.setdefault("algorithm", sr.algorithm)
                e.setdefault("seed", sr.seed)
                progress_writer(e)
            return cb

        try:
            trainer = get_trainer(sub.algorithm, **sub.trainer_params)
            result = trainer.train(sub.config, progress_callback=_make_cb())
        except Exception as exc:
            error = (
                f"Sub-run [{sub.run_index + 1}/{total_runs}] "
                f"{sub.algorithm} seed={sub.seed} failed: {exc}\n"
                f"{traceback.format_exc()}"
            )
            _write_result(experiment_dir, "failed", run_dirs, final_metrics, error=error)
            sys.exit(1)

        run_dir = result.metadata.get("run_dir")
        if run_dir:
            run_dirs.append(run_dir)
        print(f"OK  Finished {label}  run_dir={run_dir}", flush=True)

        # Collect best fitness for the aggregate summary.
        algo_key = f"{sub.algorithm}_s{sub.seed}"
        final_metrics[algo_key] = {
            "final_fitness": result.metrics.get("final_fitness"),
            "num_generations": result.metrics.get("num_generations"),
            "training_time_s": result.training_time,
        }

    # 5. Success
    print(f"OK  All {total_runs} sub-runs completed.", flush=True)
    _write_result(experiment_dir, "completed", run_dirs, final_metrics)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="DEAP-bridge experiment runner (subprocess)."
    )
    parser.add_argument(
        "--spec-file",
        required=True,
        type=Path,
        help="Path to the experiment spec JSON file.",
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        type=Path,
        help="Directory for progress, stdout and result artefacts.",
    )
    args = parser.parse_args(argv)
    run_experiment(args.spec_file, args.experiment_dir)


# Eager import (below the functions to avoid circular issues with catalog).
from deap_pettingzoo.baselines.experiment_spec import build_configs  # noqa: E402


if __name__ == "__main__":
    main()
