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
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Held-out re-evaluation defaults. Every standard sub-run is re-scored with a
# shared :class:`UnifiedEvaluator` after training so EA and RL backends carry a
# directly comparable metric.
_EVAL_EPISODES = 20
_EVAL_MAX_STEPS = 500
#: Base offset for the held-out eval RNG. The episode seeds are derived from
#: ``_EVAL_SEED_BASE + sub.seed`` so the *i*-th repeat of every algorithm is
#: scored on the *same* paired set of evaluation episodes.
_EVAL_SEED_BASE = 90_000_000
_EVAL_PROVENANCE = "held_out_unified"


# ProgressWriter


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


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as JSON to ``path`` via a temp-file + replace.

    ``sort_keys=True`` mirrors :meth:`RunManifest.to_json` so a re-evaluated
    ``manifest.json`` keeps the same key ordering as a freshly-written one.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(str(tmp), str(path))


def _persist_eval_metrics(
    run_dir: Path,
    eval_metrics: Dict[str, Any],
) -> None:
    """Merge held-out eval metrics into a run's on-disk summary.

    :attr:`RunRecord.summary` prefers ``manifest.json``'s ``final_summary``
    (written by the trainer before the runner regains control) and only falls
    back to ``summary.json``, so the eval keys are merged into **both** to keep
    the analysis/UI layers consistent regardless of which a loader reads.

    Best-effort: a persistence failure is logged but never raised, so it cannot
    fail an otherwise-complete run.
    """
    manifest_path = run_dir / "manifest.json"
    try:
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            final_summary = dict(manifest.get("final_summary") or {})
            final_summary.update(eval_metrics)
            manifest["final_summary"] = final_summary
            _atomic_write_json(manifest_path, manifest)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not merge eval metrics into %s", manifest_path)

    summary_path = run_dir / "summary.json"
    try:
        summary: Dict[str, Any] = {}
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(eval_metrics)
        _atomic_write_json(summary_path, summary)
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not merge eval metrics into %s", summary_path)


def _evaluate_policy(evaluator: Any, policy: Any, sub: Any, eval_seed: int) -> Any:
    """Run the held-out evaluation for one trained ``policy``.

    Multi-agent RL policies (those exposing ``policy_map()``, i.e.
    :class:`AgileRLPolicyWrapper`) are evaluated through the evaluator's
    per-agent ``policy_map`` path, which dispatches by agent id and so stays
    correct even when the active-agent set shrinks mid-episode. Single-policy
    backends (EA / CMA-ES / PSO) keep the simple broadcast path.
    """
    common = dict(
        env_name=sub.config.env_name,
        num_episodes=_EVAL_EPISODES,
        seed=eval_seed,
        env_config=sub.config.env_config,
    )
    policy_map = policy.policy_map() if hasattr(policy, "policy_map") else None
    if policy_map:
        return evaluator.evaluate(None, policy_map=policy_map, **common)
    return evaluator.evaluate(policy, **common)


def _record_held_out_eval(
    evaluator: Any,
    sub: Any,
    result: Any,
    metrics_row: Dict[str, Any],
    run_dir: Optional[str],
    label: str,
) -> None:
    """Re-evaluate ``result.policy`` on the held-out metric and record it.

    Mutates ``metrics_row`` in place (the per-sub-run entry of
    ``final_metrics``) with ``eval_mean_reward`` / ``eval_std_reward`` /
    ``eval_episodes`` / ``eval_metric_provenance``, and merges the same keys
    into the run's on-disk summary via :func:`_persist_eval_metrics`. The eval
    seed is derived from ``sub.seed`` so the i-th repeat of every algorithm is
    scored on the same paired episodes. Best-effort: any failure is logged and
    recorded as ``eval_metric_provenance="eval_failed"`` rather than raised, so
    re-eval can never abort an otherwise-complete run.
    """
    try:
        eval_seed = _EVAL_SEED_BASE + int(sub.seed)
        eval_result = _evaluate_policy(evaluator, result.policy, sub, eval_seed)
    except Exception as exc:  # noqa: BLE001 - re-eval must not abort the run
        logger.warning("Held-out re-eval failed for %s: %s", label, exc)
        metrics_row["eval_mean_reward"] = None
        metrics_row["eval_metric_provenance"] = "eval_failed"
        # Persist the failure marker (no eval_mean_reward) so the analysis layer
        # treats this run as "could not be scored on the held-out metric" and
        # excludes it from comparisons, rather than silently re-admitting it on
        # its incomparable self-reported metric on reload.
        if run_dir:
            _persist_eval_metrics(
                Path(run_dir), {"eval_metric_provenance": "eval_failed"}
            )
        return

    eval_metrics = {
        "eval_mean_reward": float(eval_result.mean_reward),
        "eval_std_reward": float(eval_result.std_reward),
        "eval_episodes": _EVAL_EPISODES,
        "eval_metric_provenance": _EVAL_PROVENANCE,
    }
    # Persist the held-out per-agent decomposition so the analysis layer's
    # per-agent view (Experiment.per_agent_table) can surface the comparable,
    # every-backend metric even when no opt-in trajectories.parquet exists.
    if eval_result.agent_rewards:
        eval_metrics["eval_agent_rewards"] = {
            str(agent): float(reward)
            for agent, reward in eval_result.agent_rewards.items()
        }
    metrics_row.update(eval_metrics)
    if run_dir:
        _persist_eval_metrics(Path(run_dir), eval_metrics)
    print(
        f"    held-out eval: mean={eval_result.mean_reward:.3f} "
        f"(±{eval_result.std_reward:.3f}, n={_EVAL_EPISODES})",
        flush=True,
    )


# Main entry


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
        logger.exception("Failed to parse spec %s", spec_file)
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
        logger.error("Spec validation failed: %s", errors)
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
        plan = build_configs(spec, output_root=experiment_dir)
    except Exception as exc:
        logger.exception("Failed to build experiment configs")
        _write_result(
            experiment_dir,
            "failed",
            [],
            error=f"Failed to build experiment configs: {exc}\n{traceback.format_exc()}",
        )
        sys.exit(1)

    # 4. Execute sub-runs
    from deap_pettingzoo.baselines.comparison import get_trainer
    from deap_pettingzoo.baselines.unified_evaluator import UnifiedEvaluator

    progress_writer = ProgressWriter(experiment_dir)
    run_dirs: List[str] = []
    final_metrics: Dict[str, Any] = {}
    total_runs = len(plan.sub_runs)

    # One shared held-out evaluator for the whole standard comparison so EA and
    # RL backends are scored on the same metrics. Co-evolution sub-runs
    # produce a per-side policy map rather than a single comparable policy, so
    # they keep their own reported metric and are not re-evaluated here.
    evaluator: Optional[UnifiedEvaluator] = (
        UnifiedEvaluator(max_steps=_EVAL_MAX_STEPS) if plan.mode == "standard" else None
    )

    for sub in plan.sub_runs:
        label = f"[{sub.run_index + 1}/{total_runs}] {sub.algorithm} seed={sub.seed}"
        # Keep the user-facing progress print on the terminal (always captured
        # in _stdout.log) and mirror it through logging for the structured feed.
        print(f"--> Starting {label}", flush=True)
        logger.info("Starting sub-run %s", label)

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
            logger.exception("Sub-run %s failed", label)
            error = (
                f"Sub-run [{sub.run_index + 1}/{total_runs}] "
                f"{sub.algorithm} seed={sub.seed} failed: {exc}\n"
                f"{traceback.format_exc()}"
            )
            _write_result(
                experiment_dir, "failed", run_dirs, final_metrics, error=error
            )
            sys.exit(1)

        run_dir = result.metadata.get("run_dir")
        if run_dir:
            run_dirs.append(run_dir)
        print(f"OK  Finished {label}  run_dir={run_dir}", flush=True)
        logger.info("Finished sub-run %s (run_dir=%s)", label, run_dir)

        # Collect best fitness for the aggregate summary.
        algo_key = f"{sub.algorithm}_s{sub.seed}"
        final_metrics[algo_key] = {
            "final_fitness": result.metrics.get("final_fitness"),
            "num_generations": result.metrics.get("num_generations"),
            "training_time_s": result.training_time,
        }

        # Held-out re-eval: score the trained policy with the shared evaluator
        # so every backend carries a comparable ``eval_mean_reward``.
        if evaluator is not None and result.policy is not None:
            _record_held_out_eval(
                evaluator, sub, result, final_metrics[algo_key], run_dir, label
            )

    if evaluator is not None:
        evaluator.close()

    # 5. Success
    print(f"OK  All {total_runs} sub-runs completed.", flush=True)
    logger.info("All %d sub-run(s) completed", total_runs)
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
    # Console-only: this process's stdout is redirected by the dispatcher to
    # <experiment-dir>/_stdout.log, so the structured log is retrievable per
    # experiment.  Level / format are inherited from the parent via env vars.
    from deap_pettingzoo.service.logging_config import configure_logging

    configure_logging(log_file=None)
    run_experiment(args.spec_file, args.experiment_dir)


# Eager import (below the functions to avoid circular issues with catalog).
from deap_pettingzoo.baselines.experiment_spec import build_configs  # noqa: E402

if __name__ == "__main__":
    main()
