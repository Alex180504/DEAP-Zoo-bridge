"""Regression tests: the EA-vs-RL comparison must be fair.

These pin the the following:

* the subprocess runner must re-evaluate **every** backend on a held-out
  metric, so RL rows are no longer ``None`` and EA/RL share a provenance
  (:func:`test_runner_mixed_spec_reports_comparable_metric`);
* the legacy :func:`compare_baselines` path must thread ``env_config`` into
  both training and evaluation (:func:`test_compare_baselines_threads_env_config`);
* the analysis layer must prefer the held-out ``eval_mean_reward`` and stop
  silently coalescing a self-reported training reward into the comparison
  score (:func:`test_analysis_prefers_held_out_metric`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import deap_pettingzoo.environments  # noqa: F401


# Runner: held-out re-eval for a mixed EA + RL spec


@pytest.mark.slow
def test_runner_mixed_spec_reports_comparable_metric(tmp_path):
    """A mixed ``["ea", "maddpg"]`` spec yields a non-``None`` held-out metric
    with matching provenance for **both** backends.
    """
    pytest.importorskip("agilerl")
    pytest.importorskip("torch")

    from deap_pettingzoo.baselines.experiment_spec import (
        ExperimentSpec,
        validate_spec,
    )
    from deap_pettingzoo.service.runner import run_experiment

    spec = ExperimentSpec.from_dict(
        {
            "environment": "simple_spread_v3",
            "algorithms": ["ea", "maddpg"],
            "env_config": {"max_cycles": 12},
            "budget": {"mode": "env_steps", "value": 500, "num_seeds": 1},
            "hyperparameters": {
                "ea": {
                    "population_size": 4,
                    "hidden_size": 8,
                    "max_steps_per_episode": 12,
                },
                "maddpg": {
                    "batch_size": 16,
                    "memory_size": 500,
                    "learning_delay": 50,
                },
            },
        }
    )
    report = validate_spec(spec)
    assert report.ok, [i.message for i in report.errors]

    spec_file = tmp_path / "_spec.json"
    spec_file.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    run_experiment(spec_file, tmp_path)

    result = json.loads((tmp_path / "_result.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed", result.get("error")

    final_metrics = result["final_metrics"]
    ea_key = next(k for k in final_metrics if k.startswith("ea_s"))
    maddpg_key = next(k for k in final_metrics if k.startswith("maddpg_s"))

    provenances = set()
    for key in (ea_key, maddpg_key):
        row = final_metrics[key]
        assert row.get("eval_mean_reward") is not None, (key, row)
        assert row.get("eval_metric_provenance"), (key, row)
        provenances.add(row["eval_metric_provenance"])
        # Held-out per-agent decomposition is persisted so the
        # analysis per-agent view can surface it without trajectories.parquet.
        agent_rewards = row.get("eval_agent_rewards")
        assert isinstance(agent_rewards, dict) and agent_rewards, (key, row)

    # Both backends must be comparable on the *same* held-out provenance.
    assert provenances == {"held_out_unified"}, provenances


# compare_baselines: env_config threads into training AND evaluation


def test_compare_baselines_threads_env_config(monkeypatch):
    """``compare_baselines`` propagates a non-default ``env_config`` to every
    per-run :class:`TrainingConfig` and to the held-out eval call.
    """
    from deap_pettingzoo.baselines import comparison as comp
    from deap_pettingzoo.baselines.base import (
        BudgetMode,
        ComparisonConfig,
        EvalResult,
        TrainingResult,
    )

    train_env_configs: list = []
    eval_env_configs: list = []

    class _SpyPolicy:
        def get_action(self, observation, *, action_mask=None):
            return 0

        def reset(self):
            pass

    class _SpyTrainer:
        def __init__(self, **kwargs):
            pass

        def train(self, config, *, progress_callback=None):
            train_env_configs.append(dict(config.env_config))
            return TrainingResult(
                algorithm="ea",
                policy=_SpyPolicy(),
                metrics={"final_fitness": 1.0},
                metadata={},
                training_time=0.0,
            )

        def supports_budget_mode(self, mode):
            return True

        def get_capabilities(self):
            return {}

    def _spy_evaluate(self, policy, env_name, num_episodes, seed=None,
                      render=False, env_config=None, **kwargs):
        eval_env_configs.append(dict(env_config) if env_config else None)
        return EvalResult(
            mean_reward=1.0,
            std_reward=0.0,
            min_reward=1.0,
            max_reward=1.0,
            num_episodes=num_episodes,
        )

    monkeypatch.setattr(comp, "get_trainer", lambda name, **kw: _SpyTrainer(**kw))
    monkeypatch.setattr(comp.UnifiedEvaluator, "evaluate", _spy_evaluate)

    config = ComparisonConfig(
        algorithms=["ea"],
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=500,
        num_seeds=1,
        eval_episodes=2,
        env_config={"max_cycles": 7},
    )

    comp.compare_baselines(config, validate=False)

    assert train_env_configs, "trainer.train was never called"
    assert all(ec.get("max_cycles") == 7 for ec in train_env_configs), train_env_configs
    assert eval_env_configs, "evaluator.evaluate was never called"
    assert all(
        ec is not None and ec.get("max_cycles") == 7 for ec in eval_env_configs
    ), eval_env_configs


# Analysis: held-out metric preferred; training reward not silently coalesced


def test_analysis_prefers_held_out_metric():
    """The comparison score prefers the held-out ``eval_mean_reward`` and tags
    its provenance.
    """
    from deap_pettingzoo.analysis.loaders import RunRecord

    held_out = RunRecord(
        run_dir=Path("x"),
        manifest={},
        summary={"eval_mean_reward": 42.0, "mean_episode_reward": 7.0},
    )
    assert held_out.eval_mean_reward == 42.0
    assert held_out.comparison_score == 42.0
    assert held_out.score_provenance == "held_out"

    # A self-reported RL training reward must not masquerade as final fitness.
    training_only = RunRecord(
        run_dir=Path("x"),
        manifest={},
        summary={"mean_episode_reward": 7.0},
    )
    assert training_only.final_fitness is None
    assert training_only.score_provenance == "training_reward"

    # Genuine optimisation fitness is still recognised (EA / CMA-ES path).
    fitness = RunRecord(
        run_dir=Path("x"),
        manifest={},
        summary={"final_fitness": 3.0},
    )
    assert fitness.final_fitness == 3.0
    assert fitness.comparison_score == 3.0
    assert fitness.score_provenance == "training_fitness"


def test_experiment_score_provenances_flags_mixed():
    """``Experiment.score_provenances`` exposes per-group provenance so the UI
    can block a comparison that mixes held-out and self-reported metrics.
    """
    from deap_pettingzoo.analysis.loaders import Experiment, RunRecord

    def _run(algo: str, summary: dict) -> RunRecord:
        return RunRecord(
            run_dir=Path("x"),
            manifest={"algorithm": algo, "env_name": "simple_spread_v3"},
            summary=summary,
        )

    # Mixed: one held-out group vs one training-reward-only group.
    mixed = Experiment(
        root=Path("x"),
        runs=[
            _run("ea", {"eval_mean_reward": -10.0}),
            _run("maddpg", {"mean_episode_reward": -5.0}),
        ],
    )
    provs = mixed.score_provenances(metric="final_fitness")
    assert provs["ea"] == {"held_out"}
    assert provs["maddpg"] == {"training_reward"}
    union = set().union(*provs.values())
    assert len({p for p in union if p != "none"}) > 1  # UI must block this

    # Clean: both groups carry a held-out metric -> comparable.
    clean = Experiment(
        root=Path("x"),
        runs=[
            _run("ea", {"eval_mean_reward": -10.0, "final_fitness": -8.0}),
            _run("maddpg", {"eval_mean_reward": -5.0, "mean_episode_reward": -5.0}),
        ],
    )
    clean_provs = clean.score_provenances(metric="final_fitness")
    clean_union = set().union(*clean_provs.values())
    assert clean_union == {"held_out"}
    # final_scores prefers the held-out value for both backends.
    scores = clean.final_scores(metric="final_fitness")
    assert scores["ea"] == [-10.0]
    assert scores["maddpg"] == [-5.0]


def test_build_report_suppresses_mixed_provenance_comparison(tmp_path):
    """``build_report`` must not publish a pairwise/cross-task verdict when the
    compared groups mix held-out and self-reported metrics — it suppresses the
    section instead, mirroring the Compare page gating.
    """
    from deap_pettingzoo.analysis.loaders import Experiment, RunRecord
    from deap_pettingzoo.analysis.report import build_report

    def _run(algo: str, summary: dict) -> RunRecord:
        return RunRecord(
            run_dir=Path("x"),
            manifest={"algorithm": algo, "env_name": "simple_spread_v3"},
            summary=summary,
        )

    mixed = Experiment(
        root=Path("x"),
        runs=[
            _run("ea", {"eval_mean_reward": -10.0}),
            _run("ea", {"eval_mean_reward": -11.0}),
            _run("maddpg", {"mean_episode_reward": -5.0}),
            _run("maddpg", {"mean_episode_reward": -6.0}),
        ],
    )
    report_path = build_report(mixed, tmp_path, fmt="md", include_figures=False)
    text = report_path.read_text(encoding="utf-8")
    assert "Mixed metric provenance" in text
    assert "Suppressed" in text
    assert "mixed:" in text  # provenance line names the offending tags

    clean = Experiment(
        root=Path("x"),
        runs=[
            _run("ea", {"eval_mean_reward": -10.0}),
            _run("ea", {"eval_mean_reward": -11.0}),
            _run("maddpg", {"eval_mean_reward": -5.0}),
            _run("maddpg", {"eval_mean_reward": -6.0}),
        ],
    )
    clean_path = build_report(clean, tmp_path / "clean", fmt="md", include_figures=False)
    clean_text = clean_path.read_text(encoding="utf-8")
    assert "Mixed metric provenance" not in clean_text
    assert "**Score provenance:** held_out" in clean_text
    # The clean report still renders a real pairwise table (not suppressed).
    assert "Suppressed" not in clean_text
    assert "p_value" in clean_text


# Post-review hardening (adversarial-review follow-ups)


def _exp_with(runs):
    from deap_pettingzoo.analysis.loaders import Experiment, RunRecord

    return Experiment(
        root=Path("x"),
        runs=[
            RunRecord(
                run_dir=Path("x"),
                manifest={"algorithm": algo, "env_name": "simple_spread_v3"},
                summary=summary,
            )
            for algo, summary in runs
        ],
    )


def test_final_scores_provenance_filter_and_warning(caplog):
    """``final_scores`` warns when it pools mixed provenance and supports a
    ``provenance=`` filter so direct callers can request a comparable subset.
    """
    import logging

    exp = _exp_with(
        [
            ("ea", {"eval_mean_reward": -10.0}),
            ("ea", {"final_fitness": -8.0}),  # training_fitness, no held-out
            ("maddpg", {"eval_mean_reward": -5.0}),
        ]
    )

    # Unfiltered pooling spans held_out + training_fitness -> a warning is logged.
    with caplog.at_level(logging.WARNING, logger="deap_pettingzoo.analysis.loaders"):
        pooled = exp.final_scores(metric="final_fitness")
    assert sorted(pooled["ea"]) == [-10.0, -8.0]
    assert any("mixed provenance" in r.message for r in caplog.records)

    # The held-out-only filter drops the training_fitness run and warns nothing.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="deap_pettingzoo.analysis.loaders"):
        held_out = exp.final_scores(metric="final_fitness", provenance="held_out")
    assert held_out == {"ea": [-10.0], "maddpg": [-5.0]}
    assert not any("mixed provenance" in r.message for r in caplog.records)

    # A uniformly-held-out experiment must NOT warn.
    caplog.clear()
    uniform = _exp_with(
        [("ea", {"eval_mean_reward": -10.0}), ("maddpg", {"eval_mean_reward": -5.0})]
    )
    with caplog.at_level(logging.WARNING, logger="deap_pettingzoo.analysis.loaders"):
        uniform.final_scores(metric="final_fitness")
    assert not any("mixed provenance" in r.message for r in caplog.records)


def test_eval_failed_run_excluded_from_comparison():
    """A run tagged ``eval_failed`` is excluded from the comparison rather
    than silently re-admitted on its self-reported training metric.
    """
    from deap_pettingzoo.analysis.loaders import RunRecord

    failed = RunRecord(
        run_dir=Path("x"),
        manifest={"algorithm": "maddpg"},
        # Persisted failure marker alongside a stale self-reported reward.
        summary={"eval_metric_provenance": "eval_failed", "mean_episode_reward": -3.0},
    )
    assert failed.score_provenance == "eval_failed"
    assert failed.comparison_score is None  # not re-admitted on training reward

    exp = _exp_with(
        [
            ("ea", {"eval_mean_reward": -10.0}),
            ("maddpg", {"eval_metric_provenance": "eval_failed", "mean_episode_reward": -3.0}),
        ]
    )
    scores = exp.final_scores(metric="final_fitness")
    assert scores == {"ea": [-10.0]}  # the failed run contributes nothing
    provs = exp.score_provenances(metric="final_fitness")
    assert "maddpg" not in provs  # excluded entirely, not tagged comparable


def test_agilerl_policy_map_dispatches_by_agent_id():
    """``AgileRLPolicyWrapper.policy_map`` dispatches by agent id, so it is
    correct regardless of call order or a shrunken active-agent set (unlike the
    round-robin broadcast path which assumes a fixed full roster).
    """
    import numpy as np

    from deap_pettingzoo.baselines.agilerl_policy import AgileRLPolicyWrapper

    class _FakeAgent:
        # Decentralized actor: each agent's action echoes its own observation[0].
        agent_ids = ["a0", "a1", "a2"]

        def get_action(self, obs_dict, training=False):
            return ({aid: np.asarray([obs[0]]) for aid, obs in obs_dict.items()}, None)

    wrapper = AgileRLPolicyWrapper(_FakeAgent(), ["a0", "a1", "a2"], "fake")
    pmap = wrapper.policy_map()

    # Query out of roster order and as a subset — each view returns ITS action.
    assert pmap["a2"].get_action(np.array([2.0, 9.0])) == 2.0
    assert pmap["a0"].get_action(np.array([0.0, 9.0])) == 0.0
    assert pmap["a1"].get_action(np.array([1.0, 9.0])) == 1.0

    # A fresh episode where a0 never acts (terminated) must still score a1/a2.
    pmap2 = wrapper.policy_map()
    for view in set(pmap2.values()):
        view.reset()
    assert pmap2["a1"].get_action(np.array([1.0, 9.0])) == 1.0
    assert pmap2["a2"].get_action(np.array([2.0, 9.0])) == 2.0
