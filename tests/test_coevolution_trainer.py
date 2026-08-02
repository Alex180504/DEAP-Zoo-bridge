"""Smoke tests for :class:`CoEvolutionTrainer`.

One test per variant plus mixed EA-vs-frozen, exercised end-to-end on
``simple_tag_v3`` with tiny budgets. These are not convergence tests —
they verify the orchestration plumbing (bootstrap, synchronous /
alternating scheduling, HoF snapshotting, reporting policy
construction) produces sane :class:`TrainingResult`s.
"""

from __future__ import annotations

import pytest

pytest.importorskip("deap")
pytest.importorskip("pettingzoo.mpe")

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.coevolution import (
    CoEvolutionConfig,
    CoEvolutionTrainer,
    ElitistHallOfFame,
    RingHallOfFame,
    Side,
)
from deap_pettingzoo.environments.pettingzoo_wrapper import PettingZooWrapper


def _tiny_ea_cfg(seed: int) -> dict:
    return {"population_size": 4, "hidden_size": 8, "seed": seed}


def _tiny_cma_cfg(seed: int) -> dict:
    return {"population_size": 4, "hidden_size": 8, "seed": seed, "sigma0": 0.3}


def _base_tcfg(cfg: CoEvolutionConfig, budget: int = 40) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_tag_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=budget,
        coevolution_config=cfg,
    )


class TestCoevolutionConfigTypedEntryPoint:
    """Coevolution is configured via the typed
    ``TrainingConfig.coevolution_config`` entry point, not the untyped
    ``algorithm_params['coevolution']``."""

    def test_bad_type_fails_at_construction(self):
        with pytest.raises(TypeError, match="coevolution_config"):
            TrainingConfig(
                env_name="simple_tag_v3",
                budget_mode=BudgetMode.EVAL_EPISODES,
                budget_value=40,
                coevolution_config={"not": "a CoEvolutionConfig"},
            )

    def test_none_coevolution_config_rejected_by_trainer(self):
        # A TrainingConfig with no coevolution_config at all is a valid
        # construction (e.g. for a non-coevolution trainer sharing the same
        # dataclass) but CoEvolutionTrainer.train() must reject it clearly.
        config = TrainingConfig(
            env_name="simple_tag_v3",
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=40,
        )
        with pytest.raises(ValueError, match="coevolution_config"):
            CoEvolutionTrainer().train(config)

    def test_valid_config_is_accepted_and_used(self):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pred", ("adversary",), "ea", _tiny_ea_cfg(0)),
                Side("prey", ("agent",), "ea", _tiny_ea_cfg(1)),
            ],
        )
        config = _base_tcfg(cfg, budget=40)
        assert config.coevolution_config is cfg
        result = CoEvolutionTrainer().train(config)
        assert result.algorithm == "coevolution"


def test_translate_budget_counts_qdax_batch_size() -> None:
    cfg = CoEvolutionConfig(
        mode="competitive_2pop",
        sides=[
            Side(
                "pursuers",
                ("adversary",),
                "qdax_map_elites",
                {"batch_size": 8},
            ),
            Side("evaders", ("agent",), "ea", {"population_size": 4}),
        ],
        k_opponents=2,
        max_steps_per_episode=10,
    )
    train_cfg = TrainingConfig(
        env_name="simple_tag_v3",
        budget_mode=BudgetMode.ENV_STEPS,
        budget_value=720,
        coevolution_config=cfg,
    )

    assert CoEvolutionTrainer()._translate_budget(train_cfg, cfg) == 3


class TestCompetitive2Pop:
    def test_end_to_end_synchronous(self):
        cfg = CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
                Side("evaders", ("agent",), "ea", _tiny_ea_cfg(1)),
            ],
            k_opponents=2,
            max_steps_per_episode=20,
            seed=42,
        )
        result = CoEvolutionTrainer().train(_base_tcfg(cfg))
        assert result.algorithm == "coevolution"
        assert result.metadata["mode"] == "competitive_2pop"
        assert result.metrics["num_generations"] >= 1
        # Both evolving sides populated.
        assert set(result.metrics["final_best"]) == {"pursuers", "evaders"}
        # Per-gen history accumulates.
        for name in ("pursuers", "evaders"):
            hist = result.metrics["per_side_history"][name]
            assert len(hist) == result.metrics["num_generations"]
            for row in hist:
                assert {"generation", "mean", "std", "min", "max"}.issubset(row)
        # Optimisation actually happened: at least ONE side's best-ever
        # fitness improved end-vs-start. We don't insist both sides
        # improve simultaneously (competitive co-evolution is zero-sum-ish on tiny budgets)
        improved = []
        for name in ("pursuers", "evaders"):
            hist = result.metrics["per_side_history"][name]
            if len(hist) < 2:
                continue  # single-gen run, nothing to compare
            # Exclude gen 0 from the max.
            best_ever = max(row["max"] for row in hist[1:])
            improved.append(best_ever >= hist[0]["max"])
        assert not improved or any(improved), (
            "neither side's best-ever fitness improved end-vs-start: "
            f"{result.metrics['per_side_history']}"
        )
        # Reporting policy is executable.
        env = PettingZooWrapper("simple_tag_v3")
        obs = env.reset(seed=0)
        adv_obs = next(o for a, o in obs.items() if a.startswith("adversary"))
        action = result.policy.get_action(adv_obs)
        assert action is not None
        env.close()


class TestFinalChampionEval:
    """``CoEvolutionConfig.final_eval_fn`` — the absolute end-of-run score.

    ``final_best`` is a training fitness against co-evolved opponents, so it
    lives on a different scale per algorithm and pairing and cannot carry a
    cross-run comparison; this hook is where a comparable number comes from.
    """

    @staticmethod
    def _cfg(**kwargs) -> CoEvolutionConfig:
        return CoEvolutionConfig(
            mode="competitive_2pop",
            sides=[
                Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
                Side("evaders", ("agent",), "ea", _tiny_ea_cfg(1)),
            ],
            k_opponents=1,
            max_steps_per_episode=20,
            seed=7,
            **kwargs,
        )

    def test_hook_receives_states_and_context(self):
        seen = {}

        def _final_eval(states, context):
            seen["sides"] = sorted(states)
            seen["context"] = dict(context)
            return {"score": 1.25, "episodes": 4}

        result = CoEvolutionTrainer().train(
            _base_tcfg(self._cfg(final_eval_fn=_final_eval))
        )

        assert seen["sides"] == ["evaders", "pursuers"]
        ctx = seen["context"]
        assert ctx["env"] is not None
        assert ctx["num_generations"] == result.metrics["num_generations"]
        assert ctx["env_steps_consumed"] == result.metrics["env_steps_consumed"]
        assert result.metrics["final_eval"] == {"score": 1.25, "episodes": 4}

    def test_champions_are_evaluable_from_the_hook(self):
        """The hook runs before env.close(), so champions can still play."""
        rewards = {}

        def _final_eval(states, context):
            env = context["env"]
            obs = env.reset(seed=0)
            for name, state in states.items():
                policy = state.best_policy(env)
                agent_obs = next(
                    o for a, o in obs.items()
                    if a.startswith(state.side.role_prefixes)
                )
                rewards[name] = float(policy.get_action(agent_obs) is not None)
            return rewards

        CoEvolutionTrainer().train(_base_tcfg(self._cfg(final_eval_fn=_final_eval)))
        assert rewards == {"pursuers": 1.0, "evaders": 1.0}

    def test_absent_hook_leaves_final_eval_empty(self):
        result = CoEvolutionTrainer().train(_base_tcfg(self._cfg()))
        assert result.metrics["final_eval"] == {}

    def test_failing_hook_warns_but_keeps_the_run(self):
        def _boom(states, context):
            raise RuntimeError("evaluation exploded")

        with pytest.warns(RuntimeWarning, match="final_eval_fn raised"):
            result = CoEvolutionTrainer().train(
                _base_tcfg(self._cfg(final_eval_fn=_boom))
            )

        assert result.metrics["final_eval"] == {}
        assert result.metrics["num_generations"] >= 1
        assert result.policy is not None

    def test_summary_carries_eval_prefixed_keys(self, tmp_path):
        import json

        def _final_eval(states, context):
            return {"catches_mean": 2.5, "episodes": 8}

        config = _base_tcfg(self._cfg(final_eval_fn=_final_eval))
        config.output_dir = tmp_path
        CoEvolutionTrainer().train(config)

        summaries = list(tmp_path.rglob("summary.json"))
        assert summaries, "no summary.json written"
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        assert summary["eval_catches_mean"] == 2.5
        assert summary["eval_episodes"] == 8
        # The training-fitness score stays alongside, not replaced.
        assert "final_best" in summary


class TestSelfPlayHoF:
    def test_ring_archive_accumulates(self):
        cfg = CoEvolutionConfig(
            mode="self_play_hof",
            sides=[
                Side(
                    "pursuers", ("adversary",), "ea", _tiny_ea_cfg(0),
                    hof=RingHallOfFame(capacity=8),
                ),
                Side(
                    "evaders", ("agent",), "ea", _tiny_ea_cfg(1),
                    hof=ElitistHallOfFame(capacity=8),
                ),
            ],
            k_opponents=2,
            max_steps_per_episode=20,
            seed=42,
            snapshot_every=1,
        )
        result = CoEvolutionTrainer().train(_base_tcfg(cfg, budget=60))
        # HoF grows with each generation (plus bootstrap snapshot).
        for name in ("pursuers", "evaders"):
            assert result.metrics["hof_sizes"][name] >= 1
        assert result.metadata["mode"] == "self_play_hof"

    def test_requires_hof_on_each_side(self):
        with pytest.raises(ValueError, match="hof"):
            CoEvolutionConfig(
                mode="self_play_hof",
                sides=[
                    Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
                    Side("evaders", ("agent",), "ea", _tiny_ea_cfg(1)),
                ],
            )

    def test_requires_multiple_sides(self):
        with pytest.raises(ValueError, match="at least 2 sides"):
            CoEvolutionConfig(
                mode="self_play_hof",
                sides=[
                    Side(
                        "solo", ("adversary",), "ea", _tiny_ea_cfg(0),
                        hof=RingHallOfFame(capacity=4),
                    ),
                ],
            )


class TestCooperative:
    def test_mixed_ea_cma_alternating(self):
        cfg = CoEvolutionConfig(
            mode="cooperative",
            sides=[
                Side("pursuers", ("adversary",), "ea", _tiny_ea_cfg(0)),
                Side("evaders", ("agent",), "cma", _tiny_cma_cfg(1)),
            ],
            k_opponents=2,
            max_steps_per_episode=20,
            seed=42,
            schedule="alternating",
            alternating_generations=1,
        )
        result = CoEvolutionTrainer().train(_base_tcfg(cfg, budget=40))
        assert result.metadata["schedule"] == "alternating"
        assert result.metadata["sides"][0]["optimizer"] == "ea"
        assert result.metadata["sides"][1]["optimizer"] == "cma"


class TestConfigValidation:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            CoEvolutionConfig(
                mode="bogus",
                sides=[Side("a", ("agent",), "ea", _tiny_ea_cfg(0))],
            )

    def test_rejects_duplicate_side_names(self):
        with pytest.raises(ValueError, match="Duplicate side names"):
            CoEvolutionConfig(
                mode="cooperative",
                sides=[
                    Side("dup", ("adversary",), "ea", _tiny_ea_cfg(0)),
                    Side("dup", ("agent",), "ea", _tiny_ea_cfg(1)),
                ],
            )

    def test_rejects_unknown_reporting_side(self):
        with pytest.raises(ValueError, match="reporting_side"):
            CoEvolutionConfig(
                mode="competitive_2pop",
                sides=[
                    Side("a", ("adversary",), "ea", _tiny_ea_cfg(0)),
                    Side("b", ("agent",), "ea", _tiny_ea_cfg(1)),
                ],
                reporting_side="does_not_exist",
            )

    def test_competitive_requires_two_sides(self):
        with pytest.raises(ValueError, match="competitive_2pop"):
            CoEvolutionConfig(
                mode="competitive_2pop",
                sides=[Side("a", ("adversary",), "ea", _tiny_ea_cfg(0))],
            )
