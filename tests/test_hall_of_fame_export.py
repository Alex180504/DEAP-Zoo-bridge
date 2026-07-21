"""Best-N hall-of-fame export for the genome-based trainers.

EATrainer/CMAESTrainer/PSOTrainer share
:func:`deap_pettingzoo.baselines.ea_trainer.build_genome_hall_of_fame_export`,
so exercising EA end-to-end covers the shared export path; the helper itself is
also unit-tested here with fake individuals (no env needed).
"""
import pytest


class TestBuildGenomeHallOfFameExport:
    """Unit tests for the shared genome export helper (no env / deps)."""

    class _FakeFitness:
        def __init__(self, value):
            self.values = (value,)

    class _FakeInd:
        def __init__(self, genome, value):
            self.genome = genome
            self.fitness = TestBuildGenomeHallOfFameExport._FakeFitness(value)

    def test_skips_champion_checkpoint_and_ranks_best_first(self, monkeypatch):
        import deap_pettingzoo.baselines.ea_trainer as ea

        saved = []

        def _fake_save(logger, genome, *, filename, kind, extra):
            saved.append((filename, extra))

        # Avoid constructing a real policy from fake genomes.
        monkeypatch.setattr(
            ea, "GenomePolicyWrapper", lambda genome, env_name, action_space=None: genome
        )

        # HallOfFame iterates best-first; emulate that with a descending list.
        hof = [self._FakeInd("g1", 5.0), self._FakeInd("g2", 3.0), self._FakeInd("g3", 1.0)]
        policies, summary = ea.build_genome_hall_of_fame_export(
            hof,
            env_name="env",
            action_space=None,
            logger=object(),
            save_genome=_fake_save,
            kind="k",
        )

        assert [p for p in policies] == ["g1", "g2", "g3"]
        assert summary == [
            {"rank": 1, "fitness": 5.0},
            {"rank": 2, "fitness": 3.0},
            {"rank": 3, "fitness": 1.0},
        ]
        # Champion (rank 1) is skipped; only ranks 2..N are written here.
        assert [name for name, _ in saved] == ["hof_02.json", "hof_03.json"]
        assert saved[0][1]["hof_rank"] == 2 and saved[0][1]["final_fitness"] == 3.0


pytest.importorskip("mpe2", reason="mpe2 not installed.")

from deap_pettingzoo.baselines import BudgetMode, EATrainer, TrainingConfig


def _cfg(tmp_path, seed: int = 0) -> TrainingConfig:
    return TrainingConfig(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=12,  # pop_size=6 -> ~2 generations
        num_seeds=1,
        seed=seed,
        max_steps_per_episode=10,
        output_dir=tmp_path,
    )


class TestEAHallOfFameExportLive:
    def test_hof_size_gt_one_retains_and_exports_top_n(self, tmp_path):
        trainer = EATrainer(hidden_size=8, population_size=6, hof_size=3)
        result = trainer.train(_cfg(tmp_path, seed=0))

        # In-memory top-N policies, best first; [0] is the returned champion.
        assert result.hall_of_fame is not None
        assert len(result.hall_of_fame) == 3
        assert result.hall_of_fame[0] is result.policy

        # JSON-safe summary is rank-ordered by descending fitness.
        summary = result.metrics["hall_of_fame"]
        assert [row["rank"] for row in summary] == [1, 2, 3]
        fits = [row["fitness"] for row in summary]
        assert fits == sorted(fits, reverse=True)
        assert fits[0] == pytest.approx(float(result.metrics["final_fitness"]))

        # Disk: champion policy.json + hof_02.json + hof_03.json.
        ckpts = {p.name for p in tmp_path.glob("**/checkpoints/*.json")}
        assert {"policy.json", "hof_02.json", "hof_03.json"} <= ckpts

    def test_default_hof_size_one_single_champion(self, tmp_path):
        trainer = EATrainer(hidden_size=8, population_size=6)  # hof_size defaults to 1
        result = trainer.train(_cfg(tmp_path, seed=0))

        assert result.hall_of_fame is not None
        assert len(result.hall_of_fame) == 1
        assert result.metrics["hall_of_fame"] == [
            {"rank": 1, "fitness": result.metrics["hall_of_fame"][0]["fitness"]}
        ]
        # No extra hof_NN.json checkpoints when hof_size == 1.
        extra = [p.name for p in tmp_path.glob("**/checkpoints/hof_*.json")]
        assert extra == []
