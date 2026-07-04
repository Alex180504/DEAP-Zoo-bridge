"""Round-trip tests for policy checkpoint persistence.

Covers:
- ``save_ea_genome`` writes a valid JSON file under ``<run_dir>/checkpoints/``
  and registers it in ``manifest.artifacts['policy']``.
- ``analysis.policies.load_policy`` rehydrates that JSON into a working
  :class:`GenomePolicyWrapper`.
- ``coevolution_v1`` bundle round-trips per-side payloads.
- The save/register helpers are no-ops for :class:`NoOpLogger`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deap_pettingzoo.analysis.loaders import load_run
from deap_pettingzoo.analysis.policies import (
    CoevolutionPolicyBundle,
    PolicyReloadUnavailable,
    load_policy,
)
from deap_pettingzoo.genomes.fixed_topology import FixedTopologyGenome
from deap_pettingzoo.utils.logging import (
    KIND_COEVOLUTION,
    KIND_QDAX_REPERTOIRE,
    ExperimentLogger,
    NoOpLogger,
    register_existing_artifact,
    save_agilerl_agent,
    save_ea_genome,
)


def _make_logger(tmp_path: Path, *, algorithm: str = "ea") -> ExperimentLogger:
    return ExperimentLogger.create(
        output_dir=tmp_path,
        algorithm=algorithm,
        env_name="simple_spread_v3",
        seed=0,
        budget_mode="GENERATIONS",
        budget_value=1,
        hyperparams={},
    )


def test_save_ea_genome_writes_and_registers(tmp_path: Path):
    genome = FixedTopologyGenome(layer_sizes=[4, 8, 2], activation="tanh", random_seed=1)
    with _make_logger(tmp_path) as log:
        path = save_ea_genome(log, genome, extra={"final_fitness": 1.5})
        assert path is not None
        assert path.is_file()
        assert path.parent.name == "checkpoints"
        entry = log.manifest.artifacts["policy"]
        assert entry["kind"] == "ea_genome_v1"
        assert entry["path"] == "checkpoints/policy.json"
        assert entry["final_fitness"] == 1.5

    rd = next(iter(tmp_path.iterdir()))
    payload = json.loads((rd / "checkpoints" / "policy.json").read_text())
    assert payload["layer_sizes"] == [4, 8, 2]
    assert payload["activation"] == "tanh"


def test_save_ea_genome_is_noop_for_noop_logger():
    noop = NoOpLogger()
    genome = FixedTopologyGenome(layer_sizes=[3, 2], random_seed=0)
    assert save_ea_genome(noop, genome) is None


def test_load_policy_round_trip_ea(tmp_path: Path):
    genome = FixedTopologyGenome(
        layer_sizes=[6, 4, 2], activation="tanh", random_seed=42
    )
    expected_weights = genome.weights.copy()

    with _make_logger(tmp_path) as log:
        save_ea_genome(log, genome)

    rd = next(iter(tmp_path.iterdir()))
    rec = load_run(rd)
    policy = load_policy(rec)

    assert policy is not None
    inner = policy._genome
    np.testing.assert_array_equal(inner.weights, expected_weights)
    assert inner.layer_sizes == [6, 4, 2]

    # Smoke: get_action must return something for a matching observation.
    obs = np.zeros(6, dtype=np.float32)
    action = policy.get_action(obs)
    assert action is not None


def test_load_policy_missing_artefact_raises(tmp_path: Path):
    with _make_logger(tmp_path) as log:
        log.summary({"final_fitness": 0.0})
    rd = next(iter(tmp_path.iterdir()))
    rec = load_run(rd)
    with pytest.raises(FileNotFoundError):
        load_policy(rec)


def test_load_policy_unknown_kind_raises(tmp_path: Path):
    genome = FixedTopologyGenome(layer_sizes=[3, 2], random_seed=0)
    with _make_logger(tmp_path) as log:
        save_ea_genome(log, genome)
        # Tamper with the manifest entry to use a fake kind.
        log.manifest.artifacts["policy"]["kind"] = "totally_made_up_v1"

    rd = next(iter(tmp_path.iterdir()))
    rec = load_run(rd)
    with pytest.raises(ValueError, match="Unknown policy artefact kind"):
        load_policy(rec)


def test_register_existing_artifact_for_qdax(tmp_path: Path):
    fake_repertoire = tmp_path / "repertoire.pkl"
    fake_repertoire.write_bytes(b"\x80\x05.")  # not a real pickle, just a file
    with _make_logger(tmp_path, algorithm="map_elites") as log:
        path = register_existing_artifact(
            log, kind=KIND_QDAX_REPERTOIRE, path=fake_repertoire
        )
        assert path == fake_repertoire.resolve()
        entry = log.manifest.artifacts["policy"]
        assert entry["kind"] == KIND_QDAX_REPERTOIRE
        # Lives outside run_dir, so absolute path should be stored.
        assert Path(entry["path"]).is_absolute()


def test_register_existing_artifact_skips_missing_file(tmp_path: Path):
    with _make_logger(tmp_path, algorithm="map_elites") as log:
        path = register_existing_artifact(
            log, kind=KIND_QDAX_REPERTOIRE, path=tmp_path / "nope.pkl"
        )
        assert path is None
        assert "policy" not in log.manifest.artifacts


def test_qdax_kind_raises_unavailable_on_load(tmp_path: Path):
    fake = tmp_path / "repertoire.pkl"
    fake.write_bytes(b"\x80\x05.")
    with _make_logger(tmp_path, algorithm="map_elites") as log:
        register_existing_artifact(log, kind=KIND_QDAX_REPERTOIRE, path=fake)

    # tmp_path also holds the fake repertoire.pkl, so pick the run *directory*
    # explicitly rather than relying on iterdir() ordering (differs by OS).
    rd = next(p for p in tmp_path.iterdir() if p.is_dir())
    rec = load_run(rd)
    with pytest.raises(PolicyReloadUnavailable):
        load_policy(rec)


def test_save_agilerl_agent_uses_save_checkpoint(tmp_path: Path):
    captured = {}

    class FakeAgent:
        def save_checkpoint(self, path: str) -> None:
            captured["path"] = path
            Path(path).write_text("ok", encoding="utf-8")

    with _make_logger(tmp_path, algorithm="ippo") as log:
        path = save_agilerl_agent(log, FakeAgent())
        assert path is not None
        assert path.is_file()
        assert captured["path"] == str(path)
        assert log.manifest.artifacts["policy"]["kind"] == "agilerl_v1"


def test_coevolution_bundle_round_trip(tmp_path: Path):
    g_alpha = FixedTopologyGenome(layer_sizes=[4, 3, 2], random_seed=1)
    g_beta = FixedTopologyGenome(layer_sizes=[4, 3, 2], random_seed=2)

    with _make_logger(tmp_path, algorithm="coevolution") as log:
        from deap_pettingzoo.utils.logging.checkpoints import (
            _checkpoints_dir,
            _register,
        )

        target = _checkpoints_dir(log.run_dir) / "policy.json"
        target.write_text(
            json.dumps(
                {
                    "schema": "coevolution_v1",
                    "reporting_side": "alpha",
                    "sides": {
                        "alpha": {
                            "kind": "ea_genome_v1",
                            "genome": g_alpha.to_dict(),
                            "best_fitness": 1.0,
                        },
                        "beta": {
                            "kind": "ea_genome_v1",
                            "genome": g_beta.to_dict(),
                            "best_fitness": 0.5,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        _register(
            log,
            kind=KIND_COEVOLUTION,
            path=target,
            extra={"reporting_side": "alpha", "side_names": ["alpha", "beta"]},
        )

    rd = next(iter(tmp_path.iterdir()))
    rec = load_run(rd)
    bundle = load_policy(rec)
    assert isinstance(bundle, CoevolutionPolicyBundle)
    assert bundle.reporting_side == "alpha"
    assert bundle.side_names() == ["alpha", "beta"]

    pol_alpha = bundle.get_side("alpha")
    np.testing.assert_array_equal(pol_alpha._genome.weights, g_alpha.weights)

    with pytest.raises(KeyError):
        bundle.get_side("gamma")
