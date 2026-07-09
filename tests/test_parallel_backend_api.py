"""Explicit parallel-backend API surface and validation-warning tests.

Covers:
* ``num_workers`` / ``parallel_backend`` as first-class ``TrainingConfig``
  fields (validation + precedence over the trainer constructor).
* The ``backend_not_installed`` validation warning emitted by
  ``_validate_parallelism`` when a real backend is selected but absent.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig


# ---------------------------------------------------------------------------
# TrainingConfig fields
# ---------------------------------------------------------------------------


def _cfg(**kw):
    base = dict(
        env_name="simple_spread_v3",
        budget_mode=BudgetMode.EVAL_EPISODES,
        budget_value=8,
    )
    base.update(kw)
    return TrainingConfig(**base)


class TestTrainingConfigParallelFields:
    def test_defaults(self):
        c = _cfg()
        assert c.num_workers == 1
        assert c.parallel_backend == "process"

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="parallel_backend"):
            _cfg(parallel_backend="bogus")

    def test_rejects_negative_workers(self):
        with pytest.raises(ValueError, match="num_workers"):
            _cfg(num_workers=-1)

    def test_accepts_known_backends(self):
        for backend in ("process", "dask"):
            assert _cfg(parallel_backend=backend).parallel_backend == backend

    def test_scheduler_address_requires_dask(self):
        # Default backend is "process" — pairing must be rejected.
        with pytest.raises(ValueError, match="scheduler_address"):
            _cfg(scheduler_address="tcp://head:8786")

    def test_scheduler_address_accepted_with_dask(self):
        c = _cfg(parallel_backend="dask", scheduler_address="tcp://head:8786")
        assert c.scheduler_address == "tcp://head:8786"

    def test_blank_scheduler_address_normalised_to_none(self):
        # An empty UI field must not trip the backend-pairing check.
        c = _cfg(scheduler_address="   ")
        assert c.scheduler_address is None


class TestTrainerConstructorSchedulerAddress:
    def test_ea_rejects_scheduler_without_dask(self):
        from deap_pettingzoo.baselines.ea_trainer import EATrainer

        with pytest.raises(ValueError, match="scheduler_address"):
            EATrainer(scheduler_address="tcp://head:8786")

    def test_cma_rejects_scheduler_without_dask(self):
        from deap_pettingzoo.baselines.cma_trainer import CMAESTrainer

        with pytest.raises(ValueError, match="scheduler_address"):
            CMAESTrainer(scheduler_address="tcp://head:8786")

    def test_pso_rejects_scheduler_without_dask(self):
        from deap_pettingzoo.baselines.pso_trainer import PSOTrainer

        with pytest.raises(ValueError, match="scheduler_address"):
            PSOTrainer(scheduler_address="tcp://head:8786")

    def test_ea_accepts_scheduler_with_dask(self):
        from deap_pettingzoo.baselines.ea_trainer import EATrainer

        trainer = EATrainer(
            parallel_backend="dask", scheduler_address="tcp://head:8786"
        )
        assert trainer.scheduler_address == "tcp://head:8786"


# ---------------------------------------------------------------------------
# Trainer reads the explicit fields
# ---------------------------------------------------------------------------


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs at least 2 CPUs")
class TestTrainerReadsConfigFields:
    def _make_config(self, **kw):
        return _cfg(seed=0, **kw)

    def test_ea_reads_config_num_workers(self):
        pytest.importorskip("pettingzoo.mpe")
        from deap_pettingzoo.baselines.ea_trainer import EATrainer

        # Constructor leaves num_workers at the serial default; the config
        # field drives the effective worker count.
        trainer = EATrainer(hidden_size=8, population_size=4)
        result = trainer.train(self._make_config(num_workers=2))
        assert result.metadata["num_workers"] == 2
        assert result.metadata["parallel_backend"] == "process"
        # No remote scheduler in play: both remote-metadata fields stay None.
        assert result.metadata["scheduler_address"] is None
        assert result.metadata["dask_workers"] is None

    def test_config_backend_overrides_constructor(self):
        pytest.importorskip("pettingzoo.mpe")
        pytest.importorskip("dask.distributed")
        from deap_pettingzoo.baselines.ea_trainer import EATrainer

        # Constructor default backend is "process"; the config field wins.
        trainer = EATrainer(hidden_size=8, population_size=4)
        result = trainer.train(
            self._make_config(num_workers=2, parallel_backend="dask")
        )
        assert result.metadata["parallel_backend"] == "dask"
        assert np.isfinite(result.metrics["final_fitness"])

    def test_constructor_used_when_config_default(self):
        pytest.importorskip("pettingzoo.mpe")
        from deap_pettingzoo.baselines.ea_trainer import EATrainer

        # Config leaves num_workers at the default (1) -> constructor value used.
        trainer = EATrainer(hidden_size=8, population_size=4, num_workers=2)
        result = trainer.train(self._make_config())
        assert result.metadata["num_workers"] == 2

    def test_remote_scheduler_dispatch_and_metadata(self, monkeypatch):
        """A scheduler_address reaches a real cluster even with num_workers=1.

        Covers the serial-gate bypass through the trainer API plus the new
        result-metadata fields the HPC rigs assert on.
        """
        pytest.importorskip("pettingzoo.mpe")
        pytest.importorskip("dask.distributed")
        from dask.distributed import LocalCluster

        from deap_pettingzoo.baselines.ea_trainer import EATrainer
        from deap_pettingzoo.utils.parallel import DASK_SCHEDULER_ENV_VAR

        monkeypatch.delenv(DASK_SCHEDULER_ENV_VAR, raising=False)
        # Real worker processes: the worker-side env caches are per-process.
        cluster = LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
        )
        try:
            trainer = EATrainer(hidden_size=8, population_size=4)
            result = trainer.train(
                self._make_config(
                    num_workers=1,
                    parallel_backend="dask",
                    scheduler_address=cluster.scheduler_address,
                )
            )
        finally:
            cluster.close()
        assert result.metadata["parallel_backend"] == "dask"
        assert result.metadata["scheduler_address"] == cluster.scheduler_address
        # num_workers stays advisory metadata on the remote path.
        assert result.metadata["num_workers"] == 1
        assert result.metadata["dask_workers"]
        assert np.isfinite(result.metrics["final_fitness"])


# ---------------------------------------------------------------------------
# Validation warning for an uninstalled backend
# ---------------------------------------------------------------------------


def _parallel_stub(num_workers, backend, scheduler_address=None):
    return SimpleNamespace(
        parallelism=SimpleNamespace(
            num_workers=num_workers,
            parallel_backend=backend,
            scheduler_address=scheduler_address,
        )
    )


class TestBackendInstalledWarning:
    def _validate(self, spec):
        from deap_pettingzoo.baselines.experiment_spec import _validate_parallelism
        from deap_pettingzoo.catalog.schema import ValidationReport

        report = ValidationReport()
        _validate_parallelism(spec, report)
        return report

    def test_warns_when_backend_missing(self, monkeypatch):
        import deap_pettingzoo.catalog.availability as avail

        monkeypatch.setattr(avail, "module_available", lambda name: False)
        report = self._validate(_parallel_stub(2, "dask"))

        # Non-blocking: the report stays OK but carries a warning.
        assert report.ok
        codes = [i.code for i in report.issues]
        assert "backend_not_installed" in codes
        assert all(i.severity == "warning" for i in report.issues)
        assert all(
            i.path == "parallelism.parallel_backend" for i in report.issues
        )

    def test_no_warning_when_backend_available(self, monkeypatch):
        import deap_pettingzoo.catalog.availability as avail

        monkeypatch.setattr(avail, "module_available", lambda name: True)
        report = self._validate(_parallel_stub(2, "dask"))
        assert report.issues == []

    def test_no_warning_for_process_backend(self, monkeypatch):
        import deap_pettingzoo.catalog.availability as avail

        # Even if everything looks "missing", the stdlib process backend never
        # warns.
        monkeypatch.setattr(avail, "module_available", lambda name: False)
        report = self._validate(_parallel_stub(8, "process"))
        assert report.issues == []

    def test_no_warning_when_serial(self, monkeypatch):
        import deap_pettingzoo.catalog.availability as avail

        # num_workers <= 1 means the backend is never spun up.
        monkeypatch.setattr(avail, "module_available", lambda name: False)
        report = self._validate(_parallel_stub(1, "dask"))
        assert report.issues == []

    def test_error_when_scheduler_with_non_dask_backend(self):
        report = self._validate(
            _parallel_stub(2, "process", scheduler_address="tcp://head:8786")
        )
        assert not report.ok
        codes = [i.code for i in report.issues]
        assert "backend_mismatch" in codes
        assert all(
            i.path == "parallelism.scheduler_address" for i in report.issues
        )

    def test_no_error_when_scheduler_with_dask(self, monkeypatch):
        import deap_pettingzoo.catalog.availability as avail

        monkeypatch.setattr(avail, "module_available", lambda name: True)
        report = self._validate(
            _parallel_stub(2, "dask", scheduler_address="tcp://head:8786")
        )
        assert report.issues == []
