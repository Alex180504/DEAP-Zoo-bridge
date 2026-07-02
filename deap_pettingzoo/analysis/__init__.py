"""Analysis library for deap-bridge experiment results.

This package is a *pure consumer* of the on-disk artefacts produced by
:mod:`deap_pettingzoo.utils.logging`. It is intentionally decoupled from
the trainer code path: nothing in :mod:`deap_pettingzoo.baselines`
imports from here, so the analysis stack can be installed and used
without touching production training paths (and vice versa — training
runs without ``deap-pettingzoo[analysis]`` installed).

The public surface is intentionally small:

- :func:`load_run` / :func:`load_experiment` — hydrate on-disk runs
  into :class:`RunRecord` / :class:`Experiment` objects.
- :func:`iqm`, :func:`bootstrap_ci`, :func:`mean_with_ci`,
  :func:`learning_curve_with_ci` — aggregate statistics with
  uncertainty estimates.
- :func:`mann_whitney`, :func:`welch_t`, :func:`cohens_d`,
  :func:`cliffs_delta`, :func:`pairwise_table` — pairwise comparison
  utilities.
- :mod:`rliable_bridge` — opt-in wrappers around the ``rliable`` package
  (performance profiles, probability of improvement, sample efficiency).
- :mod:`plots` / :mod:`report` — visualisation and report rendering;
  both lazy-import their optional dependencies.

Heavy optional dependencies (``pandas``, ``scipy``, ``rliable``,
``matplotlib``, ``seaborn``, ``jinja2``) are imported lazily inside the
functions that need them so that ``import deap_pettingzoo.analysis``
succeeds with only ``numpy`` available.
"""

from deap_pettingzoo.analysis.aggregate import (
    bootstrap_ci,
    iqm,
    learning_curve_with_ci,
    mean_with_ci,
    resolve_y_col,
)
from deap_pettingzoo.analysis.compare import (
    cliffs_delta,
    cohens_d,
    mann_whitney,
    pairwise_table,
    welch_t,
)
from deap_pettingzoo.analysis.loaders import (
    Experiment,
    RunRecord,
    load_experiment,
    load_run,
    per_agent_table,
)
from deap_pettingzoo.analysis.multi_task import (
    collect_score_matrices,
    final_score_table,
)
from deap_pettingzoo.analysis.normalize import (
    NormalizationSpec,
    normalize_score_matrices,
)
from deap_pettingzoo.analysis.policies import (
    CoevolutionPolicyBundle,
    PolicyReloadUnavailable,
    load_policy,
)
from deap_pettingzoo.analysis.rliable_bridge import (
    aggregate_metrics_forest,
)

__all__ = [
    "CoevolutionPolicyBundle",
    "Experiment",
    "NormalizationSpec",
    "PolicyReloadUnavailable",
    "RunRecord",
    "aggregate_metrics_forest",
    "bootstrap_ci",
    "cliffs_delta",
    "cohens_d",
    "collect_score_matrices",
    "final_score_table",
    "iqm",
    "learning_curve_with_ci",
    "load_experiment",
    "load_policy",
    "load_run",
    "mann_whitney",
    "mean_with_ci",
    "normalize_score_matrices",
    "pairwise_table",
    "per_agent_table",
    "resolve_y_col",
    "welch_t",
]
