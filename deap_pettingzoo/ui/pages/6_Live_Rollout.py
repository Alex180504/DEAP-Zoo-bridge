"""Live Rollout — reconstruct a saved policy and run an episode in real time.

Currently supports ``ea_genome_v1`` / ``cma_genome_v1`` /
``coevolution_v1`` checkpoints (see :mod:`deap_pettingzoo.analysis.policies`).
For ``agilerl_v1`` and ``qdax_repertoire_v1`` we surface the
:class:`PolicyReloadUnavailable` message produced by
:func:`load_policy` rather than crashing the page.
"""

from __future__ import annotations

import streamlit as st

from deap_pettingzoo.analysis.loaders import load_run
from deap_pettingzoo.analysis.policies import (
    CoevolutionPolicyBundle,
    PolicyReloadUnavailable,
    load_policy,
)
from deap_pettingzoo.ui.services import UIState, load_settings
from deap_pettingzoo.ui.services.data import discover_runs

st.set_page_config(page_title="Live Rollout", layout="wide")
st.title("Live rollout")

state = UIState.from_streamlit()
settings = load_settings(
    results_root=str(state.results_root) if state.results_root else None
)

if not state.selected_run_ids:
    st.info("Pick a run on the **Browse** page first.")
    st.stop()

all_runs = {r.run_id: r for r in discover_runs(settings.results_root)}
options = [rid for rid in state.selected_run_ids if rid in all_runs]
if not options:
    st.warning("Selected runs no longer exist on disk.")
    st.stop()

picked = st.selectbox("Run", options)
record = load_run(all_runs[picked].run_dir)

st.caption(
    f"Algorithm: **{record.algorithm}** · Env: **{record.env_name}** · "
    f"Seed: **{record.seed}**"
)

try:
    policy = load_policy(record)
except PolicyReloadUnavailable as exc:
    st.warning(str(exc))
    st.stop()
except FileNotFoundError as exc:
    st.error(f"No policy artefact registered: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 - last-resort surface
    st.error(f"Could not load policy: {exc}")
    st.stop()

if isinstance(policy, CoevolutionPolicyBundle):
    side = st.selectbox(
        "Side",
        policy.side_names(),
        index=(
            policy.side_names().index(policy.reporting_side)
            if policy.reporting_side in policy.side_names()
            else 0
        ),
    )
    policy = policy.get_side(side)

st.success(
    f"Loaded policy: `{type(policy).__name__}`. "
    "Live rollout requires an env factory; this scaffolding hooks the "
    "policy up — wire your env in the next phase of the UI plan."
)

st.code(
    "from deap_pettingzoo.analysis.loaders import load_run\n"
    "from deap_pettingzoo.analysis.policies import load_policy\n\n"
    f"record = load_run('{record.run_dir}')\n"
    "policy = load_policy(record)\n"
    "# Then call your env factory's reset/step loop with policy.get_action(obs).",
    language="python",
)
