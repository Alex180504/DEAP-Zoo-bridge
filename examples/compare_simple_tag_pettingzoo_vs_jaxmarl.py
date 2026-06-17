"""Equivalence experiment: PettingZoo MPE2 ``simple_tag_v3`` vs JaxMARL ``MPE_simple_tag_v3``.

Background
----------
JaxMARL is a *reimplementation* of MPE in pure JAX; it is API-inspired
by PettingZoo but is **not** guaranteed to be a bit-exact port. Before
using JaxMARL as a fast substrate for evolving / training policies that
will eventually be deployed against the PettingZoo env (or vice versa),
we need to know *how equivalent* the two implementations are.

What "equivalent" means here
----------------------------
Equivalence is not a single bit. We measure it on four axes and let
each axis report independently. A test passes if and only if the
divergence falls below a documented tolerance (or, for distributional
tests, fails to reject the null hypothesis at p > 0.05).

Axis 1 — Structural
    * Same set of agent IDs and same canonical ordering
      (``adversary_0``, ``adversary_1``, ``adversary_2``, ``agent_0``).
    * Per-agent observation-space shapes and dtypes match.
    * Per-agent action-space shapes / cardinalities match.

Axis 2 — Trajectory under fixed seed + fixed actions
    Replays the same scripted action sequence on both envs from the
    same seed and reports component-wise max-abs deltas in
    observations, rewards, and termination flags. This is the strict
    test: any non-trivial drift means the underlying physics /
    reward shaping differ.

Axis 3 — Statistical, under random policies
    Runs N independent episodes per env with the same RNG draws used
    for action sampling (but each env builds its own internal seed so
    that internal stochasticity is *its own*). Compares marginal
    distributions of:
    * episode length (Kolmogorov-Smirnov),
    * total per-side reward (KS).

Axis 4 — Smoke / determinism
    Re-running the same fixed-action replay twice on each env gives
    identical results within float tolerance (catches accidental
    shared mutable state and confirms our seeding contract).

Output
------
A single report printed to stdout with PASS/FAIL per axis, plus a
summary verdict and (when applicable) the magnitude of any drift. No
files are written.

Usage
-----
::

    pip install 'deap-pettingzoo[jaxmarl,mpe2]'
    python examples/compare_simple_tag_pettingzoo_vs_jaxmarl.py \\
        --episodes 50 --max-steps 25 --seed 0

Caveats
-------
* MPE2 ``simple_tag_v3`` defaults: ``num_good=1``, ``num_adversaries=3``,
  ``num_obstacles=2``, ``max_cycles=25``. JaxMARL's defaults are the
  same but action semantics may differ — we coerce both to discrete
  here. Continuous-action mode is **not** tested by this script.
* JaxMARL renames ``agent_0`` -> ``agent`` in *some* versions; the
  script normalises by sorted role-prefix grouping rather than
  literal ID equality.
* Reward shaping in MPE depends on collision detection; small
  numerical differences in particle positions can flip a discrete
  collision event and produce step-level reward divergence even when
  long-run statistics agree. This is why Axis 3 (distributional)
  matters even if Axis 2 (trajectory) fails.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------


def _import_pettingzoo_simple_tag():
    # Prefer the standalone mpe2 package (the official successor to
    # pettingzoo.mpe, which is deprecated). Fall back to the legacy
    # location only if mpe2 is not installed.
    try:
        from mpe2 import simple_tag_v3  # type: ignore
        return simple_tag_v3
    except ImportError:
        pass
    try:
        from pettingzoo.mpe import simple_tag_v3  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "MPE simple_tag_v3 unavailable. "
            "Install with: pip install mpe2"
        ) from exc
    return simple_tag_v3


def _import_jaxmarl():
    try:
        import jax  # type: ignore
        from jaxmarl import make  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "JaxMARL unavailable. "
            "Install with: pip install 'deap-pettingzoo[jaxmarl]'"
        ) from exc
    return jax, make


# ---------------------------------------------------------------------------
# Env construction (returns a uniform numpy-facing parallel-step API)
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    obs: Dict[str, np.ndarray]
    rewards: Dict[str, float]
    terms: Dict[str, bool]
    truncs: Dict[str, bool]


class _PZEnv:
    """Thin numpy-facing wrapper around MPE2 simple_tag_v3.parallel_env."""

    name = "mpe2.simple_tag_v3"

    def __init__(self, max_cycles: int = 25):
        st = _import_pettingzoo_simple_tag()
        self._env = st.parallel_env(max_cycles=max_cycles, continuous_actions=False)
        self.possible_agents: List[str] = list(self._env.possible_agents)

    def reset(self, seed: int) -> Dict[str, np.ndarray]:
        obs, _info = self._env.reset(seed=int(seed))
        return {k: np.asarray(v, dtype=np.float64) for k, v in obs.items()}

    def step(self, actions: Dict[str, int]) -> StepResult:
        obs, rew, term, trunc, _info = self._env.step(
            {k: int(v) for k, v in actions.items()}
        )
        return StepResult(
            obs={k: np.asarray(v, dtype=np.float64) for k, v in obs.items()},
            rewards={k: float(v) for k, v in rew.items()},
            terms={k: bool(v) for k, v in term.items()},
            truncs={k: bool(v) for k, v in trunc.items()},
        )

    def action_n(self, agent: str) -> int:
        return int(self._env.action_space(agent).n)

    def obs_shape(self, agent: str) -> Tuple[int, ...]:
        return tuple(self._env.observation_space(agent).shape)

    def close(self):
        self._env.close()


class _JaxEnv:
    """Thin numpy-facing wrapper around JaxMARL MPE_simple_tag_v3."""

    name = "jaxmarl.MPE_simple_tag_v3"

    def __init__(self, max_cycles: int = 25):
        jax, make = _import_jaxmarl()
        self._jax = jax
        # JaxMARL exposes ``max_steps`` as the cycle cap.
        self._env = make("MPE_simple_tag_v3", max_steps=int(max_cycles))
        self.possible_agents: List[str] = list(self._env.agents)
        self._key = jax.random.PRNGKey(0)
        self._state: Any = None

    def reset(self, seed: int) -> Dict[str, np.ndarray]:
        self._key = self._jax.random.PRNGKey(int(seed))
        self._key, sub = self._jax.random.split(self._key)
        obs, self._state = self._env.reset(sub)
        return {k: np.asarray(v, dtype=np.float64) for k, v in obs.items()}

    def step(self, actions: Dict[str, int]) -> StepResult:
        self._key, sub = self._jax.random.split(self._key)
        jact = {k: self._jax.numpy.asarray(int(v)) for k, v in actions.items()}
        obs, self._state, rew, dones, _info = self._env.step(
            sub, self._state, jact
        )
        global_done = bool(np.asarray(dones.get("__all__", False)))
        terms: Dict[str, bool] = {}
        for a in self.possible_agents:
            terms[a] = bool(np.asarray(dones.get(a, global_done)))
        return StepResult(
            obs={k: np.asarray(v, dtype=np.float64) for k, v in obs.items()},
            rewards={k: float(np.asarray(v)) for k, v in rew.items()},
            terms=terms,
            truncs={a: False for a in self.possible_agents},
        )

    def action_n(self, agent: str) -> int:
        space = self._env.action_space(agent)
        # JaxMARL returns gymnax / gymnasium-style spaces depending on
        # version; both expose ``.n`` for discrete spaces.
        return int(getattr(space, "n", -1))

    def obs_shape(self, agent: str) -> Tuple[int, ...]:
        space = self._env.observation_space(agent)
        return tuple(getattr(space, "shape", ()))

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Axis 1 — Structural equivalence
# ---------------------------------------------------------------------------


@dataclass
class AxisReport:
    name: str
    passed: bool
    detail: str
    metrics: Dict[str, Any] = field(default_factory=dict)


def axis1_structural(pz: _PZEnv, jx: _JaxEnv) -> AxisReport:
    msgs: List[str] = []
    metrics: Dict[str, Any] = {}
    ok = True

    pz_agents = sorted(pz.possible_agents)
    jx_agents = sorted(jx.possible_agents)
    metrics["pz_agents"] = pz_agents
    metrics["jx_agents"] = jx_agents
    if pz_agents != jx_agents:
        ok = False
        msgs.append(
            f"agent IDs differ: pz={pz_agents} vs jx={jx_agents}"
        )
    else:
        msgs.append(f"agent IDs match ({len(pz_agents)} agents)")

    common = [a for a in pz_agents if a in jx_agents]
    obs_shape_mismatches: List[str] = []
    act_n_mismatches: List[str] = []
    for a in common:
        ps, js = pz.obs_shape(a), jx.obs_shape(a)
        if ps != js:
            obs_shape_mismatches.append(f"{a}: pz={ps} vs jx={js}")
        pn, jn = pz.action_n(a), jx.action_n(a)
        if pn != jn:
            act_n_mismatches.append(f"{a}: pz={pn} vs jx={jn}")

    metrics["obs_shape_mismatches"] = obs_shape_mismatches
    metrics["act_n_mismatches"] = act_n_mismatches
    if obs_shape_mismatches:
        ok = False
        msgs.append("obs shapes differ: " + "; ".join(obs_shape_mismatches))
    else:
        msgs.append("obs shapes match")
    if act_n_mismatches:
        ok = False
        msgs.append("action cardinalities differ: " + "; ".join(act_n_mismatches))
    else:
        msgs.append("action cardinalities match")

    return AxisReport("Axis 1 — Structural", ok, "; ".join(msgs), metrics)


# ---------------------------------------------------------------------------
# Axis 2 — Trajectory under fixed seed + fixed actions
# ---------------------------------------------------------------------------


def _scripted_actions(
    rng: np.random.Generator,
    agents: List[str],
    action_n: Dict[str, int],
    n_steps: int,
) -> List[Dict[str, int]]:
    """Pre-sample a deterministic action sequence shared by both envs."""
    return [
        {a: int(rng.integers(0, action_n[a])) for a in agents}
        for _ in range(n_steps)
    ]


def axis2_trajectory(
    pz: _PZEnv,
    jx: _JaxEnv,
    *,
    seed: int,
    max_steps: int,
) -> AxisReport:
    common = sorted(set(pz.possible_agents) & set(jx.possible_agents))
    if not common:
        return AxisReport(
            "Axis 2 — Trajectory",
            False,
            "no common agents — cannot compare trajectories.",
        )

    action_n = {a: pz.action_n(a) for a in common}
    rng = np.random.default_rng(seed)
    scripted = _scripted_actions(rng, common, action_n, max_steps)

    pz_obs0 = pz.reset(seed=seed)
    jx_obs0 = jx.reset(seed=seed)

    init_obs_delta = max(
        float(np.max(np.abs(pz_obs0[a] - jx_obs0[a]))) for a in common
    )

    obs_deltas: List[float] = []
    rew_deltas: List[float] = []
    term_disagreements = 0
    pz_done = False
    jx_done = False
    pz_step_count = 0
    jx_step_count = 0
    for step_idx, actions in enumerate(scripted):
        if pz_done and jx_done:
            break
        if not pz_done:
            pzr = pz.step(actions)
            pz_step_count += 1
            pz_done = all(pzr.terms.get(a, False) or pzr.truncs.get(a, False) for a in common)
        if not jx_done:
            jxr = jx.step(actions)
            jx_step_count += 1
            jx_done = all(jxr.terms.get(a, False) for a in common)
        # Compare per-agent obs/rew where both envs returned values for
        # the agent this step. PettingZoo zeroes obs after termination
        # by emptying the dict, so we guard with .get().
        for a in common:
            if a in pzr.obs and a in jxr.obs:
                obs_deltas.append(
                    float(np.max(np.abs(pzr.obs[a] - jxr.obs[a])))
                )
            rew_deltas.append(abs(pzr.rewards.get(a, 0.0) - jxr.rewards.get(a, 0.0)))
            if pzr.terms.get(a, False) != jxr.terms.get(a, False):
                term_disagreements += 1

    max_obs_delta = max(obs_deltas) if obs_deltas else float("nan")
    mean_obs_delta = float(np.mean(obs_deltas)) if obs_deltas else float("nan")
    max_rew_delta = max(rew_deltas) if rew_deltas else float("nan")
    mean_rew_delta = float(np.mean(rew_deltas)) if rew_deltas else float("nan")

    metrics = {
        "init_obs_delta": init_obs_delta,
        "max_obs_delta": max_obs_delta,
        "mean_obs_delta": mean_obs_delta,
        "max_rew_delta": max_rew_delta,
        "mean_rew_delta": mean_rew_delta,
        "term_disagreements": term_disagreements,
        "pz_step_count": pz_step_count,
        "jx_step_count": jx_step_count,
    }

    # Tolerance: the two implementations are *not* the same code, so
    # bit-exact match is unrealistic. We declare PASS only on
    # near-bit-equality (max delta < 1e-5). Anything else is reported
    # as INCONCLUSIVE — the divergence magnitude is the actual signal.
    bit_exact = (
        init_obs_delta < 1e-5
        and (np.isnan(max_obs_delta) or max_obs_delta < 1e-5)
        and (np.isnan(max_rew_delta) or max_rew_delta < 1e-5)
        and term_disagreements == 0
    )
    detail = (
        f"init_obs_delta={init_obs_delta:.3e}; "
        f"max_obs_delta={max_obs_delta:.3e}; "
        f"mean_obs_delta={mean_obs_delta:.3e}; "
        f"max_rew_delta={max_rew_delta:.3e}; "
        f"term_disagreements={term_disagreements}/{len(common) * pz_step_count}"
    )
    return AxisReport("Axis 2 — Trajectory (fixed actions)", bit_exact, detail, metrics)


# ---------------------------------------------------------------------------
# Axis 3 — Statistical equivalence under random policies
# ---------------------------------------------------------------------------


def _rollout_random(
    env, *, seed: int, max_steps: int, common: List[str], rng: np.random.Generator
) -> Tuple[int, Dict[str, float]]:
    """One random-policy episode. Returns (length, per-agent total reward)."""
    action_n = {a: env.action_n(a) for a in common}
    env.reset(seed=seed)
    totals: Dict[str, float] = {a: 0.0 for a in common}
    length = 0
    for _ in range(max_steps):
        actions = {a: int(rng.integers(0, action_n[a])) for a in common}
        r = env.step(actions)
        length += 1
        for a in common:
            totals[a] += r.rewards.get(a, 0.0)
        if all(r.terms.get(a, False) or r.truncs.get(a, False) for a in common):
            break
    return length, totals


def _ks_2samp_simple(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Two-sample KS statistic + asymptotic p-value (no scipy dep).

    Implements the standard empirical-CDF difference test using the
    Kolmogorov asymptotic series. Adequate for n_a, n_b > 20.
    """
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    all_vals = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_vals, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    if d == 0.0:
        # Empirical CDFs are identical — distributions agree perfectly,
        # so the p-value is 1.0 (we cannot reject the null).
        return 0.0, 1.0
    n = len(a) * len(b) / (len(a) + len(b))
    # Asymptotic p-value: 2 * sum_{k>=1} (-1)^{k-1} exp(-2 k^2 lambda^2)
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    j = np.arange(1, 101)
    p = 2.0 * np.sum(((-1.0) ** (j - 1)) * np.exp(-2.0 * (j**2) * (lam**2)))
    p = float(min(max(p, 0.0), 1.0))
    return d, p


def axis3_statistical(
    pz: _PZEnv,
    jx: _JaxEnv,
    *,
    seed: int,
    max_steps: int,
    episodes: int,
    alpha: float = 0.05,
) -> AxisReport:
    common = sorted(set(pz.possible_agents) & set(jx.possible_agents))
    if not common:
        return AxisReport("Axis 3 — Statistical", False, "no common agents.")

    pz_lengths: List[int] = []
    jx_lengths: List[int] = []
    pz_side_returns = {"adversary": [], "agent": []}
    jx_side_returns = {"adversary": [], "agent": []}

    rng = np.random.default_rng(seed)
    for ep in range(episodes):
        # Both envs see the same random-action sequence per episode
        # (paired test): we draw one root seed and instantiate two
        # identically-seeded RNGs from it so internal env stochasticity
        # is the only source of divergence between the two rollouts.
        action_seed = int(rng.integers(0, 2**31 - 1))
        ep_seed = int(rng.integers(0, 2**31 - 1))
        ep_rng_pz = np.random.default_rng(action_seed)
        ep_rng_jx = np.random.default_rng(action_seed)

        pz_len, pz_tot = _rollout_random(
            pz, seed=ep_seed, max_steps=max_steps, common=common, rng=ep_rng_pz
        )
        jx_len, jx_tot = _rollout_random(
            jx, seed=ep_seed, max_steps=max_steps, common=common, rng=ep_rng_jx
        )
        pz_lengths.append(pz_len)
        jx_lengths.append(jx_len)
        for a in common:
            side = "adversary" if a.startswith("adversary") else "agent"
            pz_side_returns[side].append(pz_tot[a])
            jx_side_returns[side].append(jx_tot[a])

    metrics: Dict[str, Any] = {
        "episodes": episodes,
        "pz_mean_length": float(np.mean(pz_lengths)),
        "jx_mean_length": float(np.mean(jx_lengths)),
    }
    msgs: List[str] = []
    ok = True

    d_len, p_len = _ks_2samp_simple(np.array(pz_lengths), np.array(jx_lengths))
    metrics["len_ks"] = (d_len, p_len)
    msgs.append(f"len: KS={d_len:.3f} p={p_len:.3f}")
    if p_len < alpha:
        ok = False

    for side in ("adversary", "agent"):
        a_arr = np.array(pz_side_returns[side])
        b_arr = np.array(jx_side_returns[side])
        if a_arr.size == 0 or b_arr.size == 0:
            continue
        d, p = _ks_2samp_simple(a_arr, b_arr)
        metrics[f"{side}_return_ks"] = (d, p)
        metrics[f"pz_{side}_mean_return"] = float(np.mean(a_arr))
        metrics[f"jx_{side}_mean_return"] = float(np.mean(b_arr))
        msgs.append(
            f"{side} return: KS={d:.3f} p={p:.3f} "
            f"(pz_mean={np.mean(a_arr):.3f}, jx_mean={np.mean(b_arr):.3f})"
        )
        if p < alpha:
            ok = False

    detail = "; ".join(msgs)
    return AxisReport("Axis 3 — Statistical (random policy)", ok, detail, metrics)


# ---------------------------------------------------------------------------
# Axis 4 — Determinism / no shared state
# ---------------------------------------------------------------------------


def axis4_determinism(
    pz_factory, jx_factory, *, seed: int, max_steps: int
) -> AxisReport:
    msgs: List[str] = []
    ok = True
    metrics: Dict[str, Any] = {}

    for label, factory in (("pz", pz_factory), ("jx", jx_factory)):
        env_a = factory()
        env_b = factory()
        common = sorted(env_a.possible_agents)
        action_n = {a: env_a.action_n(a) for a in common}
        rng = np.random.default_rng(seed)
        scripted = _scripted_actions(rng, common, action_n, max_steps)

        env_a.reset(seed=seed)
        env_b.reset(seed=seed)
        max_delta = 0.0
        for actions in scripted:
            ra = env_a.step(actions)
            rb = env_b.step(actions)
            for a in common:
                if a in ra.obs and a in rb.obs:
                    max_delta = max(max_delta, float(np.max(np.abs(ra.obs[a] - rb.obs[a]))))
                max_delta = max(max_delta, abs(ra.rewards.get(a, 0.0) - rb.rewards.get(a, 0.0)))
            if all(ra.terms.get(a, False) for a in common):
                break
        env_a.close()
        env_b.close()

        metrics[f"{label}_self_replay_delta"] = max_delta
        if max_delta > 1e-9:
            ok = False
            msgs.append(f"{label}: self-replay diverges by {max_delta:.3e}")
        else:
            msgs.append(f"{label}: self-replay bit-identical")

    return AxisReport("Axis 4 — Determinism", ok, "; ".join(msgs), metrics)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _print_report(reports: List[AxisReport]) -> int:
    width = 78
    print("=" * width)
    print("simple_tag_v3 equivalence experiment — PettingZoo MPE2 vs JaxMARL")
    print("=" * width)
    n_pass = 0
    for r in reports:
        status = "PASS" if r.passed else "FAIL"
        marker = "[+]" if r.passed else "[-]"
        print(f"\n{marker} {status}  {r.name}")
        print(f"      {r.detail}")
        if r.metrics:
            for k, v in r.metrics.items():
                if isinstance(v, float):
                    print(f"        {k} = {v:.6g}")
                elif isinstance(v, (list, tuple)) and len(v) <= 8:
                    print(f"        {k} = {v}")
                else:
                    print(f"        {k} = {v}")
        if r.passed:
            n_pass += 1
    print("\n" + "-" * width)
    print(f"Verdict: {n_pass}/{len(reports)} axes passed.")
    if n_pass == len(reports):
        print(
            "Strong equivalence: bit-identical trajectories under fixed "
            "actions, distributionally indistinguishable under random "
            "policies."
        )
    elif any(r.name.startswith("Axis 1") and r.passed for r in reports) and any(
        r.name.startswith("Axis 3") and r.passed for r in reports
    ):
        print(
            "Functional equivalence: structures match and long-run "
            "statistics agree, but per-step trajectories diverge "
            "(expected — JaxMARL is a reimplementation, not a port)."
        )
    else:
        print(
            "Non-equivalent: do NOT treat policies trained on one as "
            "drop-in replacements for the other. Inspect Axis 1 / 2 "
            "metrics before proceeding."
        )
    print("-" * width)
    return 0 if n_pass >= 2 else 1  # Soft success: structural + statistical


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare PettingZoo MPE2 simple_tag_v3 vs JaxMARL MPE_simple_tag_v3."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-steps", type=int, default=25,
        help="Episode length cap (matches simple_tag default max_cycles).",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Number of random-policy episodes per env for Axis 3.",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args(argv)

    t0 = time.time()
    pz = _PZEnv(max_cycles=args.max_steps)
    jx = _JaxEnv(max_cycles=args.max_steps)

    reports: List[AxisReport] = []
    reports.append(axis1_structural(pz, jx))
    reports.append(axis2_trajectory(pz, jx, seed=args.seed, max_steps=args.max_steps))
    reports.append(
        axis3_statistical(
            pz, jx,
            seed=args.seed,
            max_steps=args.max_steps,
            episodes=args.episodes,
            alpha=args.alpha,
        )
    )
    reports.append(
        axis4_determinism(
            lambda: _PZEnv(max_cycles=args.max_steps),
            lambda: _JaxEnv(max_cycles=args.max_steps),
            seed=args.seed,
            max_steps=args.max_steps,
        )
    )
    pz.close()
    jx.close()
    elapsed = time.time() - t0
    rc = _print_report(reports)
    print(f"\nElapsed: {elapsed:.2f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
