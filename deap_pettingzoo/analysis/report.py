"""Programmatic experiment-report builder.

Generates a self-contained Markdown (default) or HTML summary of an
:class:`~deap_pettingzoo.analysis.loaders.Experiment`. Optional figures
(performance profile, learning curves, pairwise PI matrix) are saved
alongside the report and referenced via relative paths.

The report layout is fixed but data-driven; for custom layouts callers
can use :mod:`aggregate`, :mod:`compare`, and :mod:`plots` directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from deap_pettingzoo.analysis.aggregate import aggregate_summary
from deap_pettingzoo.analysis.compare import pairwise_table
from deap_pettingzoo.analysis.loaders import Experiment


def build_report(
    experiment: Experiment,
    output_dir: str | Path,
    *,
    metric: str = "final_fitness",
    fmt: str = "md",
    include_figures: bool = True,
    title: Optional[str] = None,
) -> Path:
    """Render a self-contained report under ``output_dir``.

    Parameters
    ----------
    experiment:
        Loaded experiment.
    output_dir:
        Directory to write the report and any figures into. Created if
        missing.
    metric:
        Final-score metric to aggregate over (default ``final_fitness``).
    fmt:
        ``"md"`` (default) or ``"html"``.
    include_figures:
        When ``True``, generate matplotlib figures alongside the report.

    Returns
    -------
    Path
        Path to the rendered report file.
    """
    if fmt not in {"md", "html"}:
        raise ValueError(f"fmt must be 'md' or 'html', got {fmt!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = experiment.summary_table()
    score_groups = experiment.final_scores(metric=metric)
    aggregate = aggregate_summary(score_groups)
    pairwise = pairwise_table(experiment, metric=metric, test="mann_whitney")

    figures: list = []
    if include_figures and score_groups:
        figures.extend(_render_figures(experiment, score_groups, output_dir))

    multitask: Optional[dict] = None
    if len(experiment.by_env()) >= 2:
        multitask = _build_multitask_section(
            experiment,
            output_dir,
            metric=metric,
            include_figures=include_figures,
        )

    if fmt == "md":
        return _write_markdown(
            output_dir,
            title=title or "Experiment report",
            n_runs=len(experiment),
            summary=summary,
            aggregate=aggregate,
            pairwise=pairwise,
            figures=figures,
            metric=metric,
            multitask=multitask,
        )
    return _write_html(
        output_dir,
        title=title or "Experiment report",
        n_runs=len(experiment),
        summary=summary,
        aggregate=aggregate,
        pairwise=pairwise,
        figures=figures,
        metric=metric,
        multitask=multitask,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_figures(experiment: Experiment, score_groups, output_dir: Path) -> list:
    """Render and save the standard figure set; return ``[(label, path), ...]``."""
    from deap_pettingzoo.analysis import plots  # noqa: WPS433

    out: list = []
    try:
        fig = plots.plot_eval_distribution(
            score_groups, title="Final scores per algorithm"
        )
        path = output_dir / "fig_eval_distribution.png"
        fig.savefig(path, dpi=140)
        out.append(("Final-score distribution", path.name))
    except Exception:  # pragma: no cover - matplotlib missing
        pass

    # Learning curves (mean ± bootstrap CI) per algorithm
    try:
        from deap_pettingzoo.analysis.aggregate import (  # noqa: WPS433
            learning_curve_with_ci,
        )

        curves = {}
        for algo, runs in experiment.by_algorithm().items():
            try:
                curves[algo] = learning_curve_with_ci(runs)
            except (ValueError, KeyError):
                continue
        if curves:
            fig = plots.plot_learning_curves(
                curves, title="Learning curves (bootstrap 95% CI)"
            )
            path = output_dir / "fig_learning_curves.png"
            fig.savefig(path, dpi=140)
            out.append(("Learning curves", path.name))
    except Exception:  # pragma: no cover
        pass
    return out


def _build_multitask_section(
    experiment: Experiment,
    output_dir: Path,
    *,
    metric: str,
    include_figures: bool,
) -> Optional[dict]:
    """Compute multi-task aggregate frame + (optional) figures.

    Returns ``None`` when fewer than two environments yield non-empty
    score columns. Always uses auto min-max normalisation.
    """
    from deap_pettingzoo.analysis.multi_task import collect_score_matrices  # noqa: WPS433
    from deap_pettingzoo.analysis.normalize import (  # noqa: WPS433
        NormalizationSpec,
        normalize_score_matrices,
    )

    matrices, env_names = collect_score_matrices(experiment, metric=metric)
    if len(env_names) < 2 or not matrices:
        return None

    spec = NormalizationSpec.from_minmax(matrices, env_names)
    normed = normalize_score_matrices(matrices, env_names, spec)

    forest = None
    try:
        from deap_pettingzoo.analysis.rliable_bridge import (  # noqa: WPS433
            aggregate_metrics_forest,
        )
        forest = aggregate_metrics_forest(normed, reps=2000)
    except ImportError:
        forest = None  # rliable not installed → table omitted

    figures: list = []
    if include_figures:
        try:
            from deap_pettingzoo.analysis import plots  # noqa: WPS433
            if forest is not None and len(forest) > 0:
                fig = plots.plot_aggregate_metrics_forest(
                    forest, title="Aggregate metrics across tasks (95% CI)"
                )
                p = output_dir / "fig_multitask_forest.png"
                fig.savefig(p, dpi=140)
                figures.append(("Aggregate-metrics forest", p.name))
            fig = plots.plot_per_env_eval(
                matrices, env_names, title="Per-env score distributions (raw)"
            )
            p = output_dir / "fig_multitask_per_env.png"
            fig.savefig(p, dpi=140)
            figures.append(("Per-env distributions", p.name))
        except Exception:  # pragma: no cover - matplotlib missing
            pass

    return {
        "envs": list(env_names),
        "n_algos": len(matrices),
        "spec_low": dict(spec.low),
        "spec_high": dict(spec.high),
        "forest": forest,
        "figures": figures,
    }


def _write_markdown(
    output_dir: Path,
    *,
    title: str,
    n_runs: int,
    summary,
    aggregate,
    pairwise,
    figures,
    metric: str,
    multitask: Optional[dict] = None,
) -> Path:
    lines: list = [f"# {title}", ""]
    lines.append(f"- **Runs analysed:** {n_runs}")
    lines.append(f"- **Score metric:** `{metric}`")
    lines.append("")

    lines.append("## Per-run summary")
    lines.append("")
    lines.append(_df_to_md(summary))
    lines.append("")

    lines.append("## Aggregate (mean / median / IQM with 95% bootstrap CI)")
    lines.append("")
    lines.append(_df_to_md(aggregate.reset_index() if len(aggregate) else aggregate))
    lines.append("")

    lines.append("## Pairwise comparison (Mann-Whitney U + Cliff's delta)")
    lines.append("")
    lines.append(_df_to_md(pairwise))
    lines.append("")

    if multitask is not None:
        lines.append("## Multi-task aggregate")
        lines.append("")
        lines.append(
            f"- **Environments ({len(multitask['envs'])}):** "
            + ", ".join(f"`{e}`" for e in multitask["envs"])
        )
        lines.append(f"- **Algorithms compared:** {multitask['n_algos']}")
        lines.append("- **Normalisation:** auto min-max per task")
        lines.append("")
        if multitask["forest"] is not None and len(multitask["forest"]) > 0:
            lines.append("### Aggregate metrics on normalised scores")
            lines.append("")
            lines.append(_df_to_md(multitask["forest"]))
            lines.append("")
        for label, fname in multitask["figures"]:
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"![{label}]({fname})")
            lines.append("")

    if figures:
        lines.append("## Figures")
        lines.append("")
        for label, fname in figures:
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"![{label}]({fname})")
            lines.append("")

    path = output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_html(
    output_dir: Path,
    *,
    title: str,
    n_runs: int,
    summary,
    aggregate,
    pairwise,
    figures,
    metric: str,
    multitask: Optional[dict] = None,
) -> Path:
    summary_html = _df_to_html(summary)
    aggregate_html = _df_to_html(
        aggregate.reset_index() if len(aggregate) else aggregate
    )
    pairwise_html = _df_to_html(pairwise)
    fig_html_parts: list = []
    for label, fname in figures:
        fig_html_parts.append(
            f'<figure><img src="{fname}" alt="{label}" />'
            f"<figcaption>{label}</figcaption></figure>"
        )

    multitask_html = ""
    if multitask is not None:
        envs_str = ", ".join(f"<code>{e}</code>" for e in multitask["envs"])
        forest_html = (
            _df_to_html(multitask["forest"])
            if multitask["forest"] is not None and len(multitask["forest"]) > 0
            else ""
        )
        mt_figs = "".join(
            f'<figure><img src="{fn}" alt="{lbl}" />'
            f"<figcaption>{lbl}</figcaption></figure>"
            for lbl, fn in multitask["figures"]
        )
        multitask_html = (
            "<h2>Multi-task aggregate</h2>"
            f"<p><strong>Environments ({len(multitask['envs'])}):</strong> {envs_str}<br/>"
            f"<strong>Algorithms compared:</strong> {multitask['n_algos']}<br/>"
            "<strong>Normalisation:</strong> auto min-max per task</p>"
            + (
                "<h3>Aggregate metrics on normalised scores</h3>" + forest_html
                if forest_html
                else ""
            )
            + mt_figs
        )

    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>
:root {{
    color-scheme: light dark;
    --bg: #ffffff;
    --fg: #1a1a1a;
    --muted: #666666;
    --border: #cccccc;
    --th-bg: #f4f4f4;
}}
@media (prefers-color-scheme: dark) {{
    :root {{
        --bg: #0e1117;
        --fg: #e6e6e6;
        --muted: #a0a0a0;
        --border: #3a3a3a;
        --th-bg: #1c2129;
    }}
}}
body {{
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    margin: 2rem; max-width: 980px;
    background: var(--bg); color: var(--fg);
}}
table {{ border-collapse: collapse; margin: 1rem 0; color: var(--fg); }}
th, td {{
    border: 1px solid var(--border);
    padding: 0.3rem 0.6rem; font-size: 0.92rem;
}}
th {{ background: var(--th-bg); }}
tr:nth-child(even) td {{ background: rgba(127,127,127,0.06); }}
a {{ color: #4ea1ff; }}
code {{
    background: rgba(127,127,127,0.15); padding: 0 0.25rem; border-radius: 3px;
}}
figure {{ margin: 1.5rem 0; }}
figure img {{ max-width: 100%; background: #ffffff; border-radius: 4px; }}
figcaption {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.3rem; }}
</style></head><body>
<h1>{title}</h1>
<p><strong>Runs analysed:</strong> {n_runs} &middot;
<strong>Score metric:</strong> <code>{metric}</code></p>
<h2>Per-run summary</h2>
{summary_html}
<h2>Aggregate (mean / median / IQM with 95% bootstrap CI)</h2>
{aggregate_html}
<h2>Pairwise comparison (Mann-Whitney U + Cliff's delta)</h2>
{pairwise_html}
{multitask_html}
{"<h2>Figures</h2>" + "".join(fig_html_parts) if fig_html_parts else ""}
</body></html>
"""
    path = output_dir / "report.html"
    path.write_text(body, encoding="utf-8")
    return path


def _df_to_md(df: Any) -> str:
    if df is None or len(df) == 0:
        return "_(empty)_"
    try:
        return df.to_markdown(index=False)
    except (ImportError, AttributeError):
        # tabulate not installed -> simple fallback
        cols = list(df.columns)
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = [
            "| " + " | ".join(_fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)
        ]
        return "\n".join([header, sep, *rows])


def _df_to_html(df: Any) -> str:
    if df is None or len(df) == 0:
        return "<p><em>(empty)</em></p>"
    try:
        return df.to_html(index=False, float_format=lambda v: f"{v:.4g}")
    except Exception:  # pragma: no cover
        return f"<pre>{df}</pre>"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
