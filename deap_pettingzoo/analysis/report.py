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
