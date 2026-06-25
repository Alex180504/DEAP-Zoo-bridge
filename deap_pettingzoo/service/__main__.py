"""Entry-point: ``python -m deap_pettingzoo.service``."""

from __future__ import annotations

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="DEAP-bridge Experiment Service (FastAPI sidecar)."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8100, help="Bind port.")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Root directory for results / database (env: DEAP_RESULTS_ROOT).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity: -v logs normal operation (INFO), "
        "-vv adds debug detail (env: DEAP_LOG_LEVEL).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["debug", "info", "warning", "error"],
        help="Explicit log level (overrides --verbose; env: DEAP_LOG_LEVEL).",
    )
    parser.add_argument(
        "--log-format",
        default=None,
        choices=["plain", "json"],
        help="Log output format (env: DEAP_LOG_FORMAT; default plain).",
    )
    args = parser.parse_args(argv)

    from deap_pettingzoo.service import run

    run(
        host=args.host,
        port=args.port,
        results_root=args.results_root,
        verbose=args.verbose,
        log_level=args.log_level,
        log_format=args.log_format,
    )


if __name__ == "__main__":
    main()
