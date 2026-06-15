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
    args = parser.parse_args(argv)

    from deap_pettingzoo.service import run

    run(host=args.host, port=args.port, results_root=args.results_root)


if __name__ == "__main__":
    main()
