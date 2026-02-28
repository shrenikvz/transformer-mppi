from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transformer-MPPI reproducible training and benchmarking pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reproduce = subparsers.add_parser("reproduce", help="Run training + benchmark pipeline")
    reproduce.add_argument("--task", choices=["navigation2d", "racing", "both"], default="both")
    reproduce.add_argument("--profile", choices=["quick", "paper"], default="quick")
    reproduce.add_argument("--output-dir", default="artifacts")
    reproduce.add_argument("--circuit-csv", default="circuit.csv")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "reproduce":
        from transformer_mppi.pipelines import run_reproduction

        output_paths = run_reproduction(
            task=args.task,
            profile=args.profile,
            output_dir=Path(args.output_dir),
            circuit_csv=Path(args.circuit_csv),
        )
        for path in output_paths:
            print(path)


if __name__ == "__main__":
    main()
