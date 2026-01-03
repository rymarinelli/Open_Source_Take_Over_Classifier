import argparse

from .phase1_ecosystem import select_top_packages
from .phase2_resolve import resolve_repos
from .phase3_history import extract_semantic_units
from .phase4_governance import detect_governance_shifts
from .phase5_windowing import build_windows
from .phase6_embeddings import embed_semantic_units
from .evaluate_classifier import evaluate_classifier
from .benchmark import run_benchmark
from .phase7_risk import score_risk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open-source takeover detection pipeline")
    parser.add_argument("--pypi-csv")
    parser.add_argument("--npm-csv")
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--packages", default="data/packages.jsonl")
    parser.add_argument("--resolved", default="data/resolved_repos.jsonl")
    parser.add_argument("--semantic-units", default="data/semantic_units.jsonl")
    parser.add_argument("--shifts", default="data/governance_shifts.jsonl")
    parser.add_argument("--windows", default="data/windows.jsonl")
    parser.add_argument("--embeddings", default="data/embeddings.parquet")
    parser.add_argument("--scores", default="data/risk_scores.jsonl")
    parser.add_argument("--classifier-report", default="data/classifier_report.json")
    parser.add_argument("--known-takeovers", default="data/known_takeovers.csv")
    parser.add_argument("--benchmark-dir", default="data/benchmark")
    parser.add_argument("--repo-cache", default="data/repos")
    parser.add_argument("--include-prs", action="store_true")
    parser.add_argument("--max-prs", type=int, default=500)
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--max-windows", type=int, default=50)
    parser.add_argument("--max-commits", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--no-osv", action="store_true")
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    select_top_packages(args.pypi_csv, args.npm_csv, args.top_n, args.packages)
    resolve_repos(args.packages, args.resolved, use_osv=not args.no_osv)
    extract_semantic_units(
        args.resolved,
        args.semantic_units,
        args.repo_cache,
        include_prs=args.include_prs,
        max_prs=args.max_prs,
    )
    detect_governance_shifts(args.semantic_units, args.shifts, args.window_days)
    build_windows(
        args.shifts,
        args.resolved,
        args.repo_cache,
        args.windows,
        args.window_days,
        args.max_windows,
        args.max_commits,
    )
    embed_semantic_units(args.semantic_units, args.windows, args.embeddings, args.dimension)
    score_risk(args.semantic_units, args.shifts, args.embeddings, args.scores, args.alpha)
    if args.evaluate:
        evaluate_classifier(args.scores, args.known_takeovers, args.classifier_report, train_ratio=0.7)
    if args.benchmark:
        run_benchmark(
            args.scores,
            args.known_takeovers,
            args.benchmark_dir,
            train_ratio=0.7,
            seed=7,
            plots=True,
        )


if __name__ == "__main__":
    main()