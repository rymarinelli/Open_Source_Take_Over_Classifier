import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .evaluate_classifier import _auc, _metrics, _sigmoid, _train_logistic_regression
from .utils import read_csv, read_jsonl


def _prepare_dataset(risk_scores_path: str, known_takeovers_path: str) -> List[Tuple[str, float, int]]:
    known_rows = read_csv(known_takeovers_path)
    known_packages = {f"npm:{row['package']}" for row in known_rows if row.get("package")}

    data = []
    for row in read_jsonl(risk_scores_path):
        package_id = row.get("package_id")
        if not package_id:
            continue
        score = float(row.get("risk_score", 0.0))
        label = 1 if package_id in known_packages else 0
        data.append((package_id, score, label))
    return data


def _balance_dataset(data: List[Tuple[str, float, int]], seed: int) -> List[Tuple[str, float, int]]:
    positives = [row for row in data if row[2] == 1]
    negatives = [row for row in data if row[2] == 0]
    if not positives or not negatives:
        return data
    rng = np.random.default_rng(seed)
    sample_size = min(len(positives), len(negatives))
    negatives_sample = rng.choice(len(negatives), size=sample_size, replace=False)
    balanced = positives + [negatives[idx] for idx in negatives_sample]
    rng.shuffle(balanced)
    return balanced


def _plot_histogram(scores: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(scores[labels == 0], bins=20, alpha=0.6, label="Normal", color="#4C78A8")
    plt.hist(scores[labels == 1], bins=20, alpha=0.6, label="Takeover", color="#F58518")
    plt.xlabel("Risk score")
    plt.ylabel("Count")
    plt.title("Risk score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_roc(scores: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    thresholds = np.linspace(0.0, 1.0, 101)
    tpr = []
    fpr = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int32)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        tpr.append(tp / (tp + fn) if tp + fn else 0.0)
        fpr.append(fp / (fp + tn) if fp + tn else 0.0)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_benchmark(
    risk_scores_path: str,
    known_takeovers_path: str,
    output_dir: str,
    train_ratio: float,
    seed: int,
    plots: bool,
) -> None:
    data = _prepare_dataset(risk_scores_path, known_takeovers_path)
    if not data:
        raise ValueError("No labeled data available for benchmark")

    balanced = _balance_dataset(data, seed)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(balanced))
    split = int(len(balanced) * train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]

    scores = np.array([balanced[i][1] for i in train_idx], dtype=np.float32)
    labels = np.array([balanced[i][2] for i in train_idx], dtype=np.float32)

    weight, bias = _train_logistic_regression(scores, labels)

    test_scores = np.array([balanced[i][1] for i in test_idx], dtype=np.float32)
    test_labels = np.array([balanced[i][2] for i in test_idx], dtype=np.int32)
    probs = _sigmoid(weight * test_scores + bias)
    preds = (probs >= 0.5).astype(np.int32)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = {
        "total_samples": len(balanced),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "weight": float(weight),
        "bias": float(bias),
        "auc": float(_auc(test_scores, test_labels)),
        "metrics": _metrics(preds, test_labels),
    }

    report_path = output_path / "benchmark_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    if plots:
        histogram_path = output_path / "risk_score_histogram.png"
        roc_path = output_path / "roc_curve.png"
        _plot_histogram(test_scores, test_labels, histogram_path)
        _plot_roc(probs, test_labels, roc_path)
        report["plots"] = {
            "histogram": str(histogram_path),
            "roc_curve": str(roc_path),
        }
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark risk-score classifier")
    parser.add_argument("--risk-scores", default="data/risk_scores.jsonl")
    parser.add_argument("--known-takeovers", default="data/known_takeovers.csv")
    parser.add_argument("--output-dir", default="data/benchmark")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_benchmark(
        args.risk_scores,
        args.known_takeovers,
        args.output_dir,
        args.train_ratio,
        args.seed,
        plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()