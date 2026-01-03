import argparse
import json
import math
from typing import Dict, List, Tuple

import numpy as np

from .utils import read_csv, read_jsonl


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _train_logistic_regression(
    scores: np.ndarray, labels: np.ndarray, lr: float = 0.1, epochs: int = 500
) -> Tuple[float, float]:
    weight = 0.0
    bias = 0.0
    for _ in range(epochs):
        logits = weight * scores + bias
        preds = _sigmoid(logits)
        error = preds - labels
        grad_w = np.mean(error * scores)
        grad_b = np.mean(error)
        weight -= lr * grad_w
        bias -= lr * grad_b
    return weight, bias


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    sorted_labels = labels[order]
    pos = np.sum(sorted_labels)
    neg = len(sorted_labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    cum_pos = np.cumsum(sorted_labels)
    cum_neg = np.cumsum(1 - sorted_labels)
    auc = np.sum(cum_pos * (1 - sorted_labels)) / (pos * neg)
    return float(auc)


def _metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / max(1, len(labels))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_classifier(
    risk_scores_path: str,
    known_takeovers_path: str,
    output_path: str,
    train_ratio: float,
) -> None:
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

    if not data:
        raise ValueError("No labeled data available for evaluation")

    rng = np.random.default_rng(0)
    indices = rng.permutation(len(data))
    split = int(len(data) * train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]

    scores = np.array([data[i][1] for i in train_idx], dtype=np.float32)
    labels = np.array([data[i][2] for i in train_idx], dtype=np.float32)

    weight, bias = _train_logistic_regression(scores, labels)

    test_scores = np.array([data[i][1] for i in test_idx], dtype=np.float32)
    test_labels = np.array([data[i][2] for i in test_idx], dtype=np.int32)
    probs = _sigmoid(weight * test_scores + bias)
    preds = (probs >= 0.5).astype(np.int32)

    report = {
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "weight": float(weight),
        "bias": float(bias),
        "auc": float(_auc(test_scores, test_labels)),
        "metrics": _metrics(preds, test_labels),
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate classifier on risk scores")
    parser.add_argument("--risk-scores", default="data/risk_scores.jsonl")
    parser.add_argument("--known-takeovers", default="data/known_takeovers.csv")
    parser.add_argument("--output", default="data/classifier_report.json")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    evaluate_classifier(args.risk_scores, args.known_takeovers, args.output, args.train_ratio)


if __name__ == "__main__":
    main()