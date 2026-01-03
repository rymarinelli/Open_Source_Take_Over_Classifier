import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .utils import read_jsonl


def _parse_time(value: str) -> np.datetime64:
    if not value:
        return np.datetime64("1970-01-01")
    return np.datetime64(value.replace("Z", ""))


def _load_embeddings(path: str) -> Dict[str, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to read parquet embeddings") from exc
    df = pd.read_parquet(path)
    return {row["semantic_id"]: np.array(row["vector"], dtype=np.float32) for _, row in df.iterrows()}


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return 1.0 - float(np.dot(a, b) / denom)


def _mean_vector(vectors: List[np.ndarray]) -> np.ndarray:
    if not vectors:
        return np.zeros(1, dtype=np.float32)
    stacked = np.vstack(vectors)
    return stacked.mean(axis=0)


def score_risk(
    semantic_units_path: str,
    shifts_path: str,
    embeddings_path: str,
    output_path: str,
    alpha: float,
) -> None:
    embeddings = _load_embeddings(embeddings_path)
    units_by_repo: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for unit in read_jsonl(semantic_units_path):
        units_by_repo[unit["repo_url"]].append(unit)

    shifts_by_repo: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for shift in read_jsonl(shifts_path):
        shifts_by_repo[shift["repo_url"]].append(shift)

    scores = []
    for repo_url, shifts in shifts_by_repo.items():
        units = units_by_repo.get(repo_url, [])
        if not units:
            continue
        package_ids = sorted({unit.get("package_id") for unit in units if unit.get("package_id")})
        package_id = package_ids[0] if package_ids else ""
        units.sort(key=lambda item: item.get("created_time") or "")
        for shift in shifts:
            window_start = _parse_time(shift["window_start"])
            pre_units = [u for u in units if _parse_time(u.get("created_time") or "") < window_start]
            post_units = [u for u in units if _parse_time(u.get("created_time") or "") >= window_start]

            pr_pre = [embeddings.get(u["semantic_id"]) for u in pre_units if u.get("unit_type", "").startswith("pr_")]
            pr_post = [embeddings.get(u["semantic_id"]) for u in post_units if u.get("unit_type", "").startswith("pr_")]
            pr_pre = [vec for vec in pr_pre if vec is not None]
            pr_post = [vec for vec in pr_post if vec is not None]

            mu_pre = _mean_vector(pr_pre) if pr_pre else None
            mu_post = _mean_vector(pr_post) if pr_post else None
            d_pr = _cosine_distance(mu_pre, mu_post)

            author_pre: Dict[str, List[np.ndarray]] = defaultdict(list)
            author_post: Dict[str, List[np.ndarray]] = defaultdict(list)
            for unit in pre_units:
                vec = embeddings.get(unit["semantic_id"])
                if vec is not None:
                    author_pre[unit.get("author") or "unknown"].append(vec)
            for unit in post_units:
                vec = embeddings.get(unit["semantic_id"])
                if vec is not None:
                    author_post[unit.get("author") or "unknown"].append(vec)

            new_authors = [author for author in author_post if author not in author_pre]
            author_drifts = []
            for author in new_authors:
                pre_vec = _mean_vector(author_pre.get(author, []))
                post_vec = _mean_vector(author_post.get(author, []))
                author_drifts.append(_cosine_distance(pre_vec, post_vec))
            d_author = max(author_drifts) if author_drifts else 0.0

            risk = alpha * d_pr + (1 - alpha) * d_author
            scores.append(
                {
                    "repo_url": repo_url,
                    "package_id": package_id,
                    "risk_score": round(risk, 4),
                    "d_pr": round(d_pr, 4),
                    "d_author": round(d_author, 4),
                }
            )
    from .utils import write_jsonl

    write_jsonl(output_path, scores)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 7: drift & risk scoring")
    parser.add_argument("--semantic-units", default="data/semantic_units.jsonl")
    parser.add_argument("--shifts", default="data/governance_shifts.jsonl")
    parser.add_argument("--embeddings", default="data/embeddings.parquet")
    parser.add_argument("--output", default="data/risk_scores.jsonl")
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    score_risk(args.semantic_units, args.shifts, args.embeddings, args.output, args.alpha)


if __name__ == "__main__":
    main()