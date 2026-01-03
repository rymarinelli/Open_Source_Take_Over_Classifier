import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

from .utils import read_jsonl, write_jsonl


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _bucket_commits(units: Iterable[Dict[str, str]], window_days: int) -> Dict[str, Dict[str, int]]:
    buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for unit in units:
        if unit.get("unit_type") != "commit_message":
            continue
        timestamp = unit.get("created_time")
        if not timestamp:
            continue
        author = unit.get("author") or "unknown"
        bucket_start = _parse_time(timestamp).date()
        bucket_key = bucket_start.isoformat()
        buckets[bucket_key][author] += 1
    return buckets


def detect_governance_shifts(semantic_units_path: str, output_path: str, window_days: int) -> None:
    units = list(read_jsonl(semantic_units_path))
    units_by_repo: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for unit in units:
        units_by_repo[unit["repo_url"]].append(unit)

    shifts = []
    for repo_url, repo_units in units_by_repo.items():
        commit_units = [unit for unit in repo_units if unit.get("unit_type") == "commit_message"]
        commit_units.sort(key=lambda item: item.get("created_time") or "")
        if not commit_units:
            continue
        buckets = _bucket_commits(commit_units, window_days)
        bucket_keys = sorted(buckets.keys())
        if len(bucket_keys) < 2:
            continue
        previous_authors = Counter()
        previous_total = 0
        for bucket_key in bucket_keys:
            authors = buckets[bucket_key]
            total_commits = sum(authors.values())
            if previous_total:
                new_authors = [author for author in authors if author not in previous_authors]
                active_new = [author for author in new_authors if authors[author] >= 3]
                if active_new:
                    shifts.append(
                        {
                            "repo_url": repo_url,
                            "window_start": bucket_key,
                            "shift_type": "new_author",
                        }
                    )
                previous_avg = previous_total / max(len(previous_authors), 1)
                if previous_avg and total_commits > previous_avg * 3:
                    shifts.append(
                        {
                            "repo_url": repo_url,
                            "window_start": bucket_key,
                            "shift_type": "activity_spike",
                        }
                    )
                dropped = [author for author in previous_authors if author not in authors]
                if dropped and len(dropped) >= max(1, len(previous_authors) // 2):
                    shifts.append(
                        {
                            "repo_url": repo_url,
                            "window_start": bucket_key,
                            "shift_type": "maintainer_drop",
                        }
                    )
            previous_authors = Counter(authors)
            previous_total = total_commits
    write_jsonl(output_path, shifts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4: governance analysis")
    parser.add_argument("--semantic-units", default="data/semantic_units.jsonl")
    parser.add_argument("--output", default="data/governance_shifts.jsonl")
    parser.add_argument("--window-days", type=int, default=30)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    detect_governance_shifts(args.semantic_units, args.output, args.window_days)


if __name__ == "__main__":
    main()