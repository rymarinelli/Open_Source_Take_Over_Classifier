import argparse
import os
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

from .utils import read_jsonl, write_jsonl


def _git(cmd: List[str], cwd: str) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _chunk_diff(diff_text: str, max_chars: int = 4000) -> List[str]:
    if len(diff_text) <= max_chars:
        return [diff_text]
    chunks = []
    current = []
    current_len = 0
    for line in diff_text.splitlines():
        if line.startswith("diff --git") and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line)
            continue
        if current_len + len(line) + 1 > max_chars:
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue
        lines = chunk.splitlines()
        midpoint = len(lines) // 2
        final_chunks.append("\n".join(lines[:midpoint]))
        final_chunks.append("\n".join(lines[midpoint:]))
    return final_chunks


def _collect_commits(repo_path: str, since: str, until: str) -> List[str]:
    output = _git(
        ["git", "log", f"--since={since}", f"--until={until}", "--pretty=format:%H"],
        cwd=repo_path,
    )
    return [line for line in output.splitlines() if line]


def build_windows(
    shifts_path: str,
    repos_path: str,
    repo_cache: str,
    output_path: str,
    window_days: int,
    max_windows: int,
    max_commits: int,
) -> None:
    repo_lookup = {row["repo_url"]: row for row in read_jsonl(repos_path)}
    windows = []
    for shift in read_jsonl(shifts_path):
        if max_windows and len(windows) >= max_windows:
            break
        repo_url = shift["repo_url"]
        repo_entry = repo_lookup.get(repo_url)
        if not repo_entry:
            continue
        repo_slug = repo_url.rstrip("/").split("/")[-1]
        repo_path = os.path.join(repo_cache, repo_slug)
        if not os.path.isdir(repo_path):
            continue
        start = _parse_time(shift["window_start"]).date()
        end = start + timedelta(days=window_days)
        commits = _collect_commits(repo_path, start.isoformat(), end.isoformat())
        if max_commits:
            commits = commits[:max_commits]
        diff_chunks = []
        for commit in commits:
            diff = _git(["git", "show", commit, "--patch"], cwd=repo_path)
            for idx, chunk in enumerate(_chunk_diff(diff)):
                diff_chunks.append(
                    {
                        "chunk_id": f"diff:{commit}:{idx}",
                        "commit_hash": commit,
                        "text": chunk,
                    }
                )
        windows.append(
            {
                "repo_url": repo_url,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "shift_type": shift["shift_type"],
                "commits": commits,
                "diff_chunks": diff_chunks,
            }
        )
    write_jsonl(output_path, windows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5: timeline windowing")
    parser.add_argument("--shifts", default="data/governance_shifts.jsonl")
    parser.add_argument("--repos", default="data/resolved_repos.jsonl")
    parser.add_argument("--repo-cache", default="data/repos")
    parser.add_argument("--output", default="data/windows.jsonl")
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--max-windows", type=int, default=50)
    parser.add_argument("--max-commits", type=int, default=200)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    build_windows(
        args.shifts,
        args.repos,
        args.repo_cache,
        args.output,
        args.window_days,
        args.max_windows,
        args.max_commits,
    )


if __name__ == "__main__":
    main()