import argparse
import json
import os
import subprocess
from typing import Dict, Iterable, List, Optional

from .utils import parse_github_owner_repo, read_jsonl, write_jsonl


def _git(cmd: List[str], cwd: Optional[str] = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout


def ensure_repo_clone(repo_url: str, repo_dir: str) -> str:
    os.makedirs(repo_dir, exist_ok=True)
    slug = repo_url.rstrip("/").split("/")[-1]
    target = os.path.join(repo_dir, slug)
    if os.path.isdir(os.path.join(target, ".git")):
        return target
    _git(["git", "clone", repo_url, target])
    return target


def _parse_git_log(repo_path: str, package_id: str, repo_url: str) -> Iterable[Dict[str, str]]:
    format_str = "%H%x1f%an%x1f%ae%x1f%aI%x1f%s"
    output = _git(["git", "log", f"--pretty=format:{format_str}"], cwd=repo_path)
    for line in output.splitlines():
        commit_hash, author_name, author_email, timestamp, subject = line.split("\x1f")
        yield {
            "semantic_id": f"commit_msg:{commit_hash}",
            "package_id": package_id,
            "repo_url": repo_url,
            "commit_hash": commit_hash,
            "unit_type": "commit_message",
            "text": subject,
            "created_time": timestamp,
            "role": "author",
            "position": 0,
            "platform": "github" if "github.com" in repo_url else "other",
            "source": "git",
            "author": f"{author_name}|{author_email}",
        }


def _github_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_get(url: str, token: Optional[str]) -> Dict[str, str]:
    import urllib.request

    request = urllib.request.Request(url, headers=_github_headers(token))
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def _collect_github_prs(owner_repo: str, package_id: str, repo_url: str, token: Optional[str], max_prs: int) -> List[Dict[str, str]]:
    items = []
    page = 1
    collected = 0
    while collected < max_prs:
        url = f"https://api.github.com/repos/{owner_repo}/pulls?state=all&per_page=100&page={page}"
        data = _github_get(url, token)
        if not data:
            break
        for pr in data:
            if collected >= max_prs:
                break
            pr_id = pr["number"]
            created = pr.get("created_at")
            author = pr.get("user", {})
            items.append(
                {
                    "semantic_id": f"pr_title:{pr_id}",
                    "package_id": package_id,
                    "repo_url": repo_url,
                    "commit_hash": pr.get("merge_commit_sha") or pr.get("head", {}).get("sha") or "",
                    "unit_type": "pr_title",
                    "text": pr.get("title") or "",
                    "created_time": created or "",
                    "role": "author",
                    "position": 0,
                    "platform": "github",
                    "source": "pull_request",
                    "author": author.get("login") or "",
                }
            )
            if pr.get("body"):
                items.append(
                    {
                        "semantic_id": f"pr_description:{pr_id}",
                        "package_id": package_id,
                        "repo_url": repo_url,
                        "commit_hash": pr.get("merge_commit_sha") or pr.get("head", {}).get("sha") or "",
                        "unit_type": "pr_description",
                        "text": pr.get("body") or "",
                        "created_time": created or "",
                        "role": "author",
                        "position": 0,
                        "platform": "github",
                        "source": "pull_request",
                        "author": author.get("login") or "",
                    }
                )
            collected += 1
        page += 1
        if len(data) < 100:
            break
    return items


def extract_semantic_units(
    repos_path: str,
    output_path: str,
    repo_cache: str,
    include_prs: bool,
    max_prs: int,
) -> None:
    token = os.getenv("GITHUB_TOKEN")
    units = []
    for row in read_jsonl(repos_path):
        package_id = row["package_id"]
        repo_url = row["repo_url"]
        repo_path = ensure_repo_clone(repo_url, repo_cache)
        units.extend(_parse_git_log(repo_path, package_id, repo_url))
        if include_prs:
            owner_repo = parse_github_owner_repo(repo_url)
            if owner_repo:
                units.extend(_collect_github_prs(owner_repo, package_id, repo_url, token, max_prs))
    write_jsonl(output_path, units)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3: history reconstruction")
    parser.add_argument("--repos", default="data/resolved_repos.jsonl")
    parser.add_argument("--output", default="data/semantic_units.jsonl")
    parser.add_argument("--repo-cache", default="data/repos")
    parser.add_argument("--include-prs", action="store_true")
    parser.add_argument("--max-prs", type=int, default=500)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    extract_semantic_units(
        args.repos,
        args.output,
        args.repo_cache,
        include_prs=args.include_prs,
        max_prs=args.max_prs,
    )


if __name__ == "__main__":
    main()