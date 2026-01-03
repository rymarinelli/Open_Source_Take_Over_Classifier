import argparse
from typing import Dict, Iterable, List, Optional

from .utils import (
    http_get_json,
    http_post_json,
    normalize_repo_url,
    prefer_github_url,
    read_jsonl,
    write_jsonl,
)

OSV_QUERY_URL = "https://api.osv.dev/v1/query"


def _query_osv(package_id: str) -> List[str]:
    ecosystem, name = package_id.split(":", 1)
    body = {"package": {"name": name, "ecosystem": ecosystem.upper()}}
    response = http_post_json(OSV_QUERY_URL, body)
    urls = []
    for affected in response.get("vulns", []):
        for item in affected.get("affected", []):
            for rng in item.get("ranges", []):
                repo = rng.get("repo")
                if repo:
                    urls.append(repo)
        for reference in affected.get("references", []):
            if reference.get("type") == "REPOSITORY":
                urls.append(reference.get("url"))
    return [normalize_repo_url(url) for url in urls if url]


def _pypi_repo_urls(name: str) -> List[str]:
    data = http_get_json(f"https://pypi.org/pypi/{name}/json")
    info = data.get("info", {})
    urls = []
    project_urls = info.get("project_urls") or {}
    for _, url in project_urls.items():
        if url:
            urls.append(url)
    if info.get("home_page"):
        urls.append(info["home_page"])
    if info.get("package_url"):
        urls.append(info["package_url"])
    return [normalize_repo_url(url) for url in urls if url]


def _npm_repo_urls(name: str) -> List[str]:
    data = http_get_json(f"https://registry.npmjs.org/{name}")
    repo = data.get("repository")
    urls = []
    if isinstance(repo, dict):
        if repo.get("url"):
            urls.append(repo["url"])
    elif isinstance(repo, str):
        urls.append(repo)
    return [normalize_repo_url(url) for url in urls if url]


def resolve_repo(package_id: str, use_osv: bool = True) -> Optional[Dict[str, str]]:
    urls = []
    source = None
    if use_osv:
        try:
            urls = _query_osv(package_id)
            source = "osv" if urls else None
        except Exception:
            urls = []
            source = None
    ecosystem, name = package_id.split(":", 1)
    if not urls:
        try:
            if ecosystem == "pypi":
                urls = _pypi_repo_urls(name)
            elif ecosystem == "npm":
                urls = _npm_repo_urls(name)
            source = "registry" if urls else None
        except Exception:
            urls = []
            source = None
    if not urls:
        return None
    repo_url = prefer_github_url(urls)
    if not repo_url:
        return None
    return {"package_id": package_id, "repo_url": repo_url, "source": source or "registry"}


def resolve_repos(packages_path: str, output_path: str, use_osv: bool = True) -> None:
    resolved = []
    for row in read_jsonl(packages_path):
        package_id = row["package_id"]
        entry = resolve_repo(package_id, use_osv=use_osv)
        if entry:
            resolved.append(entry)
    write_jsonl(output_path, resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2: resolve repositories")
    parser.add_argument("--packages", default="data/packages.jsonl")
    parser.add_argument("--output", default="data/resolved_repos.jsonl")
    parser.add_argument("--no-osv", action="store_true", help="Skip OSV lookup")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    resolve_repos(args.packages, args.output, use_osv=not args.no_osv)


if __name__ == "__main__":
    main()