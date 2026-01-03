import argparse
from typing import Dict, Iterable, List

from .utils import read_csv, write_jsonl


def _parse_reverse_deps(rows: Iterable[Dict[str, str]], ecosystem: str) -> List[Dict[str, str]]:
    packages = []
    for row in rows:
        name = row.get("package") or row.get("name") or row.get("project")
        count = row.get("reverse_dependency_count") or row.get("reverse_deps") or row.get("reverse_dependencies")
        if not name or count is None:
            continue
        packages.append(
            {
                "package_id": f"{ecosystem}:{name}",
                "ecosystem": ecosystem,
                "reverse_deps": int(float(count)),
            }
        )
    return packages


def select_top_packages(
    pypi_csv: str,
    npm_csv: str,
    top_n: int,
    output_path: str,
) -> None:
    packages = []
    if pypi_csv:
        packages.extend(_parse_reverse_deps(read_csv(pypi_csv), "pypi"))
    if npm_csv:
        packages.extend(_parse_reverse_deps(read_csv(npm_csv), "npm"))
    packages.sort(key=lambda item: item["reverse_deps"], reverse=True)
    if top_n:
        packages = packages[:top_n]
    write_jsonl(output_path, packages)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1: ecosystem scoping")
    parser.add_argument("--pypi-csv", help="CSV with PyPI reverse dependency counts")
    parser.add_argument("--npm-csv", help="CSV with npm reverse dependency counts")
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--output", default="data/packages.jsonl")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    select_top_packages(args.pypi_csv, args.npm_csv, args.top_n, args.output)


if __name__ == "__main__":
    main()