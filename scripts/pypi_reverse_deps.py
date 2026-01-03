#!/usr/bin/env python3
"""Query PyPI reverse dependency counts from BigQuery and export to CSV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Iterable

from google.cloud import bigquery


QUERY_TEMPLATE = """
WITH latest AS (
  SELECT project, requires_dist
  FROM (
    SELECT
      project,
      requires_dist,
      ROW_NUMBER() OVER (PARTITION BY project ORDER BY upload_time DESC) AS rn
    FROM `bigquery-public-data.pypi.distribution_metadata`
  )
  WHERE rn = 1
),
parsed AS (
  SELECT
    project AS depender,
    LOWER(REGEXP_EXTRACT(req, r'^([A-Za-z0-9_.-]+)')) AS dependency
  FROM latest,
  UNNEST(requires_dist) AS req
  WHERE req IS NOT NULL
)
SELECT
  dependency AS package,
  COUNT(DISTINCT depender) AS reverse_dependency_count
FROM parsed
WHERE dependency IS NOT NULL
GROUP BY package
ORDER BY reverse_dependency_count DESC
{limit_clause}
""".strip()


@dataclass
class ReverseDepRow:
    package: str
    reverse_dependency_count: int


def build_query(limit: int | None) -> str:
    limit_clause = ""
    if limit is not None:
        limit_clause = f"LIMIT {limit}"
    return QUERY_TEMPLATE.format(limit_clause=limit_clause)


def fetch_reverse_deps(
    client: bigquery.Client,
    limit: int | None,
    query: str,
) -> Iterable[ReverseDepRow]:
    job = client.query(query)
    for row in job.result():
        yield ReverseDepRow(
            package=row["package"],
            reverse_dependency_count=row["reverse_dependency_count"],
        )


def write_csv(path: str, rows: Iterable[ReverseDepRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["package", "reverse_dependency_count"])
        for row in rows:
            writer.writerow([row.package, row.reverse_dependency_count])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Query BigQuery public PyPI metadata and export reverse dependency counts."
        )
    )
    parser.add_argument(
        "--output",
        default="pypi_reverse_deps.csv",
        help="Path to write CSV output (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows returned.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional Google Cloud project ID for BigQuery billing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = bigquery.Client(project=args.project)
    query = build_query(args.limit)
    rows = fetch_reverse_deps(client, args.limit, query)
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()