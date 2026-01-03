import argparse
from typing import Dict, Iterable, List

import hashlib
import numpy as np

from .utils import read_jsonl, tokenize


def hash_vector(text: str, dimension: int = 256) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    tokens = tokenize(text)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        bucket = int(digest, 16) % dimension
        vector[bucket] += 1.0
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def embed_semantic_units(semantic_units_path: str, windows_path: str, output_path: str, dimension: int) -> None:
    records = []
    for unit in read_jsonl(semantic_units_path):
        text = unit.get("text") or ""
        records.append({"semantic_id": unit["semantic_id"], "vector": hash_vector(text, dimension)})
    for window in read_jsonl(windows_path):
        for chunk in window.get("diff_chunks", []):
            records.append({"semantic_id": chunk["chunk_id"], "vector": hash_vector(chunk["text"], dimension)})

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to write parquet embeddings") from exc
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to write parquet embeddings") from exc

    df = pd.DataFrame({"semantic_id": [r["semantic_id"] for r in records], "vector": [r["vector"].tolist() for r in records]})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 6: semantic representation")
    parser.add_argument("--semantic-units", default="data/semantic_units.jsonl")
    parser.add_argument("--windows", default="data/windows.jsonl")
    parser.add_argument("--output", default="data/embeddings.parquet")
    parser.add_argument("--dimension", type=int, default=256)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    embed_semantic_units(args.semantic_units, args.windows, args.output, args.dimension)


if __name__ == "__main__":
    main()