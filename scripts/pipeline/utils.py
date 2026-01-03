import csv
import dataclasses
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def http_get_json(url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=30) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def http_post_json(url: str, body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = Request(url, data=data, headers=request_headers, method="POST")
    with urlopen(request, timeout=30) as response:
        payload = response.read()
    return json.loads(payload.decode("utf-8"))


def normalize_repo_url(url: str) -> str:
    url = url.strip()
    if url.startswith("git+"):
        url = url[4:]
    if url.startswith("git://"):
        url = "https://" + url[len("git://") :]
    if url.startswith("git@github.com:"):
        url = url.replace("git@github.com:", "https://github.com/")
    if url.startswith("ssh://git@github.com/"):
        url = url.replace("ssh://git@github.com/", "https://github.com/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.rstrip("/")


def prefer_github_url(urls: Iterable[str]) -> Optional[str]:
    github_urls = [url for url in urls if "github.com" in url]
    if github_urls:
        return normalize_repo_url(github_urls[0])
    for url in urls:
        if url:
            return normalize_repo_url(url)
    return None


def parse_github_owner_repo(repo_url: str) -> Optional[str]:
    try:
        parsed = urlparse(repo_url)
    except ValueError:
        return None
    if "github.com" not in parsed.netloc:
        return None
    path = parsed.path.strip("/")
    if not path:
        return None
    return path


def stable_hash(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest, 16)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_./-]+", text.lower())


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())