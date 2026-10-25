"""A local, fixed document corpus that stands in for the web.

Experiments need the same search results every time, and pages we control so we can plant faults and
prompt injections. Search is a small deterministic TF-IDF ranking; ties break by URL.
"""
from __future__ import annotations
import re
import json
import math
import hashlib
from pathlib import Path
from collections import Counter
from ares.config import PROJECT_ROOT
from pydantic import BaseModel, Field

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus" / "docs.jsonl"
SNAPSHOTS = PROJECT_ROOT / "runs" / "corpora"

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with "
    "what which who how about into than then there their our your you we they".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


class Document(BaseModel):
    url: str
    title: str
    text: str
    tags: list[str] = Field(default_factory=list)


class Corpus:
    def __init__(self, docs: list[Document]):
        self._docs = {normalize_url(d.url): d for d in docs}
        self._tf = {u: Counter(tokenize(d.title) * 3 + tokenize(d.text)) for u, d in self._docs.items()}
        df: Counter[str] = Counter()
        for tf in self._tf.values():
            df.update(tf.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + n / c) for t, c in df.items()}

    @classmethod
    def load(cls, path: Path = DEFAULT_CORPUS) -> Corpus:
        lines = path.read_text(encoding="utf-8").splitlines()
        return cls([Document.model_validate_json(line) for line in lines if line.strip()])

    def __len__(self) -> int:
        return len(self._docs)

    def get(self, url: str) -> Document | None:
        return self._docs.get(normalize_url(url))

    def search(self, query: str, k: int = 5) -> list[Document]:
        terms = set(tokenize(query))
        scored = []
        for url, tf in self._tf.items():
            score = sum((1 + math.log(tf[t])) * self._idf[t] for t in terms if tf[t])
            if score > 0:
                scored.append((-score, url))
        scored.sort()
        return [self._docs[url] for _, url in scored[:k]]

    def fingerprint(self) -> str:
        """Identifies the exact document set (recorded in traces so replays use the same corpus)."""
        blob = json.dumps([self._docs[u].model_dump() for u in sorted(self._docs)], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def snapshot(self, directory: Path = SNAPSHOTS) -> Path:
        """Store this exact document set under its fingerprint (once), so old runs stay replayable after the
        corpus evolves."""
        path = Path(directory) / f"{self.fingerprint()}.jsonl"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.dump(path)
        return path

    def dump(self, path: Path) -> None:
        path.write_text("".join(json.dumps(d.model_dump()) + "\n" for d in self._docs.values()), encoding="utf-8")
