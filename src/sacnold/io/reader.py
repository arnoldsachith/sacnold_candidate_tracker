"""Candidate data reader — supports JSONL and JSON array formats."""
from __future__ import annotations
import json
from typing import Generator


def iter_candidates(source) -> Generator[dict, None, None]:
    """Yield candidate dicts from a file path or an in-memory list.

    Supports:
      - A list of dicts (for testing / in-memory use)
      - A JSONL file (one JSON object per line)
      - A JSON array file (single JSON array)
    """
    if isinstance(source, list):
        yield from source
        return
    with open(source, encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            for c in json.load(f):
                yield c
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
