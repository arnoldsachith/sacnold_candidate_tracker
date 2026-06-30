"""Output writers for submission artifacts."""
from __future__ import annotations
import csv
from pathlib import Path


def write_csv(results: list[dict], out_path: str | Path) -> None:
    """Write ranked candidates to a CSV file.

    Columns: candidate_id, rank, score, reasoning
    """
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in results:
            w.writerow([
                r["candidate_id"],
                r["rank"],
                "%.6f" % r["score"],
                r["reasoning"],
            ])
