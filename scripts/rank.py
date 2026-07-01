#!/usr/bin/env python3
"""CLI entry point for the Redrob candidate ranker.

Usage:
    python scripts/rank.py
    python scripts/rank.py --candidates path/to/candidates.jsonl --top 100
    python scripts/rank.py --criteria config/jd_criteria.json --out data/submission.csv

Install the package first for cleaner imports:
    rank-candidates --help
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sacnold.config.loader import load_criteria
from sacnold.pipeline.ranker import rank_candidates
from sacnold.io.writer import write_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    ap.add_argument(
        "--candidates",
        default="../rules_hackathon/candidates.jsonl",
        help="Path to candidates.jsonl (JSONL or JSON array)",
    )
    ap.add_argument(
        "--out",
        default="data/team_sacnold.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--criteria",
        default="config/jd_criteria.json",
        help="Path to JD criteria JSON",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of candidates to return",
    )
    args = ap.parse_args()

    crit = load_criteria(args.criteria)

    t0 = time.time()
    results, stats = rank_candidates(args.candidates, crit, top=args.top)
    elapsed = time.time() - t0

    write_csv(results, args.out)
    print(
        f"Scored {stats['scored']:,} candidates "
        f"({stats['gated']} honeypots gated) in {elapsed:.1f}s → {args.out}"
    )
    print("\nTop-10 preview:")
    for r in results[:10]:
        print(f"  #{r['rank']:3d}  {r['candidate_id']}  {r['score']:.4f}  {r['reasoning'][:100]}")


if __name__ == "__main__":
    main()
