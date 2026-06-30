"""Domain model dataclasses shared across the package."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class HoneypotReport:
    """Result of running all impossible-profile checks on a single candidate."""
    candidate_id: str
    is_honeypot: bool
    reasons: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)       # hard checks: A B E F G H
    soft_flags: List[str] = field(default_factory=list)  # soft checks: P1 P2 P3


@dataclass
class ScoredCandidate:
    """Output record for a single ranked candidate."""
    candidate_id: str
    rank: int
    score: float
    reasoning: str
