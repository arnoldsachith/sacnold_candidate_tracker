"""Shared math utilities used across scoring modules."""
from __future__ import annotations
import hashlib


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: t=0 → a, t=1 → b. t is clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def cid_hash_int(candidate_id: str) -> int:
    """Deterministic integer hash of a candidate_id — stable across Python runs.

    Uses MD5 instead of Python's built-in hash() because hash() is
    process-salted and produces different values on each interpreter start.
    """
    return int(hashlib.md5(candidate_id.encode()).hexdigest(), 16)
