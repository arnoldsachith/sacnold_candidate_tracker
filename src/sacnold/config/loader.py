"""Criteria loading and validation."""
from __future__ import annotations
import json
from pathlib import Path


def load_criteria(path: str | Path = "config/jd_criteria.json") -> dict:
    """Load JD scoring criteria from a JSON file.

    Validates that component_weights sum to 1.0 (same assertion as the
    runtime check in the pipeline, but surfaced early for fast feedback).
    """
    with open(path, encoding="utf-8") as f:
        crit = json.load(f)

    weights = {
        k: v for k, v in crit["component_weights"].items()
        if not k.startswith("_")
    }
    wsum = sum(weights.values())
    if abs(wsum - 1.0) >= 1e-6:
        raise ValueError(
            f"component_weights must sum to 1.0 (got {wsum:.8f}). "
            f"Edit jd_criteria.json and retry."
        )
    return crit
