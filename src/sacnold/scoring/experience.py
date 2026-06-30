"""Years-of-experience band scorer with smooth lerp curves.

The JD states '5–9 is a range, not a requirement; consider outside if other
signals strong.' A step function penalises 4 yrs and 5 yrs identically;
the lerp curve treats them continuously across five anchor points:

    soft_floor → acceptable_min → ideal_min ... ideal_max → acceptable_max → soft_ceiling
         0.15         0.70            1.0    ←→    1.0         0.80            0.45
"""
from __future__ import annotations

from sacnold.utils import lerp


def score_experience_band(c: dict, crit: dict) -> float:
    """Return a score in [0.15, 1.0] based on years of experience."""
    yoe = float(c["profile"].get("years_of_experience", 0) or 0)
    eb = crit["experience_band"]
    ideal_lo, ideal_hi = eb["ideal_min"], eb["ideal_max"]
    acc_lo, acc_hi = eb["acceptable_min"], eb["acceptable_max"]
    soft_lo, soft_hi = eb["soft_floor"], eb["soft_ceiling"]

    if ideal_lo <= yoe <= ideal_hi:
        return 1.0
    if yoe < ideal_lo:
        if yoe >= acc_lo:
            return lerp(0.70, 1.0, (yoe - acc_lo) / max(1, ideal_lo - acc_lo))
        if yoe >= soft_lo:
            return lerp(0.30, 0.70, (yoe - soft_lo) / max(1, acc_lo - soft_lo))
        return 0.15
    else:  # yoe > ideal_hi
        if yoe <= acc_hi:
            return lerp(1.0, 0.80, (yoe - ideal_hi) / max(1, acc_hi - ideal_hi))
        if yoe <= soft_hi:
            return lerp(0.80, 0.45, (yoe - acc_hi) / max(1, soft_hi - acc_hi))
        return 0.25
