"""Skill trust scoring, keyword-stuffer detection, plain-language boost.

Tiered trust formula (per skill):
    trust = 0.40 × (endorsements / 50)
          + 0.40 × (duration_months / 48)
          + 0.20 × proficiency_val
          + assessment_bonus  (up to +0.25)

Expert/advanced with zero usage months → trust floor of 0.05 (honeypot signal,
but not a gate — honeypot.py handles gating).
"""
from __future__ import annotations

from sacnold.utils import clamp
from sacnold.lexicons.skill_aliases import (
    normalise_skill,
    skill_tier,
    TIER_WEIGHTS,
    TIER_A_GROUPS,
    TIER_B_GROUPS,
)

# Combined set of Tier A + B canonical groups — used for stuffer detection
_JD_CANONICAL_GROUPS: frozenset[str] = TIER_A_GROUPS | TIER_B_GROUPS

_PROF_MAP: dict[str, float] = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.50,
    "beginner": 0.25,
}


def score_skills_tiered(c: dict, crit: dict) -> float:
    """Canonical normalization + Tier A/B/C weights + trust formula.

    Returns a score in [0, 1]. Best trust value per canonical group is kept
    to prevent synonym inflation. Breadth bonus applied for ≥3 Tier-A groups.
    """
    signals = c.get("redrob_signals", {}) or {}
    assessment_scores = signals.get("skill_assessment_scores", {}) or {}

    group_trust: dict[str, float] = {}

    for s in c.get("skills", []):
        name = s.get("name", "")
        proficiency = s.get("proficiency", "beginner")
        endorsements = int(s.get("endorsements") or 0)
        duration = int(s.get("duration_months") or 0)

        prof_val = _PROF_MAP.get(proficiency, 0.25)
        endorse_val = clamp(endorsements / 50.0)
        dur_val = clamp(duration / 48.0)

        # Expert/advanced with zero usage → honeypot signal (low but not zero)
        if prof_val >= 1.0 and duration == 0:
            trust = 0.05
        elif prof_val >= 0.75 and duration == 0:
            trust = 0.10
        else:
            trust = 0.40 * endorse_val + 0.40 * dur_val + 0.20 * prof_val

        # Assessment score bonus (up to +0.25)
        canonical = normalise_skill(name)
        name_lc = name.lower()
        for assessed_name, assessed_score in assessment_scores.items():
            al = assessed_name.lower()
            if al in name_lc or name_lc in al:
                trust = min(1.0, trust + (float(assessed_score) / 100.0) * 0.25)
                break

        if canonical not in group_trust or trust > group_trust[canonical]:
            group_trust[canonical] = trust

    # Weighted sum by tier
    raw = 0.0
    tier_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for canonical, trust in group_trust.items():
        tier = skill_tier(canonical)
        if tier is None:
            continue
        raw += TIER_WEIGHTS[tier] * trust
        tier_counts[tier] += 1

    # Breadth bonus for Tier-A coverage
    if tier_counts["A"] >= 5:
        raw *= 1.15
    elif tier_counts["A"] >= 3:
        raw *= 1.08

    # Penalty if no AI skills at all
    if tier_counts["A"] + tier_counts["B"] == 0:
        raw *= 0.3

    # Normalize (max raw ≈ 25 when all Tier A, perfect trust, breadth bonus)
    return clamp(raw / 20.0)


def keyword_stuffer_ratio(c: dict, crit: dict) -> float:
    """Fraction of JD-matching skills that have zero endorsements AND zero duration.

    High ratio → profile is likely padded with keywords.
    Uses canonical normalization (O(1) dict lookup) instead of substring scan.
    """
    skills = c.get("skills", []) or []
    jd_match = stuffer = 0
    for s in skills:
        canonical = normalise_skill(s.get("name", ""))
        if canonical in _JD_CANONICAL_GROUPS:
            jd_match += 1
            if (s.get("endorsements") or 0) == 0 and (s.get("duration_months") or 0) == 0:
                stuffer += 1
    return stuffer / jd_match if jd_match else 0.0


def plain_language_boost(text: str, crit: dict) -> float:
    """Small boost for candidates who describe retrieval/ranking in plain language.

    The JD explicitly warns against missing candidates who use non-canonical
    vocabulary. A curated phrase dictionary captures these signals.
    Boost is capped by plain_language_signals.max_boost (default 0.10).
    """
    pl = crit.get("plain_language_signals", {})
    terms = pl.get("terms", {})
    max_boost = pl.get("max_boost", 0.10)
    return min(max_boost, sum(w for phrase, w in terms.items() if phrase in text))
