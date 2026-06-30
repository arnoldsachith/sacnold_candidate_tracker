"""Soft penalty and supplementary feature scorers.

Soft penalties are multiplicative factors applied after the fit score.
They down-weight red flags without completely excluding the candidate —
that decision is left to honeypot.py for clear impossibilities.

Also contains product_vs_services, external_validation, and location scorers
which are small, standalone signals not large enough to warrant their own file.
"""
from __future__ import annotations

from sacnold.utils import clamp


# ── Supplementary scorers ─────────────────────────────────────────────────────

def score_product_vs_services(c: dict, crit: dict) -> float:
    """Penalise candidates whose entire career is at consulting/services firms.

    A mix of product + consulting is fine; it's the pure-consulting career
    the JD warns against for this role.
    """
    consult = set(crit["soft_penalties"]["consulting_career"]["companies"])
    roles = c.get("career_history", []) or []
    if not roles:
        return 0.6
    consult_roles = sum(
        1 for r in roles
        if any(cc in (r.get("company", "") or "").lower() for cc in consult)
    )
    return clamp(1.0 - 0.8 * (consult_roles / len(roles)))


def score_external_validation(c: dict) -> float:
    """Score external signals: GitHub activity and certifications."""
    sig = c.get("redrob_signals", {}) or {}
    score = 0.3
    gh = sig.get("github_activity_score")
    if gh is not None and gh > 0:
        score += 0.5 * clamp(gh / 50.0)
    if c.get("certifications"):
        score += 0.2
    return clamp(score)


def score_location(c: dict, crit: dict) -> float:
    """Score location fit: India-preferred (Noida/Pune offices)."""
    loc = (
        (c["profile"].get("location", "") or "") + " "
        + (c["profile"].get("country", "") or "")
    ).lower()
    sig = c.get("redrob_signals", {}) or {}
    lc = crit["location"]
    if lc["preferred_country"] in loc or any(ci in loc for ci in lc["preferred_cities"]):
        return 1.0
    if sig.get("willing_to_relocate"):
        return 0.8
    return 0.3


# ── Soft penalty ──────────────────────────────────────────────────────────────

def soft_penalty(
    c: dict,
    crit: dict,
    career_evidence: float,
    has_production: bool,
    has_llm_glue: bool,
    has_depth: bool,
    stuffer_ratio: float,
) -> float:
    """Return a multiplicative penalty in (0, 1] based on soft red flags.

    All inputs are precomputed by the pipeline — no text scanning here.
    """
    factor = 1.0
    title = (c["profile"].get("current_title", "") or "").lower()
    yoe = float(c["profile"].get("years_of_experience", 0) or 0)

    # 1) Pure research without production evidence
    if "research" in title and not has_production and career_evidence < 0.6:
        factor *= 0.70

    # 2) LLM-glue only (no retrieval/ranking depth), junior
    if has_llm_glue and not has_depth and yoe < 5:
        factor *= 0.82

    # 3) Title-chaser: ≥3 short seniority climbs
    climb_words = ("senior", "staff", "principal", "lead")
    short_climbs = sum(
        1 for r in c.get("career_history", [])
        if (r.get("duration_months") or 99) < 18
        and any(t in (r.get("title", "") or "").lower() for t in climb_words)
    )
    if short_climbs >= 3:
        factor *= 0.85

    # 4) Keyword stuffer penalty
    ksd = crit.get("keyword_stuffer_detection", {})
    if stuffer_ratio >= ksd.get("stuffer_ratio_threshold", 0.65):
        factor *= ksd.get("penalty_factor", 0.60)

    return factor
