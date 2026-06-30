"""Impossible-profile detection — 7 hard checks + 3 soft flags.

Design philosophy: HIGH PRECISION over recall. A false positive (removing a
real top candidate) costs far more than a missed honeypot. Hard gates are
reserved for clear arithmetic impossibilities; suspicious-but-plausible
patterns are soft flags that reduce score rather than gate.

Hard checks (gate the candidate entirely if triggered):
  A  ≥3 skills claimed expert/advanced with 0 months of actual usage
  B  A role's duration_months contradicts its date span by > 24 months
  E  Stated YoE predates the earliest job start by > 24 months
  F  Sum of role durations deviates from stated YoE by > 36 months
  G  >25 skills listed but profile summary is < 15 words
  H  A role's end_date is before its start_date

Soft flags (passed back in report; scorer applies 0.95^n penalty):
  P1 Phantom assessment scores: skills assessed but not listed in profile
  P2 Micro-company long tenure: 10+ years at a 1–10 person company
  P3 last_active_date is before signup_date

Hard flag counts on the 100K pool (approximate):
  A → ~21, B → ~24, E → ~25, F → ~18, G → ~9, H → ~3
  Union A|B|E|F|G|H → ~85 candidates (0.085% of corpus)
"""
from __future__ import annotations
from datetime import date
from typing import Optional

from sacnold.core.models import HoneypotReport

# ── Tunable thresholds ────────────────────────────────────────────────────────
DURATION_SPAN_TOLERANCE_MONTHS = 24     # check B
EXPERIENCE_SPAN_TOLERANCE_MONTHS = 24   # check E
YOE_SUM_TOLERANCE_MONTHS = 36          # check F
ZERO_USAGE_EXPERT_THRESHOLD = 3        # check A
STUFFER_SKILL_COUNT = 25               # check G
STUFFER_SUMMARY_WORDS = 15            # check G
MICRO_TENURE_MONTHS = 120             # check P2 (soft) — 10 years


def _parse(d: Optional[str]) -> Optional[date]:
    if not d:
        return None
    try:
        return date.fromisoformat(d)
    except (ValueError, TypeError):
        return None


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def honeypot_report(candidate: dict, today: Optional[date] = None) -> HoneypotReport:
    """Run all checks on a candidate and return a HoneypotReport."""
    if today is None:
        today = date.today()

    cid = candidate.get("candidate_id", "UNKNOWN")
    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    career = candidate.get("career_history", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    yoe = float(profile.get("years_of_experience", 0) or 0)
    yoe_months = yoe * 12

    reasons: list[str] = []
    flags: list[str] = []
    soft_flags: list[str] = []

    # ── Hard Check A: claimed mastery never actually used ─────────────────
    zero_usage_expert = [
        s for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and (s.get("duration_months") or 0) == 0
    ]
    if len(zero_usage_expert) >= ZERO_USAGE_EXPERT_THRESHOLD:
        names = ", ".join(s.get("name", "?") for s in zero_usage_expert[:5])
        reasons.append(
            f"{len(zero_usage_expert)} skills claimed expert/advanced with 0 months "
            f"of actual usage ({names}) — claimed mastery never used."
        )
        flags.append("A")

    # ── Hard Check B: role duration_months contradicts its date span ──────
    for r in career:
        sd = _parse(r.get("start_date"))
        ed = _parse(r.get("end_date")) or today
        if sd is None:
            continue
        span = _months_between(sd, ed)
        claimed = r.get("duration_months") or 0
        if abs(claimed - span) > DURATION_SPAN_TOLERANCE_MONTHS:
            reasons.append(
                f"Role '{r.get('title','?')}' @ {r.get('company','?')} claims "
                f"{claimed} months but its dates span only ~{span} months."
            )
            flags.append("B")
            break

    # ── Hard Check E: stated YoE > observable career span ─────────────────
    starts = [_parse(r.get("start_date")) for r in career]
    starts = [s for s in starts if s is not None]
    if starts:
        career_span = _months_between(min(starts), today)
        if yoe_months - career_span > EXPERIENCE_SPAN_TOLERANCE_MONTHS:
            reasons.append(
                f"Claims {yoe:.1f} yrs experience but earliest job began only "
                f"~{career_span / 12:.1f} yrs ago — experience predates employment."
            )
            flags.append("E")

    # ── Hard Check F: sum of role durations vs stated YoE ─────────────────
    # Common inflation pattern: roles add up to 4 yrs but profile claims 8 yrs.
    if career and yoe_months > 0:
        sum_months = sum((r.get("duration_months") or 0) for r in career)
        if abs(sum_months - yoe_months) > YOE_SUM_TOLERANCE_MONTHS:
            reasons.append(
                f"Sum of all role durations ({sum_months / 12:.1f} yrs) deviates "
                f">3 yrs from stated experience ({yoe:.1f} yrs) — timeline inconsistency."
            )
            flags.append("F")

    # ── Hard Check G: skills stuffer ──────────────────────────────────────
    # Many skills with an almost-empty summary suggests automated profile creation.
    summary_words = len((profile.get("summary") or "").split())
    if len(skills) > STUFFER_SKILL_COUNT and summary_words < STUFFER_SUMMARY_WORDS:
        reasons.append(
            f"Claims {len(skills)} skills but profile summary is only "
            f"{summary_words} words — skills-stuffer pattern."
        )
        flags.append("G")

    # ── Hard Check H: end_date before start_date ──────────────────────────
    for r in career:
        sd = _parse(r.get("start_date"))
        ed = _parse(r.get("end_date"))
        if sd and ed and ed < sd:
            reasons.append(
                f"Role '{r.get('title','?')}' @ {r.get('company','?')} ends "
                f"({r['end_date']}) before it starts ({r['start_date']})."
            )
            flags.append("H")
            break

    # ── Soft P1: phantom assessment scores ───────────────────────────────
    listed_skills_lower = {s.get("name", "").lower() for s in skills}
    assessed = signals.get("skill_assessment_scores") or {}
    phantom = [
        name for name in assessed
        if not any(name.lower() in ls or ls in name.lower() for ls in listed_skills_lower)
    ]
    if assessed and len(phantom) > len(assessed) / 2:
        soft_flags.append(f"P1:phantom_assessments:{','.join(phantom[:3])}")

    # ── Soft P2: micro-company long tenure ───────────────────────────────
    for r in career:
        if (r.get("company_size") == "1-10"
                and (r.get("duration_months") or 0) > MICRO_TENURE_MONTHS):
            soft_flags.append(
                f"P2:micro_co_long_tenure:{r.get('company','?')} "
                f"{(r.get('duration_months', 0)) // 12}y"
            )

    # ── Soft P3: last_active before signup ───────────────────────────────
    signup = _parse(signals.get("signup_date"))
    last_active = _parse(signals.get("last_active_date"))
    if signup and last_active and last_active < signup:
        soft_flags.append("P3:last_active_before_signup")

    return HoneypotReport(
        candidate_id=cid,
        is_honeypot=bool(flags),
        reasons=reasons,
        flags=flags,
        soft_flags=soft_flags,
    )


def is_honeypot(candidate: dict, today: Optional[date] = None) -> bool:
    """Convenience wrapper — returns True if the candidate should be gated."""
    return honeypot_report(candidate, today=today).is_honeypot
