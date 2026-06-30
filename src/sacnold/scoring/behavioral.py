"""Behavioral availability multiplier.

Applied as a multiplicative factor AFTER the fit score. It is a modifier,
not a primary signal — a stellar candidate with low behavioral scores still
ranks ahead of a mediocre candidate with high behavioral scores.

Signals weighted:
  recency                 — last_active_date freshness
  open_to_work            — explicit opt-in flag
  recruiter_response_rate — responsiveness to outreach
  interview_completion    — follow-through on interviews
  offer_acceptance_rate   — genuine intent to join (v2 addition)
  saved_by_recruiters_30d — market demand signal
  notice_period           — operational urgency
  verified_trust          — email + phone + LinkedIn identity signals
  profile_completeness    — data quality
"""
from __future__ import annotations
from datetime import date

from sacnold.utils import clamp
from sacnold.constants import TODAY


def availability_multiplier(
    c: dict,
    crit: dict,
    today: date = TODAY,
) -> float:
    """Return a behavioral multiplier in [bounds.min, bounds.max]."""
    bm = crit["behavioral_multiplier"]
    s = c.get("redrob_signals", {}) or {}
    w = bm["weights"]
    acc = 0.0

    # Recency
    try:
        days = (today - date.fromisoformat(s.get("last_active_date"))).days
    except Exception:
        days = 999
    if days <= bm["stale_after_days"]:
        rec = 1.0
    elif days <= bm["very_stale_after_days"]:
        rec = 0.6
    else:
        rec = 0.2
    acc += w["recency"] * rec

    # Open to work
    acc += w["open_to_work"] * (1.0 if s.get("open_to_work_flag") else 0.4)

    # Recruiter response rate
    rr = s.get("recruiter_response_rate") or 0
    rr_score = 0.2 if rr < bm["low_response_rate"] else clamp(rr / 0.6)
    acc += w["recruiter_response_rate"] * rr_score

    # Interview completion rate
    acc += w["interview_completion_rate"] * clamp(s.get("interview_completion_rate") or 0)

    # Offer acceptance rate (v2 addition)
    oar = s.get("offer_acceptance_rate") or 0
    acc += w["offer_acceptance_rate"] * clamp(oar / 0.7)

    # Saved by recruiters (demand signal)
    acc += w["saved_by_recruiters_30d"] * clamp((s.get("saved_by_recruiters_30d") or 0) / 8.0)

    # Notice period
    notice = s.get("notice_period_days") or 90
    notice_score = 1.0 if notice <= 30 else 0.7 if notice <= 60 else 0.4
    acc += w["notice_period"] * notice_score

    # Verified identity trust
    vt = (
        int(bool(s.get("verified_email")))
        + int(bool(s.get("verified_phone")))
        + int(bool(s.get("linkedin_connected")))
    ) / 3.0
    acc += w["verified_trust"] * vt

    # Profile completeness
    acc += w["profile_completeness"] * clamp((s.get("profile_completeness_score") or 0) / 100.0)

    lo, hi = bm["bounds"]["min"], bm["bounds"]["max"]
    return lo + (hi - lo) * acc
