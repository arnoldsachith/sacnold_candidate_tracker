"""Fact-grounded reasoning string generator.

Produces a ≤280-character explanation for each ranked candidate. The
explanation is deterministic: the same candidate always gets the same
template (selected via MD5 hash of candidate_id, not Python's salted
built-in hash()).

Positive signals mentioned: career trajectory keywords, verified skills,
plain-language retrieval/ranking description.
Concerns mentioned: wrong domain, consulting background, location, penalties,
notice period, low engagement, and — for lower-ranked candidates with no hard
concerns — the specific scoring component that separates them from higher-ranked
peers so that reasoning tone matches rank.
"""
from __future__ import annotations
from datetime import date

from sacnold.utils import cid_hash_int
from sacnold.constants import TODAY

# Three structurally different templates rotated by candidate_id hash.
_LEAD_TEMPLATES = [
    "{title} · {yoe:.0f}yr exp. {pos_str}.{concern_str}",
    "{pos_str}. Profile: {title}, {yoe:.0f} yrs.{concern_str}",
    "{yoe:.0f}-year {title}. {pos_str}.{concern_str}",
]

# Human-readable labels for each scoring component (used in differentiation notes)
_COMP_LABELS: dict[str, str] = {
    "career_evidence":     "career evidence depth",
    "skills_trust_tiered": "verified skill depth",
    "lexical_bm25":        "keyword match",
    "tfidf_cosine":        "topical alignment",
    "domain_nlp_ir":       "NLP/IR domain signal",
    "experience_band":     "experience band fit",
    "product_vs_services": "product company background",
    "external_validation": "external validation",
    "location":            "location fit",
}

# Rank-band closing phrases (when no hard concerns are detected)
_RANK_CLOSINGS = {
    "top":    " Strong overall fit.",           # ranks 1-25
    "upper":  " Solid fit; edges out lower-ranked peers on most signals.",  # 26-50
    "mid":    " Good fit; ranked below top half on {diff}.",                # 51-75
    "lower":  " Qualified; ranked outside top 50 — lower {diff} vs top candidates.",  # 76-100
}


def _weakest_components(comp: dict, n: int = 2) -> list[str]:
    """Return human-readable labels of the n weakest scoring components."""
    scoreable = {k: v for k, v in comp.items() if k in _COMP_LABELS}
    ranked = sorted(scoreable, key=lambda k: scoreable[k])
    return [_COMP_LABELS[k] for k in ranked[:n]]


def make_reasoning(
    c: dict,
    comp: dict,
    crit: dict,
    mult: float,
    plain_boost: float,
    ev_kws_found: list[str],
    stuffer_ratio: float,
    rel_skills: list[str],
    rank: int = 0,
    today: date = TODAY,
) -> str:
    """Build a concise, fact-grounded reasoning string.

    Args:
        rank: Final rank position (1–100). Used to calibrate tone so that
              reasoning for rank-90 candidates does not read identically to
              reasoning for rank-5 candidates.
        All other expensive precomputations are passed in — no text scanning here.
    """
    p = c["profile"]
    yoe = float(p.get("years_of_experience", 0) or 0)
    title = p.get("current_title", "") or ""
    s = c.get("redrob_signals", {}) or {}

    # ── Positive signals ──────────────────────────────────────────────────────
    pos: list[str] = []
    if comp["career_evidence"] >= 0.7:
        pos.append(
            ", ".join(ev_kws_found) + " expertise across career"
            if ev_kws_found else "relevant AI/ML trajectory"
        )
    elif comp["career_evidence"] >= 0.45:
        pos.append("adjacent production ML experience")

    if rel_skills:
        pos.append("verified skills: " + " · ".join(rel_skills))

    if plain_boost > 0.03:
        pos.append("describes retrieval/ranking concepts in own words")

    # ── Hard concerns (always surfaced regardless of rank) ────────────────────
    concerns: list[str] = []
    if comp["domain_nlp_ir"] <= 0.1:
        concerns.append("CV/speech domain focus, limited IR signal")
    if comp["product_vs_services"] <= 0.3:
        concerns.append("consulting/services background throughout")
    if comp["location"] <= 0.3:
        concerns.append("location outside preferred region")
    if comp.get("penalty", 1.0) < 0.90:
        if "research" in title.lower():
            concerns.append("research focus, limited production deployment evidence")
        elif stuffer_ratio >= 0.65:
            concerns.append("skill breadth-to-depth ratio high")
        else:
            concerns.append("recent LLM tooling only, limited pre-LLM IR depth")
    notice = s.get("notice_period_days") or 0
    if notice and notice > 60:
        concerns.append(f"{notice}d notice")
    if mult < 0.7:
        try:
            days = (today - date.fromisoformat(s.get("last_active_date"))).days
            if days > 90:
                concerns.append(f"inactive ~{days // 30}mo")
        except Exception:
            pass
        rr = s.get("recruiter_response_rate")
        if rr is not None and rr < 0.15:
            concerns.append(f"{int(rr * 100)}% recruiter response")

    # ── Closing phrase — rank-aware when no hard concerns ────────────────────
    if concerns:
        concern_str = " Note: " + "; ".join(concerns) + "."
    else:
        if rank <= 25:
            concern_str = _RANK_CLOSINGS["top"]
        elif rank <= 50:
            concern_str = _RANK_CLOSINGS["upper"]
        elif rank <= 75:
            weak = _weakest_components(comp, n=1)
            diff = weak[0] if weak else "secondary signals"
            concern_str = _RANK_CLOSINGS["mid"].format(diff=diff)
        else:
            # ranks 76-100: always name the specific differentiator
            weak = _weakest_components(comp, n=2)
            diff = " and ".join(weak) if weak else "secondary signals"
            concern_str = _RANK_CLOSINGS["lower"].format(diff=diff)

    pos_str = "; ".join(pos) if pos else "limited retrieval/ranking depth found"

    template_idx = cid_hash_int(c.get("candidate_id", "x")) % len(_LEAD_TEMPLATES)
    raw = _LEAD_TEMPLATES[template_idx].format(
        yoe=yoe, title=title, pos_str=pos_str, concern_str=concern_str
    )
    return raw.strip()[:280]
