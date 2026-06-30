"""Career evidence and domain-fit scorers.

career_evidence: weights current/past title fit + evidence keyword density
                 + build-verb presence (shipped something, vs. just studied it).

domain_nlp_ir:  rewards NLP/IR keyword hits, penalises CV/speech dominance.
"""
from __future__ import annotations
import re

from sacnold.utils import clamp

# Title lists — ordered from most to least specific
STRONG_TITLES = [
    "recommendation systems", "recommender", "search engineer", "applied ml",
    "applied scientist", "ml engineer", "machine learning engineer",
    "nlp engineer", "ai engineer", "ranking", "relevance engineer",
    "retrieval engineer", "personalization engineer", "recsys",
]
MID_TITLES = [
    "data scientist", "machine learning", "research engineer", "research scientist",
]
ADJ_TITLES = [
    "software engineer", "backend", "data engineer", "cloud engineer",
    "full stack", "full-stack", "platform engineer",
]

_AI_ML_NLP_RE = re.compile(r"\b(ai|ml|nlp)\b")


def _score_title(title_str: str) -> float:
    """Score a job title for relevance to the Senior AI Engineer role."""
    t = title_str.lower()
    has_ai_ml_nlp = bool(_AI_ML_NLP_RE.search(t))
    if any(st in t for st in STRONG_TITLES) or (
        has_ai_ml_nlp and any(w in t for w in [
            "engineer", "specialist", "architect", "lead", "scientist",
            "recommender", "recsys", "relevance", "matching", "ranker",
        ])
    ):
        return 1.0
    if any(mt in t for mt in MID_TITLES) or has_ai_ml_nlp or "data scientist" in t:
        return 0.7
    if any(at in t for at in ADJ_TITLES):
        return 0.5
    return 0.15


def score_career_evidence(
    c: dict,
    crit: dict,
    n_ev_hits: int,
    has_build_verb: bool,
) -> float:
    """Score career evidence: title trajectory + evidence keyword density.

    title_score: best of current title (full weight) and past titles (0.9x).
    evidence:    keyword hit density, boosted when paired with a build verb.
    """
    curr_score = _score_title(c["profile"].get("current_title", ""))
    past_scores = [
        _score_title(r.get("title", "")) * 0.9
        for r in c.get("career_history", []) if r.get("title")
    ]
    title_score = max(curr_score, max(past_scores, default=0.0))

    evidence = clamp(n_ev_hits / 5.0)
    if n_ev_hits >= 1 and has_build_verb:
        evidence = clamp(evidence + 0.2)
    return 0.5 * title_score + 0.5 * evidence


def score_domain_nlp_ir(nlp_hits: int, cv_hits: int) -> float:
    """Score domain fit: reward NLP/IR signals, penalise CV/speech dominance.

    nlp_hits and cv_hits are precomputed token-set intersections passed in
    from the pipeline so text is never scanned twice.
    """
    if nlp_hits == 0 and cv_hits >= 3:
        return 0.0
    base = 0.4 + 0.15 * nlp_hits - 0.12 * max(0, cv_hits - nlp_hits)
    return clamp(base)
