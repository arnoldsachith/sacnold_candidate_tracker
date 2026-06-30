"""Candidate ranking pipeline — the main orchestrator.

Two-pass streaming design (O(heap) memory, not O(N)):

  Pass 1  iter_candidates → tokenize → nlp.observe (builds IDF table)
          Honeypot skipped: 0.085% contamination has negligible IDF effect.

  Pass 2  iter_candidates → honeypot_report (gate) → score_both (BM25+TFIDF)
          → all feature scorers → weighted sum → heap of top-N

Each candidate is fully scored in a single forward pass through its data;
no text is re-scanned. All frozenset keyword lookups are O(1).

Entry point:
    results, stats = rank_candidates("candidates.jsonl", crit, top=100)
"""
from __future__ import annotations
import heapq

from sacnold.constants import TODAY
from sacnold.utils import clamp
from sacnold.nlp.tokenizer import tokenize, candidate_text
from sacnold.nlp.retrieval import StreamingNLP
from sacnold.io.reader import iter_candidates
from sacnold.lexicons.skill_aliases import normalise_skill, TIER_A_GROUPS, TIER_B_GROUPS
from sacnold.detection.honeypot import honeypot_report
from sacnold.scoring.career import score_career_evidence, score_domain_nlp_ir
from sacnold.scoring.skills import (
    score_skills_tiered,
    keyword_stuffer_ratio,
    plain_language_boost,
)
from sacnold.scoring.experience import score_experience_band
from sacnold.scoring.behavioral import availability_multiplier
from sacnold.scoring.penalties import (
    soft_penalty,
    score_product_vs_services,
    score_external_validation,
    score_location,
)
from sacnold.reasoning.generator import make_reasoning

_JD_CANONICAL_GROUPS = TIER_A_GROUPS | TIER_B_GROUPS


def rank_candidates(
    source,
    crit: dict,
    top: int = 100,
    today=TODAY,
) -> tuple[list[dict], dict]:
    """Score and rank all candidates in `source`, returning the top `top`.

    Args:
        source:  File path (JSONL or JSON array) or a list of candidate dicts.
        crit:    Loaded JD criteria dict (from config.loader.load_criteria).
        top:     Number of candidates to return.
        today:   Reference date for recency/experience calculations.

    Returns:
        (results, stats) where results is a list of dicts with keys
        candidate_id, rank, score, reasoning; and stats has scored/gated counts.
    """
    weights = {k: v for k, v in crit["component_weights"].items() if not k.startswith("_")}

    # ── Build query token set ─────────────────────────────────────────────────
    query_terms: list[str] = []
    for k in crit["core_evidence_keywords"]:
        query_terms += tokenize(k)
    for grp in crit["must_haves"].values():
        if isinstance(grp, list):
            for k in grp:
                query_terms += tokenize(k)

    nlp = StreamingNLP(query_terms)

    # ── Precompute keyword frozensets for O(1) per-candidate lookups ──────────
    core_kws = crit["core_evidence_keywords"]
    build_verbs_set = frozenset(crit["build_verbs"])

    _NLP_KW = [
        "nlp", "retrieval", "ranking", "embedding", "bert", "transformer",
        "information retrieval", "search", "recommendation", "semantic", "recsys",
    ]
    _CV_KW = crit["hard_disqualifiers"]["cv_speech_robotics_dominant_no_nlp"]["keywords"]
    _LLM_GLUE_SINGLE = frozenset(["langchain", "llamaindex"])
    _LLM_GLUE_MULTI = ["llama-index", "gpt wrapper", "openai api"]
    _DEPTH_SINGLE = frozenset([
        "faiss", "elasticsearch", "opensearch", "bm25",
        "recommendation", "ranking", "retrieval", "vector",
    ])
    _DEPTH_MULTI = ["learning to rank"]

    # Split multi-word phrases for efficient lookup
    _pl_terms = list(crit.get("plain_language_signals", {}).get("terms", {}).items())
    _pl_max = crit.get("plain_language_signals", {}).get("max_boost", 0.10)
    _core_single = frozenset(k for k in core_kws if " " not in k)
    _core_multi = [k for k in core_kws if " " in k]
    _NLP_KW_SINGLE = frozenset(k for k in _NLP_KW if " " not in k)
    _NLP_KW_MULTI = [k for k in _NLP_KW if " " in k]
    _CV_KW_SINGLE = frozenset(k for k in _CV_KW if " " not in k)
    _CV_KW_MULTI = [k for k in _CV_KW if " " in k]

    # ── Pass 1: IDF estimation ────────────────────────────────────────────────
    # Honeypot check skipped: ~85 honeypots out of 100K (0.085% contamination)
    # → negligible IDF distortion. Properly gated in pass 2.
    for c in iter_candidates(source):
        nlp.observe(tokenize(candidate_text(c)))
    nlp.finalize()

    # Calibration caps so BM25/TF-IDF land in [0, 1]
    query_tokens = tokenize(" ".join(core_kws))
    bm_cap = nlp.bm25_score(query_tokens) or 1.0
    tfidf_cap = nlp.tfidf_cosine_score(query_tokens) or 1.0

    # ── Pass 2: score and heap ────────────────────────────────────────────────
    heap: list[tuple] = []
    n_scored = 0
    n_gated = 0

    for c in iter_candidates(source):
        rep = honeypot_report(c, today=today)
        if rep.is_honeypot:
            n_gated += 1
            continue
        n_scored += 1

        text = candidate_text(c)
        tokens = tokenize(text)
        token_set = frozenset(tokens)

        # ── Feature extraction (single pass over token_set / text) ────────
        n_ev_hits = (
            len(token_set & _core_single)
            + sum(1 for k in _core_multi if k in text)
        )
        has_build_verb = bool(token_set & build_verbs_set)
        has_production = n_ev_hits >= 1 and has_build_verb

        nlp_hits = (
            len(token_set & _NLP_KW_SINGLE)
            + sum(1 for k in _NLP_KW_MULTI if k in text)
        )
        cv_hits = (
            len(token_set & _CV_KW_SINGLE)
            + sum(1 for k in _CV_KW_MULTI if k in text)
        )

        has_llm_glue = (
            bool(token_set & _LLM_GLUE_SINGLE)
            or any(m in text for m in _LLM_GLUE_MULTI)
        )
        has_depth = (
            bool(token_set & _DEPTH_SINGLE)
            or any(m in text for m in _DEPTH_MULTI)
        )

        p_boost = min(_pl_max, sum(w for p, w in _pl_terms if p in text))
        stuffer = keyword_stuffer_ratio(c, crit)

        # Top-3 evidence keywords (for reasoning — already in token_set)
        ev_kws_found = [
            k for k in core_kws
            if (k in token_set if " " not in k else k in text)
        ][:3]

        # Top-3 endorsed JD-relevant skills (for reasoning)
        rel_skills = [
            sk["name"] for sk in c.get("skills", [])
            if normalise_skill(sk.get("name", "")) in _JD_CANONICAL_GROUPS
            and (sk.get("endorsements") or 0) > 0
        ][:3]

        # ── Score components ───────────────────────────────────────────────
        _bm25_raw, _tfidf_raw = nlp.score_both(tokens)
        comp = {
            "career_evidence":     score_career_evidence(c, crit, n_ev_hits, has_build_verb),
            "skills_trust_tiered": score_skills_tiered(c, crit),
            "lexical_bm25":        clamp(_bm25_raw / bm_cap),
            "tfidf_cosine":        clamp(_tfidf_raw / tfidf_cap),
            "domain_nlp_ir":       score_domain_nlp_ir(nlp_hits, cv_hits),
            "experience_band":     score_experience_band(c, crit),
            "product_vs_services": score_product_vs_services(c, crit),
            "external_validation": score_external_validation(c),
            "location":            score_location(c, crit),
        }

        # Weighted fit score
        fit = sum(weights[k] * comp[k] for k in weights)

        # Plain-language boost (before penalties — rewards hidden gems)
        fit = clamp(fit + p_boost)

        # Soft penalties (multiplicative, precomputed inputs)
        comp["penalty"] = soft_penalty(
            c, crit, comp["career_evidence"],
            has_production, has_llm_glue, has_depth, stuffer,
        )
        soft_hp_penalty = 0.95 ** len(rep.soft_flags)
        mult = availability_multiplier(c, crit, today=today)

        final = fit * comp["penalty"] * soft_hp_penalty * mult

        # Store reasoning inputs in heap so we can generate reasoning AFTER
        # sorting — rank is required for rank-aware tone calibration.
        reasoning_inputs = (c, comp, crit, mult, p_boost, ev_kws_found, stuffer, rel_skills)
        item = (final, c["candidate_id"], reasoning_inputs)
        if len(heap) < top:
            heapq.heappush(heap, item)
        elif final > heap[0][0]:
            heapq.heapreplace(heap, item)

    # Sort descending by score, then candidate_id ascending for deterministic tie-break
    ordered = sorted(heap, key=lambda x: (-x[0], x[1]))[:top]

    # Generate reasoning now that rank is known — tone calibrated to rank position
    results = []
    for i, (score, cid, ri) in enumerate(ordered, start=1):
        c, comp, crit_, mult, p_boost, ev_kws_found, stuffer, rel_skills = ri
        reasoning = make_reasoning(
            c, comp, crit_, mult, p_boost,
            ev_kws_found, stuffer, rel_skills,
            rank=i, today=today,
        )
        results.append({
            "candidate_id": cid,
            "rank": i,
            "score": round(score, 6),
            "reasoning": reasoning,
        })
    return results, {"scored": n_scored, "gated": n_gated}
