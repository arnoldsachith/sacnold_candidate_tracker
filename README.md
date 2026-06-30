# Sacnold — Candidate Ranking Engine

A two-pass streaming ranker that scores and ranks candidates for an AI/ML engineering role using BM25 + TF-IDF cosine similarity, tiered skill trust scoring, and a honeypot detection gate — all on CPU, no external APIs, no GPU.

**Live sandbox:** https://huggingface.co/spaces/ArnoldSachith/sacnold

---

## Reproduce the submission

```bash
git clone https://github.com/arnoldsachith/sacnold_candidate_tracker.git
cd sacnold_candidate_tracker
pip install -r requirements.txt
python scripts/rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

That single command reads `candidates.jsonl`, scores all candidates, gates honeypots, and writes the top-100 ranked CSV to `submission.csv`.

**Runtime:** ~42 s on a standard laptop CPU for 100K candidates, ~350 MB peak memory.

Optional flags:

```bash
python scripts/rank.py \
  --candidates ./candidates.jsonl \
  --criteria  config/jd_criteria.json \
  --out       ./submission.csv \
  --top       100
```

---

**Requirements:** Python 3.10+, no GPU needed.

---

## Architecture

The ranker is a **two-pass streaming pipeline** that reads the candidate file twice and never holds more than a fixed-size heap in memory (O(heap), not O(N)).

### Pass 1 — IDF estimation

Every candidate's text is tokenized and observed by a `StreamingNLP` accumulator that counts document frequencies. After the full pass, inverse document frequencies are finalized. This shared IDF table is used by both BM25 and TF-IDF in Pass 2, ensuring the two signals are calibrated on the same corpus.

Honeypot gating is intentionally skipped in Pass 1: ~80 honeypots in 100K candidates (0.08% contamination) have negligible IDF distortion, and skipping saves a full gate-check pass.

### Pass 2 — Score and heap

Each candidate is gated through honeypot detection first. Honeypots are skipped entirely. For clean candidates, all scoring happens in a single forward pass over the tokenized text — no text is re-scanned.

**Scoring components and weights:**

| Component | What it measures | Weight |
|---|---|---|
| `career_evidence` | Presence of core IR/ranking keywords + build verbs across career | 0.28 |
| `skills_trust_tiered` | Tiered skill trust (endorsements + duration + proficiency + assessments) | 0.22 |
| `lexical_bm25` | BM25 score against JD query terms | 0.15 |
| `tfidf_cosine` | TF-IDF cosine similarity to JD query | 0.10 |
| `domain_nlp_ir` | NLP/IR domain signal vs. CV/speech wrong-domain signal | 0.10 |
| `experience_band` | Years-of-experience band fit to JD target range | 0.07 |
| `product_vs_services` | Product-company vs. consulting/services background | 0.04 |
| `external_validation` | Patents, publications, open-source contributions | 0.02 |
| `location` | Location match to preferred region | 0.02 |

**Tiered skill trust formula** (per skill):

```
trust = 0.40 × (endorsements / 50)
      + 0.40 × (duration_months / 48)
      + 0.20 × proficiency_val
      + assessment_bonus   # up to +0.25
```

Skills are normalized to canonical groups via `SKILL_ALIASES` → `TIER_A / TIER_B / TIER_C`. Tier A (retrieval, ranking, embeddings, search infra) is weighted 3×; Tier B (ML, deep learning, LLMs) 1.5×; Tier C (data infra, LLM glue) 0.5×.

**Soft penalties** (multiplicative, applied after weighted sum):

- LLM-glue-only depth (LangChain/LlamaIndex with no pre-LLM IR work)
- Keyword-stuffer ratio (skills list breadth vastly exceeds depth evidence)
- Research-only focus with no production deployment signals
- Honeypot soft flags (0.95 per flag)

**Availability multiplier:** scales the final score by recruiter response rate and recency of last activity.

**Plain-language boost:** up to +0.10 for candidates who describe retrieval and ranking concepts in their own words rather than just listing keywords.

### Post-sort reasoning

Reasoning strings are generated **after** sorting so that each candidate's rank is known. This lets the reasoning tone be calibrated to rank position: top-25 candidates get a different closing phrase than rank 76–100, and lower-ranked candidates explicitly name the weakest scoring component that separates them from higher-ranked peers.

### Honeypot detection

Seven hard checks gate a candidate to `is_honeypot = True`:

- **A** — start date precedes company founding date
- **B** — claimed experience years exceed company age
- **E** — expert/advanced proficiency with zero duration months across many skills
- **F** — total duration months across all jobs exceed plausible career span
- **G** — skills-to-experience ratio implausibly high
- **H** — assessment score claimed with no skills listed

Three soft flags add a 0.95× penalty per flag without hard-gating.

---

## File structure

```
sacnold/
├── config/
│   └── jd_criteria.json        # Human-authored JD scoring criteria
├── data/
│   └── submission.csv          # Final 100-candidate submission
├── scripts/
│   └── rank.py                 # CLI entry point
├── src/
│   └── sacnold/
│       ├── config/             # Criteria loader
│       ├── detection/          # Honeypot detection
│       ├── io/                 # JSONL reader, CSV writer
│       ├── lexicons/           # Skill aliases + tier classification
│       ├── nlp/                # Tokenizer + streaming BM25/TF-IDF
│       ├── pipeline/           # Main ranker orchestrator
│       ├── reasoning/          # Rank-aware reasoning generator
│       └── scoring/            # Career, skills, experience, behavioral, penalty scorers
├── streamlit_app.py            # HuggingFace Spaces demo
├── pyproject.toml
└── requirements.txt
```

---

## Compute

| Metric | Value |
|---|---|
| Candidates processed | ~100,000 |
| Honeypots gated | 68 |
| Wall time (MacBook-class CPU) | ~42 s |
| Peak memory | ~350 MB |
| GPU | Not used |
| External APIs | None |
| LLM calls | None |

---

## Team

**Team name:** Sacnold  
**Sandbox:** https://huggingface.co/spaces/ArnoldSachith/sacnold
