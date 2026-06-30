"""
Sacnold — Candidate Ranking Demo
Redrob Hackathon submission sandbox.

Run locally:  streamlit run app.py
Hosted on:    HuggingFace Spaces (Streamlit SDK)
"""
import json
import sys
import time
import io
from pathlib import Path

import streamlit as st
import pandas as pd

# ── Path setup (works both locally and on HF Spaces) ─────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from sacnold.config.loader import load_criteria
from sacnold.pipeline.ranker import rank_candidates
from sacnold.io.writer import write_csv

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sacnold — Candidate Ranker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Sacnold — Candidate Ranking Demo")
st.caption(
    "Redrob Hackathon · Intelligent Candidate Discovery · "
    "Ranks candidates for a Senior AI Engineer role using BM25 + TF-IDF + tiered skill trust scoring."
)

# ── Sidebar: input options ────────────────────────────────────────────────────
st.sidebar.header("Input")
mode = st.sidebar.radio(
    "Candidate source",
    ["Use pre-loaded sample (100 candidates)", "Upload your own JSONL file"],
)

criteria_path = ROOT / "config" / "jd_criteria.json"

@st.cache_data
def load_crit():
    return load_criteria(str(criteria_path))

@st.cache_data
def load_sample():
    path = ROOT / "data" / "sample_candidates.jsonl"
    candidates = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates

# ── Load candidates ───────────────────────────────────────────────────────────
candidates = None

if mode == "Use pre-loaded sample (100 candidates)":
    candidates = load_sample()
    st.sidebar.success(f"Loaded {len(candidates)} sample candidates.")
else:
    uploaded = st.sidebar.file_uploader(
        "Upload candidates file",
        type=["jsonl", "json"],
        help="JSONL (one candidate per line) or a JSON array. Max 100 candidates.",
    )
    if uploaded:
        raw = uploaded.read().decode("utf-8").strip()
        if raw.startswith("["):
            candidates = json.loads(raw)
        else:
            candidates = [json.loads(l) for l in raw.splitlines() if l.strip()]
        if len(candidates) > 100:
            st.sidebar.warning(f"Trimming to 100 candidates (got {len(candidates)}).")
            candidates = candidates[:100]
        st.sidebar.success(f"Loaded {len(candidates)} candidates.")

top_n = st.sidebar.slider("Candidates to rank", min_value=5, max_value=100, value=10, step=5)

# ── Run ranking ───────────────────────────────────────────────────────────────
if candidates:
    crit = load_crit()

    with st.spinner(f"Ranking {len(candidates)} candidates..."):
        t0 = time.time()
        results, stats = rank_candidates(candidates, crit, top=min(top_n, len(candidates)))
        elapsed = time.time() - t0

    st.success(
        f"Scored **{stats['scored']}** candidates · "
        f"**{stats['gated']}** honeypots gated · "
        f"completed in **{elapsed:.2f}s**"
    )

    # ── Results table ─────────────────────────────────────────────────────────
    st.subheader(f"Top {len(results)} candidates")
    df = pd.DataFrame(results)[["rank", "candidate_id", "score", "reasoning"]]
    df["score"] = df["score"].map(lambda x: f"{x:.6f}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Score distribution chart ──────────────────────────────────────────────
    st.subheader("Score distribution")
    chart_df = pd.DataFrame({"rank": [r["rank"] for r in results],
                              "score": [float(r["score"]) for r in results]})
    st.line_chart(chart_df.set_index("rank")["score"])

    # ── Download button ───────────────────────────────────────────────────────
    buf = io.StringIO()
    buf.write("candidate_id,rank,score,reasoning\n")
    for r in results:
        reasoning_escaped = r["reasoning"].replace('"', '""')
        buf.write(f'{r["candidate_id"]},{r["rank"]},{r["score"]},"{reasoning_escaped}"\n')

    st.download_button(
        label="Download ranked CSV",
        data=buf.getvalue().encode("utf-8"),
        file_name="ranked_candidates.csv",
        mime="text/csv",
    )

    # ── Methodology expander ──────────────────────────────────────────────────
    with st.expander("How scoring works"):
        weights = {k: v for k, v in crit["component_weights"].items() if not k.startswith("_")}
        st.markdown("**Component weights (sum = 1.0)**")
        wdf = pd.DataFrame(
            [(k.replace("_", " ").title(), f"{v:.0%}") for k, v in sorted(weights.items(), key=lambda x: -x[1])],
            columns=["Component", "Weight"],
        )
        st.table(wdf)
        st.markdown("""
**Scoring approach:**
- **BM25 + TF-IDF cosine** — two-pass streaming NLP; single shared IDF table built over the full corpus.
- **Tiered skill trust** — Tier A/B/C skill groups; trust = 0.40×endorsements + 0.40×duration + 0.20×proficiency.
- **Career evidence** — title trajectory + evidence keyword density + build-verb presence.
- **Honeypot detection** — 7 arithmetic consistency checks gate impossible profiles.
- **Behavioral multiplier** — recency, engagement, offer acceptance rate applied after fit score.
        """)
else:
    st.info("Select a candidate source in the sidebar to begin.")
