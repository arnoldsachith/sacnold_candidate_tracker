"""Streaming BM25 + TF-IDF cosine scorer with a shared IDF table.

BM25 (Okapi) rewards exact-term frequency with length normalisation.
TF-IDF cosine rewards proportional topical coverage of the query.

Both share the same IDF table built in a single first pass over the corpus.
The score_both() method computes them simultaneously in ONE Counter pass,
saving ~1-2s on 100K candidates vs computing them separately.
"""
from __future__ import annotations
import math
from collections import Counter


class StreamingNLP:
    """Two-pass NLP scorer.

    Pass 1 (observe + finalize): builds the IDF table from the full corpus.
    Pass 2 (score_both):         scores each candidate against the query.
    """

    def __init__(self, query_terms: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.qterms = set(query_terms)
        self.df: Counter = Counter()
        self.N = 0
        self.total_len = 0
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self._query_vec: dict[str, float] = {}

    # ── Pass 1 ────────────────────────────────────────────────────────────────

    def observe(self, tokens: list[str]) -> None:
        """Accumulate document frequencies for query terms."""
        self.N += 1
        self.total_len += len(tokens)
        for term in set(tokens):
            if term in self.qterms:
                self.df[term] += 1

    def finalize(self) -> None:
        """Compute IDF and pre-normalize the query vector for cosine similarity."""
        self.avgdl = self.total_len / self.N if self.N else 0.0
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in self.df.items()
        }
        q_raw = {t: self.idf[t] for t in self.qterms if t in self.idf}
        q_norm = math.sqrt(sum(v * v for v in q_raw.values())) or 1.0
        self._query_vec = {t: v / q_norm for t, v in q_raw.items()}

    # ── Pass 2 ────────────────────────────────────────────────────────────────

    def score_both(self, tokens: list[str]) -> tuple[float, float]:
        """Compute BM25 and TF-IDF cosine in one Counter pass.

        Returns (bm25, tfidf_cosine).
        """
        if not self.avgdl or not tokens:
            return 0.0, 0.0
        dl = len(tokens)
        denom_norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        counts = Counter(t for t in tokens if t in self.idf)
        if not counts:
            return 0.0, 0.0

        bm25 = 0.0
        for term, f in counts.items():
            bm25 += self.idf[term] * (f * (self.k1 + 1)) / (f + denom_norm)

        tfidf = 0.0
        if self._query_vec:
            d_raw = {
                t: (counts[t] / dl) * self.idf[t]
                for t in counts if t in self._query_vec
            }
            if d_raw:
                d_norm = math.sqrt(sum(v * v for v in d_raw.values())) or 1.0
                tfidf = max(
                    0.0,
                    sum(self._query_vec[t] * (d_raw[t] / d_norm) for t in d_raw),
                )

        return bm25, tfidf

    # ── Convenience wrappers (used only for calibration, not per-candidate) ──

    def bm25_score(self, tokens: list[str]) -> float:
        return self.score_both(tokens)[0]

    def tfidf_cosine_score(self, tokens: list[str]) -> float:
        return self.score_both(tokens)[1]
