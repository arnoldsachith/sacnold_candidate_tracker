"""Text tokenization and candidate document extraction."""
from __future__ import annotations
import re

# Matches tokens that start and end with alphanumeric/+/# chars, allowing
# internal hyphens, dots, slashes (e.g. "c++", "bm25", "fine-tuning").
# This eliminates the need to strip(".-/") after matching.
TOKEN_RE = re.compile(r"[a-z0-9+#][a-z0-9+#./-]*[a-z0-9+#]|[a-z0-9+#]")


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize a string into indexable terms."""
    return TOKEN_RE.findall(text.lower())


def candidate_text(c: dict) -> str:
    """Flatten all free-text fields of a candidate record into one string.

    Concatenates headline, summary, current title, industry, role titles,
    role descriptions, and skill names. Used as the BM25/TF-IDF document.
    """
    p = c.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    for r in c.get("career_history", []):
        parts.append(r.get("title", ""))
        parts.append(r.get("description", ""))
    for s in c.get("skills", []):
        parts.append(s.get("name", ""))
    return " ".join(parts).lower()
