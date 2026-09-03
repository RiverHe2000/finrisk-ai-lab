"""Lightweight tokenizer shared by the sparse index and the hashing embedder."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

# Small, domain-neutral stopword list. Financial terms are never stopwords.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "which",
        "with",
        "this",
        "these",
        "those",
        "their",
        "there",
        "than",
        "then",
        "also",
    }
)


def tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumerics (keeping ``12.5`` and ``90-day``), drop stopwords."""
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]
