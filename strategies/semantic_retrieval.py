"""Semantic retrieval strategy.

Embeds every turn once, then at query time retrieves the top-k most similar
turns (by cosine similarity to the current query) that fit within the token
budget.  Always keeps the most recent turn to preserve conversational flow.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from harness.interface import CompressionStrategy
from harness.models import Turn
from harness.tokenizer import count, count_turns

_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_MODEL_NAME)
    return _embedder


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class SemanticRetrieval(CompressionStrategy):
    """Embed all turns; retrieve top-k most relevant to the task query.

    Parameters
    ----------
    query_hint:
        A string that approximates the downstream query.  The strategy uses
        this to rank turns by relevance.  If None, falls back to the content
        of the last turn as the query.
    keep_last:
        Always keep this many of the most-recent turns regardless of score
        (preserves conversational continuity).
    """

    id = "semantic_retrieval"

    def __init__(self, query_hint: str | None = None, keep_last: int = 3) -> None:
        self.query_hint = query_hint
        self.keep_last = keep_last

    def compress(self, history: list[Turn], budget: int) -> list[Turn]:
        if count_turns(history) <= budget:
            return history

        embedder = _get_embedder()

        # Split recency-protected tail from candidates
        recent = history[-self.keep_last:]
        candidates = history[: len(history) - self.keep_last]

        if not candidates:
            return recent

        # Embed query
        query_text = self.query_hint or recent[-1].content
        query_vec = embedder.encode(query_text, normalize_embeddings=True)

        # Embed all candidate turns (batch for speed)
        texts = [t.content for t in candidates]
        vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64)

        # Score and sort by similarity descending
        scored = sorted(
            zip(candidates, vecs),
            key=lambda pair: _cosine_similarity(query_vec, pair[1]),
            reverse=True,
        )

        # Greedily pick highest-scoring turns that fit in the remaining budget
        tokens_recent = count_turns(recent)
        remaining = budget - tokens_recent
        kept: list[Turn] = []
        for turn, _ in scored:
            t_tokens = count(turn.content)
            if remaining - t_tokens < 0:
                continue
            kept.append(turn)
            remaining -= t_tokens

        # Re-sort kept turns by original turn_index to preserve order
        kept.sort(key=lambda t: t.turn_index)

        return kept + recent
