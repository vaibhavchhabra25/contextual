"""Hybrid strategy: semantic retrieval + structural exemptions.

Combines the best of SemanticRetrieval and VersionedContextEngine:

1. Turns tagged as type:rule or tool:edit* are unconditionally kept verbatim
   (same exemptions as the versioned engine's GC KEEP tier).
2. The remaining budget is filled by embedding-based retrieval — the highest
   cosine-similarity turns to the task query are greedily selected.
3. The most-recent `keep_last` turns are always included as a recency anchor.

This means:
- Instructions and code edits are never evicted (versioned engine's strength).
- Fact-carrying turns survive via semantic similarity to the query (semantic
  retrieval's strength).
- Neither LLM summarization nor a full segment store is required.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from engine.store import _extract_tags  # reuse tag detection from the store
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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _is_exempt(turn: Turn) -> bool:
    """True if this turn must be kept verbatim regardless of budget."""
    tags = _extract_tags(turn.content)
    if "type:rule" in tags:
        return True
    if any(t.startswith("tool:edit") for t in tags):
        return True
    return False


class HybridStrategy(CompressionStrategy):
    """Semantic retrieval with structural exemptions for rules and edits.

    Parameters
    ----------
    query_hint:
        Approximation of the downstream query used to rank turns by relevance.
        Set per-task in the benchmark runner (same as SemanticRetrieval).
    keep_last:
        Always keep this many of the most-recent turns as a recency anchor.
    """

    id = "hybrid"

    def __init__(self, query_hint: str | None = None, keep_last: int = 3) -> None:
        self.query_hint = query_hint
        self.keep_last = keep_last

    def compress(self, history: list[Turn], budget: int) -> list[Turn]:
        if count_turns(history) <= budget:
            return history

        # ── Phase 1: separate exempt, recency-anchor, and candidate turns ──────
        recent = history[-self.keep_last:]
        non_recent = history[: len(history) - self.keep_last]

        exempt: list[Turn] = []
        candidates: list[Turn] = []
        for turn in non_recent:
            (exempt if _is_exempt(turn) else candidates).append(turn)

        # ── Phase 2: compute remaining budget after exempt + recent are reserved
        tokens_exempt = sum(count(t.content) for t in exempt)
        tokens_recent = count_turns(recent)
        remaining = budget - tokens_exempt - tokens_recent

        # If exempt + recent already exceeds budget, drop candidates entirely
        if remaining <= 0 or not candidates:
            kept = exempt + recent
            kept.sort(key=lambda t: t.turn_index)
            return kept

        # ── Phase 3: semantic retrieval over candidates ───────────────────────
        embedder = _get_embedder()
        query_text = self.query_hint or recent[-1].content
        query_vec = embedder.encode(query_text, normalize_embeddings=True)

        texts = [t.content for t in candidates]
        vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64)

        scored = sorted(
            zip(candidates, vecs),
            key=lambda pair: _cosine(query_vec, pair[1]),
            reverse=True,
        )

        selected: list[Turn] = []
        for turn, _ in scored:
            tok = count(turn.content)
            if remaining - tok < 0:
                continue
            selected.append(turn)
            remaining -= tok

        # ── Phase 4: reassemble in original turn order ────────────────────────
        result = exempt + selected + recent
        result.sort(key=lambda t: t.turn_index)
        return result
