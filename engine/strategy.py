"""VersionedContextEngine — implements CompressionStrategy.

Wraps the SegmentStore + GC policy behind the standard compress() interface
so it can be benchmarked head-to-head against the baseline strategies.

Also exposes the store publicly so the CLI can inspect history after a run.
"""

from __future__ import annotations

from engine.gc import apply_gc
from engine.store import SegmentStore
from harness.interface import CompressionStrategy
from harness.models import Turn
from harness.tokenizer import count_turns


class VersionedContextEngine(CompressionStrategy):
    """Structured, versioned context compression.

    Each turn is ingested as a ContextSegment.  Supersession is detected
    automatically (similar content with overlapping tags → old segment marked
    superseded).  GC classifies surviving segments into keep / summarize / drop
    based on recency, tag type, and reference count.
    """

    id = "versioned_engine"

    def __init__(self, recency_window: int = 8, verbose: bool = False) -> None:
        self.recency_window = recency_window
        self.verbose = verbose
        self.store = SegmentStore()

    def compress(self, history: list[Turn], budget: int) -> list[Turn]:
        if count_turns(history) <= budget:
            # Still ingest so history is inspectable even when no compression needed
            self._ingest_all(history)
            return history

        self._ingest_all(history)
        current_turn = history[-1].turn_index if history else 0
        segments = self.store.all_in_order()
        return apply_gc(
            segments,
            budget=budget,
            current_turn=current_turn,
            recency_window=self.recency_window,
            verbose=self.verbose,
        )

    def _ingest_all(self, history: list[Turn]) -> None:
        self.store.reset()
        for turn in history:
            self.store.ingest(turn)

    def reset(self) -> None:
        self.store.reset()
