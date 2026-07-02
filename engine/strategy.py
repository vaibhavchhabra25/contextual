"""VersionedContextEngine — implements CompressionStrategy.

Wraps the SegmentStore + GC policy behind the standard compress() interface
so it can be benchmarked head-to-head against the baseline strategies.

Also exposes the store publicly so the CLI can inspect history after a run.
"""

from __future__ import annotations

import re

from engine.gc import apply_gc
from engine.segment import ContextSegment
from engine.store import SegmentStore
from harness.interface import CompressionStrategy
from harness.models import Turn
from harness.tokenizer import count_turns

# Values worth tracking as cross-turn references
_VALUE_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"   # proper nouns (Marcus Webb, Project Aurora)
    r"|\b(\d[\d,./]+\d)\b"                    # numbers / IDs (77291, 2,847,391, /var/depot)
    r"|`([^`]{3,40})`"                        # backtick-quoted tokens (file paths, identifiers)
    r'|"([^"]{3,40})"',                       # double-quoted tokens
)


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
        self._cross_reference()

    def _cross_reference(self) -> None:
        """Bump reference_count on segments whose values are cited by later segments.

        For each segment, extract distinctive values (proper nouns, numbers,
        quoted tokens).  Any later segment that contains one of those values
        counts as a back-reference to the source segment.
        """
        segs: list[ContextSegment] = self.store.all_in_order()
        for i, src in enumerate(segs):
            values = {
                m.group(0).strip('"` ')
                for m in _VALUE_RE.finditer(src.content)
                if m.group(0).strip('"` ')
            }
            if not values:
                continue
            for later in segs[i + 1:]:
                if any(v in later.content for v in values):
                    self.store.record_reference(src.id, at_turn=later.created_turn)

    def reset(self) -> None:
        self.store.reset()
