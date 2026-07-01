"""Naive truncation: drop oldest turns until history fits in budget."""

from __future__ import annotations

from harness.interface import CompressionStrategy
from harness.models import Turn
from harness.tokenizer import count


class NaiveTruncation(CompressionStrategy):
    id = "naive_truncation"

    def compress(self, history: list[Turn], budget: int) -> list[Turn]:
        # Walk from the newest turn backwards, keep as many as fit.
        kept: list[Turn] = []
        used = 0
        for turn in reversed(history):
            t_tokens = count(turn.content)
            if used + t_tokens > budget:
                break
            kept.append(turn)
            used += t_tokens
        return list(reversed(kept))
