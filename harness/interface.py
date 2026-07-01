"""Strategy and Task abstract interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from harness.models import EvalResult, Turn


class CompressionStrategy(ABC):
    """Drop-in interface for every compression strategy."""

    id: str  # unique slug, e.g. "naive_truncation"

    @abstractmethod
    def compress(self, history: list[Turn], budget: int) -> list[Turn]:
        """Return a compressed history that fits within *budget* tokens.

        The returned list must be a valid prefix/subset/rewrite of *history*
        whose total token_count sum is <= budget.
        """


class Task(ABC):
    """One benchmark task type."""

    id: str

    @abstractmethod
    def build_context(self) -> list[Turn]:
        """Return the full uncompressed turn history for this task instance."""

    @abstractmethod
    def query(self) -> str:
        """The question / prompt sent *after* the (possibly compressed) history."""

    @abstractmethod
    def evaluate(self, response: str) -> tuple[float, str]:
        """Score the model's response.  Returns (score 0..1, label)."""
