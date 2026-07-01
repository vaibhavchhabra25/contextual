"""Core data models shared across harness, strategies, and tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Turn:
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    turn_index: int = 0
    token_count: int = 0  # filled in by the harness after tokenization
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    task_id: str
    strategy_id: str
    token_budget: int
    # token usage
    tokens_original: int
    tokens_after_compression: int
    # scores
    score: float          # 0.0–1.0
    score_label: str      # e.g. "exact_match", "f1", "llm_judge"
    # per-turn token log: list of (turn_index, cumulative_tokens_at_that_point)
    turn_token_log: list[tuple[int, int]] = field(default_factory=list)
    notes: str = ""
