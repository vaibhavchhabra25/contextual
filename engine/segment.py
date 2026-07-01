"""ContextSegment — the atomic unit the versioned engine tracks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SegmentStatus(str, Enum):
    ACTIVE = "active"          # in play, keep verbatim
    SUPERSEDED = "superseded"  # replaced by a newer segment
    SUMMARIZED = "summarized"  # collapsed into a summary segment
    DROPPED = "dropped"        # GC'd, not in context


@dataclass
class ContextSegment:
    content: str
    created_turn: int
    role: str                              # user / assistant / system / tool
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    # versioning
    supersedes: list[str] = field(default_factory=list)  # IDs this replaces
    superseded_by: Optional[str] = None                   # ID that replaced this
    # tagging
    tags: list[str] = field(default_factory=list)
    # access tracking
    last_referenced_turn: int = 0
    reference_count: int = 0
    # lifecycle
    status: SegmentStatus = SegmentStatus.ACTIVE
    token_count: int = 0

    def mark_superseded_by(self, new_id: str) -> None:
        self.superseded_by = new_id
        self.status = SegmentStatus.SUPERSEDED

    def record_reference(self, turn: int) -> None:
        self.last_referenced_turn = turn
        self.reference_count += 1

    @property
    def is_active(self) -> bool:
        return self.status == SegmentStatus.ACTIVE
