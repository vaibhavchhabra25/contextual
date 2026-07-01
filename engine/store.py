"""SegmentStore — ingests turns, detects supersession, records snapshots."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from engine.segment import ContextSegment, SegmentStatus
from harness.models import Turn
from harness.tokenizer import count

_EMBEDDER: SentenceTransformer | None = None
_SUPERSESSION_THRESHOLD = 0.88  # cosine similarity above which we call it superseded


def _get_embedder() -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ── tag extraction ─────────────────────────────────────────────────────────────

_FILE_RE = re.compile(r"(?:path|file)=(\S+)|`([^`]+\.\w+)`")
_TOOL_RE = re.compile(r"tool:(\w+)")
_RULE_MARKERS = ("system rule", "important:", "[confirmed]", "must ", "never ", "always ")
_FACT_MARKERS = ("secret code", "remember this", "employee", "badge", "password", "id is")


def _extract_tags(content: str) -> list[str]:
    lower = content.lower()
    tags: list[str] = []

    for m in _TOOL_RE.finditer(lower):
        tags.append(f"tool:{m.group(1)}")

    for m in _FILE_RE.finditer(content):
        fname = m.group(1) or m.group(2)
        tags.append(f"file:{fname}")

    if any(marker in lower for marker in _RULE_MARKERS):
        tags.append("type:rule")

    if any(marker in lower for marker in _FACT_MARKERS):
        tags.append("type:fact")

    return list(dict.fromkeys(tags))  # dedup, preserve order


def _tags_overlap(a: list[str], b: list[str]) -> bool:
    return bool(set(a) & set(b))


# ── snapshot ───────────────────────────────────────────────────────────────────

@dataclass
class Snapshot:
    turn_index: int
    active_ids: set[str]
    superseded_ids: set[str]
    event: str = ""  # human-readable summary of what changed at this turn


# ── store ──────────────────────────────────────────────────────────────────────

class SegmentStore:
    def __init__(self) -> None:
        self.segments: dict[str, ContextSegment] = {}  # id → segment
        self._ordered_ids: list[str] = []               # insertion order
        self.snapshots: list[Snapshot] = []

    # ── ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, turn: Turn) -> ContextSegment:
        embedder = _get_embedder()
        emb = embedder.encode(turn.content, normalize_embeddings=True)
        tags = _extract_tags(turn.content)

        seg = ContextSegment(
            content=turn.content,
            created_turn=turn.turn_index,
            role=turn.role,
            embedding=emb,
            tags=tags,
            last_referenced_turn=turn.turn_index,
            reference_count=1,
            token_count=count(turn.content),
        )

        # Check for supersession against active segments
        superseded_event = ""
        for existing in self._active_segments():
            if existing.embedding is None:
                continue
            sim = _cosine(emb, existing.embedding)
            # Supersede if highly similar AND shares a meaningful tag (or both
            # are large enough that similarity alone is convincing)
            tag_match = _tags_overlap(tags, existing.tags) and (
                any(t.startswith("file:") or t.startswith("type:") for t in tags)
            )
            if sim >= _SUPERSESSION_THRESHOLD and (tag_match or sim >= 0.95):
                existing.mark_superseded_by(seg.id)
                seg.supersedes.append(existing.id)
                superseded_event += f" supersedes:{existing.id}"

        self.segments[seg.id] = seg
        self._ordered_ids.append(seg.id)

        # Record snapshot
        self._record_snapshot(turn.turn_index, superseded_event.strip())
        return seg

    # ── reference tracking ────────────────────────────────────────────────────

    def record_reference(self, seg_id: str, at_turn: int) -> None:
        if seg_id in self.segments:
            self.segments[seg_id].record_reference(at_turn)

    # ── queries ───────────────────────────────────────────────────────────────

    def _active_segments(self) -> list[ContextSegment]:
        return [self.segments[i] for i in self._ordered_ids
                if self.segments[i].is_active]

    def all_in_order(self) -> list[ContextSegment]:
        return [self.segments[i] for i in self._ordered_ids]

    def get(self, seg_id: str) -> ContextSegment | None:
        return self.segments.get(seg_id)

    # ── snapshots ─────────────────────────────────────────────────────────────

    def _record_snapshot(self, turn_index: int, event: str) -> None:
        active = {i for i in self._ordered_ids if self.segments[i].is_active}
        superseded = {i for i in self._ordered_ids
                      if self.segments[i].status == SegmentStatus.SUPERSEDED}
        self.snapshots.append(Snapshot(
            turn_index=turn_index,
            active_ids=active,
            superseded_ids=superseded,
            event=event,
        ))

    def snapshot_at(self, turn_index: int) -> Snapshot | None:
        """Return the snapshot recorded at or just before turn_index."""
        result = None
        for s in self.snapshots:
            if s.turn_index <= turn_index:
                result = s
        return result

    def reset(self) -> None:
        self.segments.clear()
        self._ordered_ids.clear()
        self.snapshots.clear()
