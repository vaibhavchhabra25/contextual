"""GC policy — classify segments and prune to fit a token budget.

Three tiers:
  KEEP      — active + (recent OR high reference count OR tagged as rule/fact)
  SUMMARIZE — active but stale, or superseded but referenced
  DROP      — superseded + never referenced again
"""

from __future__ import annotations

import os
from enum import Enum

from groq import Groq

from engine.segment import ContextSegment, SegmentStatus
from harness.models import Turn
from harness.tokenizer import count

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

_SUMMARIZE_PROMPT = (
    "Summarize the following conversation segments. For each segment, output "
    "exactly one line in the format:\n"
    "ID:<id> | <one dense sentence preserving all specific values: codes, IDs, numbers, file names, rules>\n\n"
    "Segments:\n{segments_text}"
)

_SINGLE_SUMMARIZE_PROMPT = (
    "Summarize the following conversation segment into one dense sentence that "
    "preserves all specific values (codes, IDs, numbers, file names, rules). "
    "Output only the summary, nothing else.\n\n{content}"
)

# ── classification ─────────────────────────────────────────────────────────────

class GCTier(str, Enum):
    KEEP = "keep"
    SUMMARIZE = "summarize"
    DROP = "drop"


def _is_high_value(seg: ContextSegment, current_turn: int, recency_window: int) -> bool:
    """True if a segment should be kept verbatim regardless of status."""
    if "type:rule" in seg.tags:
        return True
    if "type:fact" in seg.tags and seg.reference_count > 0:
        return True
    if current_turn - seg.last_referenced_turn <= recency_window:
        return True
    if seg.reference_count >= 2:
        return True
    return False


def classify(
    seg: ContextSegment,
    current_turn: int,
    recency_window: int = 8,
) -> GCTier:
    if seg.status == SegmentStatus.DROPPED or seg.status == SegmentStatus.SUMMARIZED:
        return GCTier.DROP  # already handled

    if seg.is_active:
        if _is_high_value(seg, current_turn, recency_window):
            return GCTier.KEEP
        return GCTier.SUMMARIZE  # active but not high-value → compress

    # SUPERSEDED
    if seg.reference_count > 0:
        return GCTier.SUMMARIZE  # was referenced at some point — worth a note
    return GCTier.DROP           # never used after being superseded → discard


# ── summarization ──────────────────────────────────────────────────────────────

def _call_with_retry(messages: list[dict], max_tokens: int) -> str:
    import time
    for attempt in range(4):
        try:
            resp = _client.chat.completions.create(
                model=_MODEL, max_tokens=max_tokens, messages=messages,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt * 15  # 15s, 30s, 60s, 120s
                print(f"  [GC] rate limit — waiting {wait}s before retry {attempt+1}/4")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Rate limit persisted after 4 retries")


def _summarize_batch(segments: list[ContextSegment]) -> dict[str, str]:
    """Summarize multiple segments in one LLM call.  Returns {seg.id: summary}."""
    if not segments:
        return {}
    if len(segments) == 1:
        seg = segments[0]
        text = _call_with_retry(
            [{"role": "user", "content": _SINGLE_SUMMARIZE_PROMPT.format(content=seg.content)}],
            max_tokens=120,
        )
        return {seg.id: text}

    segments_text = "\n\n".join(
        f"ID:{seg.id}\n{seg.content[:400]}" for seg in segments
    )
    raw = _call_with_retry(
        [{"role": "user", "content": _SUMMARIZE_PROMPT.format(segments_text=segments_text)}],
        max_tokens=60 * len(segments),
    )

    # Parse "ID:<id> | <summary>" lines
    results: dict[str, str] = {}
    for line in raw.splitlines():
        if line.startswith("ID:") and " | " in line:
            parts = line.split(" | ", 1)
            seg_id = parts[0][3:].strip()
            summary = parts[1].strip()
            results[seg_id] = summary

    # Fall back for any segment whose ID wasn't parsed
    for seg in segments:
        if seg.id not in results:
            results[seg.id] = raw[:200]  # use full response as best-effort

    return results


def _summarize_segment(seg: ContextSegment) -> str:
    return _summarize_batch([seg]).get(seg.id, seg.content[:100])


# ── apply GC and reconstruct turns ────────────────────────────────────────────

def apply_gc(
    segments: list[ContextSegment],
    budget: int,
    current_turn: int,
    recency_window: int = 8,
    verbose: bool = False,
) -> list[Turn]:
    """Apply GC policy to *segments* and return a list of Turns within *budget*."""
    tiers = {seg.id: classify(seg, current_turn, recency_window) for seg in segments}

    if verbose:
        counts = {t: sum(1 for v in tiers.values() if v == t) for t in GCTier}
        print(f"  [GC] keep={counts[GCTier.KEEP]} summarize={counts[GCTier.SUMMARIZE]} drop={counts[GCTier.DROP]}")

    # Batch-summarize all segments that need it in one LLM call
    to_summarize = [seg for seg in segments if tiers[seg.id] == GCTier.SUMMARIZE]
    summaries = _summarize_batch(to_summarize) if to_summarize else {}

    turns: list[Turn] = []
    used_tokens = 0

    for seg in segments:
        tier = tiers[seg.id]

        if tier == GCTier.DROP:
            seg.status = SegmentStatus.DROPPED
            continue

        if tier == GCTier.SUMMARIZE:
            summary_text = summaries.get(seg.id, seg.content[:80])
            content = f"[Summary turn {seg.created_turn}]: {summary_text}"
            tok = count(content)
            if used_tokens + tok > budget:
                seg.status = SegmentStatus.DROPPED
                continue
            seg.status = SegmentStatus.SUMMARIZED
            turns.append(Turn(
                role=seg.role,
                content=content,
                turn_index=seg.created_turn,
                token_count=tok,
                metadata={"gc": "summarized", "original_id": seg.id},
            ))
            used_tokens += tok
            continue

        # KEEP
        if used_tokens + seg.token_count > budget:
            # Over budget even for a keep-tier segment — summarize as fallback
            summary_text = _summarize_segment(seg)
            content = f"[Summary turn {seg.created_turn}]: {summary_text}"
            tok = count(content)
            if used_tokens + tok > budget:
                seg.status = SegmentStatus.DROPPED
                continue
            seg.status = SegmentStatus.SUMMARIZED
            turns.append(Turn(
                role=seg.role,
                content=content,
                turn_index=seg.created_turn,
                token_count=tok,
                metadata={"gc": "summarized_keep_overflow", "original_id": seg.id},
            ))
            used_tokens += tok
        else:
            turns.append(Turn(
                role=seg.role,
                content=seg.content,
                turn_index=seg.created_turn,
                token_count=seg.token_count,
                metadata={"gc": "kept", "original_id": seg.id},
            ))
            used_tokens += seg.token_count

    return turns
