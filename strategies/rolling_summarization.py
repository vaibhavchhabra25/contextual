"""Rolling summarization strategy.

Keeps the most-recent turns verbatim and replaces the older prefix with a
single LLM-generated summary.  The summary is generated in one call over all
old turns so nothing is silently dropped by a last-resort pop loop.
"""

from __future__ import annotations

import os

from groq import Groq

from harness.interface import CompressionStrategy
from harness.models import Turn
from harness.tokenizer import count, count_turns

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
_MODEL = "llama-3.3-70b-versatile"

_SUMMARY_PROMPT = (
    "You are a context compressor. Summarize the following conversation turns "
    "into a single, dense paragraph that preserves ALL factual details, names, "
    "codes, numbers, constraints, and decisions. Do not omit any specific values "
    "such as secret codes, IDs, or important instructions. "
    "Output only the summary paragraph, nothing else.\n\n"
    "Turns to summarize:\n{turns_text}"
)


def _summarize(turns: list[Turn], max_summary_tokens: int = 400) -> Turn:
    turns_text = "\n".join(
        f"[Turn {t.turn_index} | {t.role}]: {t.content}" for t in turns
    )
    prompt = _SUMMARY_PROMPT.format(turns_text=turns_text)
    resp = _client.chat.completions.create(
        model=_MODEL,
        max_tokens=min(max_summary_tokens, 512),
        messages=[{"role": "user", "content": prompt}],
    )
    summary_content = resp.choices[0].message.content.strip()
    summary_turn = Turn(
        role="user",
        content=f"[Summary of turns {turns[0].turn_index}–{turns[-1].turn_index}]: {summary_content}",
        turn_index=turns[0].turn_index,
        metadata={"summarized_turns": [t.turn_index for t in turns]},
    )
    summary_turn.token_count = count(summary_turn.content)
    return summary_turn


class RollingSummarization(CompressionStrategy):
    """Summarize old turns into one summary; keep recent turns verbatim.

    Parameters
    ----------
    keep_last:
        Always keep this many recent turns verbatim.
    """

    id = "rolling_summarization"

    def __init__(self, keep_last: int = 10) -> None:
        self.keep_last = keep_last

    def compress(self, history: list[Turn], budget: int) -> list[Turn]:
        if count_turns(history) <= budget:
            return history

        recent = history[-self.keep_last:]
        old = history[: len(history) - self.keep_last]

        if not old:
            # Can't compress further — just return what fits of recent
            return recent

        tokens_recent = count_turns(recent)
        # Give the summary up to (budget - recent) tokens, capped at 400
        summary_budget = max(50, budget - tokens_recent)

        summary = _summarize(old, max_summary_tokens=min(summary_budget, 400))
        compressed = [summary] + recent

        # If still over budget, trim the oldest recent turns (last resort)
        while count_turns(compressed) > budget and len(compressed) > 1:
            compressed.pop(1)  # remove oldest *recent* turn, keep summary

        return compressed
