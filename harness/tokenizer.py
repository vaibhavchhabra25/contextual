"""Token counting helpers (tiktoken cl100k_base as a proxy for Claude tokens)."""

from __future__ import annotations

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count(text: str) -> int:
    return len(_enc.encode(text))


def count_turns(turns: list) -> int:
    return sum(count(t.content) for t in turns)


def annotate(turns: list) -> list:
    """Fill in Turn.token_count in-place and return the list."""
    for t in turns:
        t.token_count = count(t.content)
    return turns
