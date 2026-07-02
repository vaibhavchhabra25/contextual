"""Core harness runner: takes a strategy + task + budget, returns EvalResult."""

from __future__ import annotations

import os

from groq import Groq

from harness.interface import CompressionStrategy, Task
from harness.models import EvalResult, Turn
from harness.tokenizer import annotate, count_turns

_client = Groq(api_key=os.environ["GROQ_API_KEY"])
# llama-3.3-70b is fast, cheap, and available on Groq's free tier
_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def run_once(
    task: Task,
    strategy: CompressionStrategy,
    token_budget: int,
    *,
    verbose: bool = False,
) -> EvalResult:
    # 1. Build and annotate full context
    history = annotate(task.build_context())
    tokens_original = count_turns(history)

    # 2. Log token counts turn by turn (before compression)
    turn_token_log: list[tuple[int, int]] = []
    running = 0
    for t in history:
        running += t.token_count
        turn_token_log.append((t.turn_index, running))

    # 3. Compress
    compressed = strategy.compress(history, token_budget)
    tokens_after = count_turns(compressed)

    if verbose:
        print(
            f"[{strategy.id}] budget={token_budget}  "
            f"original={tokens_original}  compressed={tokens_after}  "
            f"turns={len(history)}→{len(compressed)}"
        )

    # 4. Call the model with the compressed history + the task query
    messages = _build_messages(compressed, task.query())
    response_text = _call_model(messages)

    if verbose:
        print(f"  response: {response_text[:120]!r}")

    # 5. Evaluate
    score, label = task.evaluate(response_text)

    return EvalResult(
        task_id=task.id,
        strategy_id=strategy.id,
        token_budget=token_budget,
        tokens_original=tokens_original,
        tokens_after_compression=tokens_after,
        score=score,
        score_label=label,
        turn_token_log=turn_token_log,
    )


def _build_messages(history: list[Turn], query: str) -> list[dict]:
    msgs: list[dict] = []
    for t in history:
        if t.role == "system":
            continue
        msgs.append({"role": t.role, "content": t.content})
    msgs.append({"role": "user", "content": query})
    return msgs


def _call_model(messages: list[dict]) -> str:
    import time
    for attempt in range(5):
        try:
            resp = _client.chat.completions.create(
                model=_MODEL,
                max_tokens=512,
                messages=messages,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 2 ** attempt * 15  # 15s, 30s, 60s, 120s, 240s
                print(f"  [runner] rate limit — waiting {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Rate limit persisted after 5 retries")
