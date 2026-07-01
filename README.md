# Contextual — Context Compression Benchmark for LLM Agents

A research tool that measures how different context compression strategies trade off **token savings against task accuracy**. Most compression tools report token savings but never measure whether compression degrades downstream task performance. This project closes that gap.

---

## What It Does

Runs a matrix of **(strategy × task × token budget)** and produces comparison tables showing where each strategy first fails, so you can make an informed tradeoff rather than guessing.

### Example output (4 tasks × 3 strategies)

```
Results — needle_in_haystack
┏━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Budget % ┃ Budget (tok) ┃ naive_truncati… ┃ rolling_summa… ┃ semantic_retri… ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│     100% │          438 │   ✓ (437 tok)   │  ✓ (431 tok)   │   ✓ (431 tok)   │
│      75% │          328 │   ✗ (300 tok)   │  ✓ (326 tok)   │   ✓ (328 tok)   │
│      50% │          219 │   ✗ (217 tok)   │  ✓ (216 tok)   │   ✓ (219 tok)   │
│      25% │          109 │   ✗ (103 tok)   │  ✗ (102 tok)   │   ✓ (106 tok)   │
└──────────┴──────────────┴─────────────────┴────────────────┴─────────────────┘
```

---

## Project Structure

```
contextual/
├── harness/
│   ├── models.py          # Turn, EvalResult dataclasses
│   ├── interface.py       # CompressionStrategy and Task ABCs
│   ├── tokenizer.py       # tiktoken-based token counting
│   └── runner.py          # run_once() — calls LLM, returns EvalResult
│
├── strategies/
│   ├── naive_truncation.py     # Drop oldest turns (control group)
│   ├── rolling_summarization.py # LLM-summarize old turns, keep recent verbatim
│   └── semantic_retrieval.py   # Embed turns, retrieve top-k by cosine similarity
│
├── tasks/
│   ├── needle_in_haystack.py      # Recall a fact buried early in context
│   ├── multi_hop_qa.py            # Answer requires combining two separated facts
│   ├── agentic_session_replay.py  # Reproduce correct code from a compressed coding session
│   └── instruction_persistence.py # Check if an early constraint survives N turns
│
├── run_benchmark.py   # CLI: runs the full matrix, prints tables
└── pyproject.toml
```

---

## Strategies

### 1. Naive Truncation (`naive_truncation`)
The control group. Drops the oldest turns until the history fits within the token budget. Fast, zero LLM calls, but blindly discards early context regardless of importance.

### 2. Rolling Summarization (`rolling_summarization`)
Splits history into *old* and *recent* turns. Calls an LLM to summarize all old turns into a single dense paragraph, then prepends that summary to the verbatim recent turns. Preserves early facts through compression but can hallucinate specific values (IDs, codes) when paraphrasing.

### 3. Semantic Retrieval (`semantic_retrieval`)
Embeds every turn with `sentence-transformers` (`all-MiniLM-L6-v2`, runs locally). At query time, scores all turns by cosine similarity to the task query and greedily keeps the highest-scoring turns that fit in budget. Excellent for single-fact retrieval; blind to multi-hop chains where neither hop individually looks relevant.

---

## Tasks

Each task stresses a different failure mode of compression:

| Task | What it tests | Key failure mode |
|---|---|---|
| **Needle in haystack** | Verbatim recall of a fact buried early in context | Strategy silently drops the needle turn |
| **Multi-hop QA** | Answer requires combining two separated facts | Strategy retrieves only one hop (or neither) |
| **Agentic session replay** | Reproduce correct code after a bug-fix + feature turn | Strategy drops the edit turn, model regenerates old buggy code |
| **Instruction persistence** | An early constraint must survive N later turns | Strategy evicts the instruction turn; model ignores the rule |

---

## Benchmark Results (observed)

Entries show the first budget level where each strategy's score drops below 1.0:

| Task | naive_truncation | rolling_summarization | semantic_retrieval |
|---|---|---|---|
| Needle in haystack | **75%** | 25% | never fails |
| Multi-hop QA | **25%** | 50% | never fails |
| Agentic session replay | **25%** | 75% | 25% |
| Instruction persistence | **75%** | 50% | 75% |

**Key findings:**
- No single strategy dominates across all task types.
- Semantic retrieval is best for fact-retrieval tasks but fails at the same rate as naive truncation on instruction persistence (behavioral rules don't embed close to the downstream query).
- Rolling summarization is the most consistent mid-budget performer but hallucinates specific values (IDs, codes) when the summary gets squeezed.
- Agentic session replay is the only task where naive truncation outperforms rolling summarization — the key fix is near the recent end of a short session, so truncation keeps it while summarization compresses it away.

---

## Getting Started

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier is sufficient)

### Install

```bash
git clone https://github.com/vaibhavchhabra25/contextual.git
cd contextual
python3 -m venv .venv && source .venv/bin/activate
pip install anthropic tiktoken numpy rich pydantic sentence-transformers groq
```

### Run

```bash
export GROQ_API_KEY=your_key_here
python run_benchmark.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB) — subsequent runs use the cached version.

---

## Extending

### Add a new compression strategy

```python
# strategies/my_strategy.py
from harness.interface import CompressionStrategy
from harness.models import Turn

class MyStrategy(CompressionStrategy):
    id = "my_strategy"

    def compress(self, history: list[Turn], budget: int) -> list[Turn]:
        # return a compressed list of turns within budget tokens
        ...
```

Then add it to `run_benchmark.py`:
```python
from strategies.my_strategy import MyStrategy
strategies = [..., MyStrategy()]
```

### Add a new task

```python
# tasks/my_task.py
from harness.interface import Task
from harness.models import Turn

class MyTask(Task):
    id = "my_task"

    def build_context(self) -> list[Turn]: ...
    def query(self) -> str: ...
    def evaluate(self, response: str) -> tuple[float, str]: ...
```

---

## Roadmap

### Part 2 — Versioned Context Engine

The planned next strategy treats context as a structured, versioned object rather than an append-only log — analogous to how git tracks file history.

**Data model:**
```python
ContextSegment {
    id, content, embedding, created_turn,
    supersedes: [id, ...],   # what this segment replaces
    tags: [file:main.py, tool:edit, fact:user_prefs],
    last_referenced_turn, reference_count
}
```

**Core mechanics:**
- **Supersession detection** — when a file is re-read after an edit, the old file-content segment is marked superseded rather than left to linger.
- **Fine-grained GC policy** — superseded + never referenced → drop; referenced but stale → summarize; active / high-relevance → keep verbatim.
- **History inspection CLI:**
  ```
  ctx log                   # history of context states over turns
  ctx diff turn12 turn18    # what was added/removed/superseded between two turns
  ctx checkout turn12       # rehydrate the exact context at a given turn
  ```

**Why this matters:** instruction persistence and agentic session replay both fail because strategies can't distinguish "this turn is a persistent rule" from "this turn is throwaway filler." The versioned engine tags segments by type and applies type-aware GC, so behavioral constraints are never evicted.

### Other planned work
- [ ] Accuracy-vs-token-budget curve plots (matplotlib)
- [ ] JSON result export for reproducibility
- [ ] Multi-seed averaging to reduce variance in scores
- [ ] Support for Anthropic API alongside Groq (configurable backend)
- [ ] More task scenarios per task type (currently 1 scenario per type)
