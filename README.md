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

### 4. Versioned Context Engine (`versioned_engine`)
Treats context as a structured, versioned object rather than an append-only log. Each turn is ingested as a `ContextSegment` with embeddings, tags (`type:rule`, `type:fact`, `file:`, `tool:`), and a reference count. On compression, a three-tier GC policy classifies segments:
- **Keep verbatim** — segments tagged `type:rule` or `type:fact`, or referenced recently
- **Summarize** — active but stale segments, or superseded but previously referenced segments
- **Drop** — superseded segments that were never referenced again

All summarize-tier segments are batched into a single LLM call to minimise API usage. The engine also records a per-turn snapshot of the segment store, enabling the history inspection CLI (`ctx log / diff / checkout`).

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

Entries show the **first budget level where each strategy's average score drops below 1.0** (lower % = more resilient to compression). Each cell is averaged across **3 scenarios per task type** to reduce per-scenario variance. Results on `llama-3.1-8b-instant` via Groq.

| Task | naive | rolling | semantic | **versioned** |
|---|---|---|---|---|
| Needle in haystack | 75% | 50% | **never** | 25% |
| Multi-hop QA | 75% | 75% | **never** | 75% |
| Agentic session replay | 25% | 75% | 25% | **never** |
| Instruction persistence | 100%† | 100%† | 100%† | 100%† |

† All strategies fail even at full budget on instruction persistence — the 8B model inconsistently obeys format constraints regardless of whether the instruction survives compression. This is a model capability issue, not a compression issue; results would differ on a larger model.

**Key findings:**

- **No single strategy dominates.** Each strategy has a task type where it outperforms the others — the right choice depends on what's in the conversation.
- **Semantic retrieval wins on fact retrieval** (needle, multi-hop — never fails). Query-aware embedding selection is hard to beat when the query vocabulary overlaps with the fact's wording.
- **Versioned engine wins on agentic session replay** (never fails). The `tool:edit` GC exemption keeps the key fix turn verbatim regardless of budget; every other strategy eventually drops it and the model regenerates the old buggy code.
- **Rolling summarization hallucinates specific values.** It correctly preserves that "a code was mentioned" but substitutes a made-up value when the summary gets squeezed — a subtle and dangerous failure mode.
- **Naive truncation is unreliable across the board.** It fails on 3 of 4 tasks by 75% budget, and has no mechanism to prioritise any turn over another.
- **Multi-hop QA is the hardest task for compression.** Naive, rolling, and versioned all fail by 75% budget; only semantic retrieval (which scores both hop turns highly via embedding similarity to the query) never fails.

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
python run_benchmark.py                          # print tables only
python run_benchmark.py --json results/run.json  # also save JSON
python run_benchmark.py --json results/run.json --plot  # save JSON + generate charts
python run_benchmark.py --seeds 3               # average scores over 3 seeds
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB) — subsequent runs use the cached version.

By default the benchmark uses `llama-3.3-70b-versatile`. To use a different Groq model (e.g. to avoid daily token limits on the free tier):

```bash
GROQ_MODEL=llama-3.1-8b-instant python run_benchmark.py --json results/run.json --plot
```

### Generate plots from a saved run

```bash
python plot_results.py results/run.json            # charts in results/charts/
python plot_results.py results/run.json --out figs/ # custom output dir
```

Produces one chart per task (accuracy vs. compression ratio, one curve per strategy) plus an `overview.png` with all four tasks as subplots.

### Inspect context history (versioned engine CLI)

```bash
# Show the per-turn segment log for a task
python ctx.py log --task needle_in_haystack
python ctx.py log --task instruction_persistence

# Show what changed between two turns (added / removed / superseded segments)
python ctx.py diff 0 20 --task agentic_session_replay

# Rehydrate the exact context as it existed at a given turn
python ctx.py checkout 10 --task needle_in_haystack
```

Available tasks: `needle_in_haystack`, `multi_hop_qa`, `agentic_session_replay`, `instruction_persistence`.

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

### Completed
- [x] Benchmark harness with `CompressionStrategy` / `Task` interfaces
- [x] 3 baseline strategies: naive truncation, rolling summarization, semantic retrieval
- [x] 4 task types: needle in haystack, multi-hop QA, agentic session replay, instruction persistence
- [x] Versioned context engine with supersession detection, three-tier GC, and `type:rule` exemption
- [x] History inspection CLI: `ctx log / diff / checkout`
- [x] Configurable model via `GROQ_MODEL` env var with exponential-backoff retry

### Next
- [x] **Multi-hop co-reference grouping** — segments sharing a named entity with any KEEP-tier segment are promoted to KEEP, so hop chains are never separated by GC
- [x] **`tool:edit` GC exemption** — edit turns are tagged high-value and kept verbatim regardless of budget
- [x] **Accuracy-vs-token-budget curve plots** — `plot_results.py` generates per-task and overview charts from saved JSON
- [x] **JSON result export** — `--json results/run.json` saves all EvalResults for reproducibility
- [x] **Multi-seed averaging** — `--seeds N` reruns each cell N times and averages scores
- [x] **More task scenarios** — 3 scenarios per task type; benchmark averages scores across all scenarios
- [ ] **Anthropic API backend** — configurable provider alongside Groq
