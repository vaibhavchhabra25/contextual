"""Agentic session replay task.

Replays a synthetic multi-turn coding session (file reads, edits, test runs).
After compression, the model is asked to produce the final correct version of
the file.  Evaluation checks whether the output passes a set of test assertions.

Three scenarios cover different bugs and feature additions:
  0 — threshold comparison bug (>= vs >) + add 'max' field
  1 — sort direction bug (ascending vs descending) + add 'median' field
  2 — off-by-one in pagination total_pages + add 'has_next' field
"""

from __future__ import annotations

from harness.interface import Task
from harness.models import Turn

# ── Scenario 0: process_items ──────────────────────────────────────────────────

_S0_CORRECT = """\
def process_items(items: list[int], threshold: int = 10) -> dict:
    \"\"\"Return stats for items above threshold.\"\"\"
    filtered = [x for x in items if x > threshold]
    return {
        "count": len(filtered),
        "total": sum(filtered),
        "average": sum(filtered) / len(filtered) if filtered else 0.0,
        "max": max(filtered) if filtered else None,
    }
"""

_S0_TURNS = [
    ("user", "tool:read_file path=utils.py", 0),
    ("assistant", """\
```python
def process_items(items: list[int], threshold: int = 10) -> dict:
    \"\"\"Return stats for items above threshold.\"\"\"
    filtered = [x for x in items if x >= threshold]  # BUG: should be >
    return {
        "count": len(filtered),
        "total": sum(filtered),
        "average": sum(filtered) / len(filtered) if filtered else 0.0,
    }
```""", 1),
    ("user", "tool:run_tests tests/test_utils.py", 2),
    ("assistant", """\
FAILED tests/test_utils.py::test_threshold_is_exclusive
  AssertionError: expected count=2 for items=[10,15,20] threshold=10, got count=3
  The boundary item (10) should be excluded.""", 3),
    ("user", "The test expects strict greater-than. Let me fix the comparison.", 4),
    ("assistant", "Agreed. The filter should use `x > threshold` not `x >= threshold`.", 5),
    ("user", "Also, the product team wants a 'max' field added to the return dict.", 6),
    ("assistant", "Got it. I'll add `\"max\": max(filtered) if filtered else None` to the return.", 7),
    ("user", "tool:read_file tests/test_utils.py", 8),
    ("assistant", """\
```python
def test_threshold_is_exclusive():
    result = process_items([10, 15, 20], threshold=10)
    assert result["count"] == 2
    assert result["total"] == 35

def test_max_field():
    result = process_items([5, 12, 18], threshold=10)
    assert result["max"] == 18

def test_empty_result():
    result = process_items([1, 2, 3], threshold=10)
    assert result["count"] == 0
    assert result["average"] == 0.0
    assert result["max"] is None
```""", 9),
    ("user", "tool:edit_file path=utils.py", 10),
    ("assistant", """\
Applied two changes to utils.py:
1. Fixed threshold comparison: `x >= threshold` → `x > threshold`
2. Added `"max": max(filtered) if filtered else None` to return dict

Updated file:
```python
def process_items(items: list[int], threshold: int = 10) -> dict:
    \"\"\"Return stats for items above threshold.\"\"\"
    filtered = [x for x in items if x > threshold]
    return {
        "count": len(filtered),
        "total": sum(filtered),
        "average": sum(filtered) / len(filtered) if filtered else 0.0,
        "max": max(filtered) if filtered else None,
    }
```""", 11),
    ("user", "tool:run_tests tests/test_utils.py", 12),
    ("assistant", "All 3 tests passed. ✓", 13),
    ("user", "Let's also check the deployment config.", 14),
    ("assistant", "Sure, reading deploy.yaml now.", 15),
    ("user", "tool:read_file deploy.yaml", 16),
    ("assistant", "deploy.yaml looks fine — no changes needed.", 17),
    ("user", "Great. Can you also confirm the version in setup.cfg?", 18),
    ("assistant", "setup.cfg shows version=1.4.2. Ready to tag the release.", 19),
]


def _s0_evaluate(response: str) -> tuple[float, str]:
    namespace: dict = {}
    try:
        code = response
        if "```" in code:
            lines, in_block, extracted = code.split("\n"), False, []
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    extracted.append(line)
            code = "\n".join(extracted)
        exec(code, namespace)  # noqa: S102
        fn = namespace.get("process_items")
        if fn is None:
            return 0.0, "exec_no_function"
        checks = 0
        r = fn([10, 15, 20], threshold=10)
        if r.get("count") == 2: checks += 1
        if r.get("total") == 35: checks += 1
        if r.get("max") == 20: checks += 1
        r2 = fn([1, 2, 3], threshold=10)
        if r2.get("count") == 0 and r2.get("max") is None: checks += 1
        return checks / 4, f"exec_checks_{checks}/4"
    except Exception:
        return 0.0, "exec_error"


# ── Scenario 1: sort_scores ────────────────────────────────────────────────────

_S1_CORRECT = """\
def sort_scores(scores: list[float], top_n: int = 5) -> dict:
    \"\"\"Return the top N scores in descending order with their rank.\"\"\"
    sorted_scores = sorted(scores, reverse=True)[:top_n]
    return {
        "top": sorted_scores,
        "median": sorted_scores[len(sorted_scores) // 2] if sorted_scores else None,
        "count": len(sorted_scores),
    }
"""

_S1_TURNS = [
    ("user", "tool:read_file path=ranking.py", 0),
    ("assistant", """\
```python
def sort_scores(scores: list[float], top_n: int = 5) -> dict:
    \"\"\"Return the top N scores in descending order with their rank.\"\"\"
    sorted_scores = sorted(scores)[:top_n]  # BUG: ascending, should be descending
    return {
        "top": sorted_scores,
        "count": len(sorted_scores),
    }
```""", 1),
    ("user", "tool:run_tests tests/test_ranking.py", 2),
    ("assistant", """\
FAILED tests/test_ranking.py::test_top_scores_are_highest
  AssertionError: expected [95.0, 87.0, 76.0] as top 3, got [45.0, 67.0, 76.0]
  Scores are in ascending order — test expects highest first.""", 3),
    ("user", "Ah, sorted() is ascending by default. Need reverse=True.", 4),
    ("assistant", "Correct — `sorted(scores, reverse=True)` will give highest first.", 5),
    ("user", "While we're at it, analytics wants a 'median' of the top-N scores.", 6),
    ("assistant", "I'll add `\"median\": sorted_scores[len(sorted_scores) // 2]` to the return dict.", 7),
    ("user", "tool:read_file tests/test_ranking.py", 8),
    ("assistant", """\
```python
def test_top_scores_are_highest():
    result = sort_scores([45, 95, 67, 87, 23, 76], top_n=3)
    assert result["top"] == [95, 87, 76]

def test_median_of_top():
    result = sort_scores([10, 90, 50, 70, 30], top_n=5)
    assert result["median"] == 50

def test_empty():
    result = sort_scores([], top_n=5)
    assert result["count"] == 0
    assert result["median"] is None
```""", 9),
    ("user", "tool:edit_file path=ranking.py", 10),
    ("assistant", """\
Applied two changes to ranking.py:
1. Fixed sort direction: `sorted(scores)` → `sorted(scores, reverse=True)`
2. Added `"median": sorted_scores[len(sorted_scores) // 2] if sorted_scores else None`

Updated file:
```python
def sort_scores(scores: list[float], top_n: int = 5) -> dict:
    \"\"\"Return the top N scores in descending order with their rank.\"\"\"
    sorted_scores = sorted(scores, reverse=True)[:top_n]
    return {
        "top": sorted_scores,
        "median": sorted_scores[len(sorted_scores) // 2] if sorted_scores else None,
        "count": len(sorted_scores),
    }
```""", 11),
    ("user", "tool:run_tests tests/test_ranking.py", 12),
    ("assistant", "All 3 tests passed. ✓", 13),
    ("user", "OK, let's merge this and update the changelog.", 14),
    ("assistant", "Ready to merge. I'll draft the changelog entry.", 15),
    ("user", "tool:read_file CHANGELOG.md", 16),
    ("assistant", "CHANGELOG.md exists. Adding entry for v2.1.0: 'Fix score sort direction; add median field'.", 17),
    ("user", "Looks good. Ship it.", 18),
    ("assistant", "Tagging v2.1.0 and pushing to remote.", 19),
]


def _s1_evaluate(response: str) -> tuple[float, str]:
    namespace: dict = {}
    try:
        code = response
        if "```" in code:
            lines, in_block, extracted = code.split("\n"), False, []
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    extracted.append(line)
            code = "\n".join(extracted)
        exec(code, namespace)  # noqa: S102
        fn = namespace.get("sort_scores")
        if fn is None:
            return 0.0, "exec_no_function"
        checks = 0
        r = fn([45, 95, 67, 87, 23, 76], top_n=3)
        if r.get("top") == [95, 87, 76]: checks += 1
        r2 = fn([10, 90, 50, 70, 30], top_n=5)
        if r2.get("median") == 50: checks += 1
        r3 = fn([], top_n=5)
        if r3.get("count") == 0 and r3.get("median") is None: checks += 1
        if r.get("count") == 3: checks += 1
        return checks / 4, f"exec_checks_{checks}/4"
    except Exception:
        return 0.0, "exec_error"


# ── Scenario 2: paginate ───────────────────────────────────────────────────────

_S2_CORRECT = """\
def paginate(items: list, page: int, page_size: int = 10) -> dict:
    \"\"\"Return a page slice with pagination metadata.\"\"\"
    total = len(items)
    total_pages = (total + page_size - 1) // page_size  # ceiling division
    start = page * page_size
    end = start + page_size
    slice_ = items[start:end]
    return {
        "items": slice_,
        "page": page,
        "total_pages": total_pages,
        "has_next": page < total_pages - 1,
    }
"""

_S2_TURNS = [
    ("user", "tool:read_file path=pagination.py", 0),
    ("assistant", """\
```python
def paginate(items: list, page: int, page_size: int = 10) -> dict:
    \"\"\"Return a page slice with pagination metadata.\"\"\"
    total = len(items)
    total_pages = total // page_size  # BUG: floor division loses the last partial page
    start = page * page_size
    end = start + page_size
    return {
        "items": items[start:end],
        "page": page,
        "total_pages": total_pages,
    }
```""", 1),
    ("user", "tool:run_tests tests/test_pagination.py", 2),
    ("assistant", """\
FAILED tests/test_pagination.py::test_partial_last_page
  AssertionError: expected total_pages=3 for 25 items at page_size=10, got 2
  Floor division drops the final partial page.""", 3),
    ("user", "Right, we need ceiling division: (total + page_size - 1) // page_size.", 4),
    ("assistant", "Exactly. `(total + page_size - 1) // page_size` handles the partial page correctly.", 5),
    ("user", "The frontend also needs a 'has_next' boolean to know if there's another page.", 6),
    ("assistant", "I'll add `\"has_next\": page < total_pages - 1` after fixing total_pages.", 7),
    ("user", "tool:read_file tests/test_pagination.py", 8),
    ("assistant", """\
```python
def test_partial_last_page():
    result = paginate(list(range(25)), page=0, page_size=10)
    assert result["total_pages"] == 3

def test_has_next_true():
    result = paginate(list(range(25)), page=1, page_size=10)
    assert result["has_next"] is True

def test_has_next_false_on_last():
    result = paginate(list(range(25)), page=2, page_size=10)
    assert result["has_next"] is False
    assert len(result["items"]) == 5
```""", 9),
    ("user", "tool:edit_file path=pagination.py", 10),
    ("assistant", """\
Applied two changes to pagination.py:
1. Fixed total_pages: `total // page_size` → `(total + page_size - 1) // page_size`
2. Added `"has_next": page < total_pages - 1` to return dict

Updated file:
```python
def paginate(items: list, page: int, page_size: int = 10) -> dict:
    \"\"\"Return a page slice with pagination metadata.\"\"\"
    total = len(items)
    total_pages = (total + page_size - 1) // page_size  # ceiling division
    start = page * page_size
    end = start + page_size
    slice_ = items[start:end]
    return {
        "items": slice_,
        "page": page,
        "total_pages": total_pages,
        "has_next": page < total_pages - 1,
    }
```""", 11),
    ("user", "tool:run_tests tests/test_pagination.py", 12),
    ("assistant", "All 3 tests passed. ✓", 13),
    ("user", "Perfect. Let's update the API docs to mention has_next.", 14),
    ("assistant", "Updating docs/api.md with the new has_next field description.", 15),
    ("user", "tool:read_file docs/api.md", 16),
    ("assistant", "docs/api.md updated — documented has_next as bool indicating whether a next page exists.", 17),
    ("user", "Looks good. Ready to deploy.", 18),
    ("assistant", "Deployment checklist complete. Ready when you are.", 19),
]


def _s2_evaluate(response: str) -> tuple[float, str]:
    namespace: dict = {}
    try:
        code = response
        if "```" in code:
            lines, in_block, extracted = code.split("\n"), False, []
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block:
                    extracted.append(line)
            code = "\n".join(extracted)
        exec(code, namespace)  # noqa: S102
        fn = namespace.get("paginate")
        if fn is None:
            return 0.0, "exec_no_function"
        checks = 0
        r = fn(list(range(25)), page=0, page_size=10)
        if r.get("total_pages") == 3: checks += 1
        r2 = fn(list(range(25)), page=1, page_size=10)
        if r2.get("has_next") is True: checks += 1
        r3 = fn(list(range(25)), page=2, page_size=10)
        if r3.get("has_next") is False: checks += 1
        if len(r3.get("items", [])) == 5: checks += 1
        return checks / 4, f"exec_checks_{checks}/4"
    except Exception:
        return 0.0, "exec_error"


# ── dispatch ───────────────────────────────────────────────────────────────────

_SCENARIOS = [
    {
        "turns": _S0_TURNS,
        "query": (
            "Based on the session above, write the final correct Python implementation "
            "of the `process_items` function. Output only the function code, no explanation."
        ),
        "evaluate": _s0_evaluate,
    },
    {
        "turns": _S1_TURNS,
        "query": (
            "Based on the session above, write the final correct Python implementation "
            "of the `sort_scores` function. Output only the function code, no explanation."
        ),
        "evaluate": _s1_evaluate,
    },
    {
        "turns": _S2_TURNS,
        "query": (
            "Based on the session above, write the final correct Python implementation "
            "of the `paginate` function. Output only the function code, no explanation."
        ),
        "evaluate": _s2_evaluate,
    },
]

num_scenarios = len(_SCENARIOS)


class AgenticSessionReplay(Task):
    """Replay a coding session; check if model can reproduce the final correct code."""

    id = "agentic_session_replay"

    def __init__(self, scenario_index: int = 0) -> None:
        scenario = _SCENARIOS[scenario_index % len(_SCENARIOS)]
        self._turns = scenario["turns"]
        self._query_text = scenario["query"]
        self._eval_fn = scenario["evaluate"]

    def build_context(self) -> list[Turn]:
        return [
            Turn(role=role, content=content, turn_index=idx)
            for role, content, idx in self._turns
        ]

    def query(self) -> str:
        return self._query_text

    def evaluate(self, response: str) -> tuple[float, str]:
        return self._eval_fn(response)
