"""Agentic session replay task.

Replays a synthetic multi-turn coding session (file reads, edits, test runs).
After compression, the model is asked to produce the final correct version of
the file.  Evaluation checks whether the output passes a set of test assertions.

The session narrative:
  - Early turns establish a Python function with a bug.
  - Middle turns show tool calls: reading the file, running tests (failing).
  - A key fix turn patches the bug.
  - Later turns add an unrelated feature and run tests again (passing).
  - Query: "Write the final correct implementation of the function."

Correct answer must implement both the fix AND the feature.
"""

from __future__ import annotations

from harness.interface import Task
from harness.models import Turn

# The ground-truth final implementation the model should reproduce
_CORRECT_IMPL = """\
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

_SESSION_TURNS = [
    # Turn 0 — initial file content (buggy: uses >= instead of >)
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
    # Turn 2 — run tests (fail)
    ("user", "tool:run_tests tests/test_utils.py", 2),
    ("assistant", """\
FAILED tests/test_utils.py::test_threshold_is_exclusive
  AssertionError: expected count=2 for items=[10,15,20] threshold=10, got count=3
  The boundary item (10) should be excluded.""", 3),
    # Turn 4 — agent reflects
    ("user", "The test expects strict greater-than. Let me fix the comparison.", 4),
    ("assistant", "Agreed. The filter should use `x > threshold` not `x >= threshold`.", 5),
    # Turn 6 — unrelated discussion (noise)
    ("user", "Also, the product team wants a 'max' field added to the return dict.", 6),
    ("assistant", "Got it. I'll add `\"max\": max(filtered) if filtered else None` to the return.", 7),
    # Turn 8 — filler
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
    # Turn 10 — THE KEY FIX (both bug fix + new feature applied)
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
    # Turn 12 — run tests again (pass)
    ("user", "tool:run_tests tests/test_utils.py", 12),
    ("assistant", "All 3 tests passed. ✓", 13),
    # Turns 14–19: unrelated noise about deployment / other files
    ("user", "Let's also check the deployment config.", 14),
    ("assistant", "Sure, reading deploy.yaml now.", 15),
    ("user", "tool:read_file deploy.yaml", 16),
    ("assistant", "deploy.yaml looks fine — no changes needed.", 17),
    ("user", "Great. Can you also confirm the version in setup.cfg?", 18),
    ("assistant", "setup.cfg shows version=1.4.2. Ready to tag the release.", 19),
]


class AgenticSessionReplay(Task):
    """Replay a coding session; check if model can reproduce the final correct code."""

    id = "agentic_session_replay"

    def build_context(self) -> list[Turn]:
        return [
            Turn(role=role, content=content, turn_index=idx)
            for role, content, idx in _SESSION_TURNS
        ]

    def query(self) -> str:
        return (
            "Based on the session above, write the final correct Python implementation "
            "of the `process_items` function. Output only the function code, no explanation."
        )

    def evaluate(self, response: str) -> tuple[float, str]:
        # Run the response as code and check key properties
        score = 0.0
        checks_passed = 0
        total_checks = 4

        namespace: dict = {}
        try:
            # Extract code block if wrapped in markdown fences
            code = response
            if "```" in code:
                lines = code.split("\n")
                in_block = False
                extracted = []
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

            # Check 1: threshold is exclusive (the bug fix)
            r = fn([10, 15, 20], threshold=10)
            if r.get("count") == 2:
                checks_passed += 1

            # Check 2: total is correct
            if r.get("total") == 35:
                checks_passed += 1

            # Check 3: max field exists and is correct
            if r.get("max") == 20:
                checks_passed += 1

            # Check 4: empty case
            r_empty = fn([1, 2, 3], threshold=10)
            if r_empty.get("count") == 0 and r_empty.get("max") is None:
                checks_passed += 1

        except Exception:
            return 0.0, "exec_error"

        score = checks_passed / total_checks
        return score, f"exec_checks_{checks_passed}/{total_checks}"
