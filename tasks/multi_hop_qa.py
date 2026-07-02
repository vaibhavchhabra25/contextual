"""Multi-hop QA task.

Two facts are planted at separated turn indices.  Answering the query requires
combining both facts — neither alone is sufficient, and neither turn uses
the same vocabulary as the query (so embedding-based retrieval must get lucky
or fail).

Example hop chain:
  Turn A: "The lead engineer on Project Aurora is Marcus Webb."
  Turn B: "Marcus Webb's employee ID is 77291."
  Query:  "What is the employee ID of the lead engineer on Project Aurora?"

The answer (77291) appears only in Turn B, but Turn B is only meaningful
if the model also has Turn A to link "Marcus Webb" → "lead engineer on Aurora".
"""

from __future__ import annotations

import random

from harness.interface import Task
from harness.models import Turn

# Each entry: (hop_a_content, hop_b_content, query, answer)
# The query uses vocabulary from neither hop directly.
_HOP_SCENARIOS = [
    (
        "The lead engineer on Project Aurora is Marcus Webb.",
        "Marcus Webb's employee badge number is 77291.",
        "What is the employee badge number of the lead engineer on Project Aurora?",
        "77291",
    ),
    (
        "The capital of the fictional country Valdoria is Selmoor.",
        "Selmoor's population as of the last census was 2,847,391.",
        "What is the population of the capital of Valdoria?",
        "2,847,391",
    ),
    (
        "Our server configuration stores backups in the directory owned by the 'archiver' user.",
        "The 'archiver' user's home directory is /var/depot/arc.",
        "What directory path stores the server backups?",
        "/var/depot/arc",
    ),
]

_FILLER_TEMPLATES = [
    "The quarterly review meeting has been rescheduled to next Thursday.",
    "Please remember to update your time-tracking entries before end of day.",
    "The new parking policy takes effect starting next month.",
    "IT support will be conducting system maintenance this weekend.",
    "The office kitchen will be closed for cleaning on Friday afternoon.",
    "All expense reports must be submitted through the updated portal.",
    "The fire drill scheduled for Wednesday has been postponed.",
    "New ergonomic chair requests should go through facilities management.",
    "The team building event has been confirmed for the last Friday of the month.",
    "Please ensure your VPN client is updated to the latest version.",
    "Badge access to the east wing will require re-registration by month end.",
    "The library of shared assets has been moved to the new intranet location.",
]


num_scenarios = len(_HOP_SCENARIOS)


class MultiHopQA(Task):
    """Plant two chained facts at separated turn indices in filler context."""

    id = "multi_hop_qa"

    def __init__(
        self,
        total_turns: int = 40,
        hop_a_turn: int = 8,
        hop_b_turn: int = 28,
        scenario_index: int = 0,
        seed: int = 7,
    ) -> None:
        self.total_turns = total_turns
        self.hop_a_turn = hop_a_turn
        self.hop_b_turn = hop_b_turn
        self._seed = seed
        self._rng = random.Random(seed)
        scenario = _HOP_SCENARIOS[scenario_index % len(_HOP_SCENARIOS)]
        self._hop_a, self._hop_b, self._query, self._answer = scenario

    def build_context(self) -> list[Turn]:
        self._rng = random.Random(self._seed)  # reset so every call is identical
        fillers = list(self._rng.sample(_FILLER_TEMPLATES, len(_FILLER_TEMPLATES)))
        filler_idx = 0
        turns: list[Turn] = []
        for i in range(self.total_turns):
            if i == self.hop_a_turn:
                content = self._hop_a
            elif i == self.hop_b_turn:
                content = self._hop_b
            else:
                content = fillers[filler_idx % len(fillers)]
                filler_idx += 1
            role = "user" if i % 2 == 0 else "assistant"
            turns.append(Turn(role=role, content=content, turn_index=i))
        return turns

    def query(self) -> str:
        return self._query + " Answer with only the specific value, nothing else."

    def evaluate(self, response: str) -> tuple[float, str]:
        answer_clean = self._answer.replace(",", "").lower()
        response_clean = response.strip().replace(",", "").lower()
        score = 1.0 if answer_clean in response_clean else 0.0
        return score, "exact_match"
