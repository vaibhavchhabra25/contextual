"""Instruction persistence task.

An early turn sets a hard constraint (e.g. "always respond in French",
"never use the word 'the'", "prefix every answer with [CONFIRMED]").
After N turns of filler conversation, the model is asked a plain question.
The evaluator checks whether the constraint is still obeyed.

This stresses strategies differently from needle-in-haystack: the instruction
turn is imperative (not a factual nugget), and obeying it requires the model
to carry forward a *behavioral* rule, not just recall a value.
"""

from __future__ import annotations

import random

from harness.interface import Task
from harness.models import Turn

_FILLER_QA = [
    ("What is the boiling point of water?", "Water boils at 100°C (212°F) at sea level."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("What is the speed of light?", "The speed of light is approximately 299,792 km/s."),
    ("How many continents are there?", "There are seven continents on Earth."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
    ("How many days are in a leap year?", "A leap year has 366 days."),
    ("What is the tallest mountain?", "Mount Everest is the tallest mountain at 8,849 m."),
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the telephone."),
    ("What is photosynthesis?", "Photosynthesis is how plants convert sunlight and CO2 into energy and oxygen."),
    ("What planet is closest to the Sun?", "Mercury is the closest planet to the Sun."),
]

# Each scenario: (instruction, final_query, check_fn description, check keyword)
_SCENARIOS = [
    {
        "instruction": (
            "IMPORTANT SYSTEM RULE: For the remainder of this conversation, "
            "you must prefix every single response with the token [CONFIRMED] "
            "before any other text. This applies to all future responses without exception."
        ),
        "query": "What is the capital of France?",
        "check": lambda r: r.strip().startswith("[CONFIRMED]"),
        "check_desc": "response starts with [CONFIRMED]",
    },
    {
        "instruction": (
            "IMPORTANT SYSTEM RULE: For the remainder of this conversation, "
            "respond only in French, regardless of what language questions are asked in."
        ),
        "query": "What is 7 multiplied by 8?",
        # Simple heuristic: French responses typically contain these common words
        "check": lambda r: any(w in r.lower() for w in ["est", "le", "la", "les", "de", "du", "sept", "huit", "cinquante", "résultat", "réponse"]),
        "check_desc": "response is in French",
    },
    {
        "instruction": (
            "IMPORTANT SYSTEM RULE: For the remainder of this conversation, "
            "end every response with the exact phrase: END OF RESPONSE"
        ),
        "query": "Name one programming language.",
        "check": lambda r: "END OF RESPONSE" in r,
        "check_desc": "response ends with END OF RESPONSE",
    },
]


class InstructionPersistence(Task):
    """Plant an instruction constraint early; verify it survives compression."""

    id = "instruction_persistence"

    def __init__(
        self,
        total_turns: int = 40,
        instruction_turn: int = 2,
        scenario_index: int = 0,
        seed: int = 13,
    ) -> None:
        self.total_turns = total_turns
        self.instruction_turn = instruction_turn
        self._seed = seed
        self._rng = random.Random(seed)
        scenario = _SCENARIOS[scenario_index % len(_SCENARIOS)]
        self._instruction = scenario["instruction"]
        self._query = scenario["query"]
        self._check = scenario["check"]
        self._check_desc = scenario["check_desc"]

    def build_context(self) -> list[Turn]:
        self._rng = random.Random(self._seed)  # reset so every call is identical
        filler = list(self._rng.sample(_FILLER_QA, len(_FILLER_QA)))
        filler_idx = 0
        turns: list[Turn] = []

        for i in range(self.total_turns):
            if i == self.instruction_turn:
                turns.append(Turn(
                    role="user",
                    content=self._instruction,
                    turn_index=i,
                ))
            elif i == self.instruction_turn + 1:
                # Model acknowledges the instruction
                turns.append(Turn(
                    role="assistant",
                    content="[CONFIRMED] Understood. I will follow that rule for all future responses.",
                    turn_index=i,
                ))
            else:
                q, a = filler[filler_idx % len(filler)]
                filler_idx += 1
                role = "user" if i % 2 == 0 else "assistant"
                turns.append(Turn(role=role, content=q if role == "user" else a, turn_index=i))

        return turns

    def query(self) -> str:
        return self._query

    def evaluate(self, response: str) -> tuple[float, str]:
        passed = self._check(response)
        return (1.0 if passed else 0.0), f"constraint:{self._check_desc}"
