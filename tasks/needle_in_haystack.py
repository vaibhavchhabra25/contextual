"""Needle-in-haystack task.

A secret value is planted at a configurable depth (turn index) inside a long
conversation of filler turns.  The evaluator checks exact-match retrieval.

Three scenarios vary the needle position and secret format:
  0 — alphanumeric code at turn 10  (middle-early)
  1 — digit-only PIN at turn 5      (very early)
  2 — alphanumeric code at turn 30  (very late)
"""

from __future__ import annotations

import random
import string

from harness.interface import Task
from harness.models import Turn

# Each scenario: (needle_turn, needle_template, secret_charset, secret_length, query)
_SCENARIOS = [
    (
        10,
        "Important: remember this secret code for later: {secret}. "
        "Do not forget it — you will be asked to recall it.",
        string.ascii_uppercase + string.digits,
        8,
        "What was the secret code mentioned earlier in our conversation? "
        "Respond with only the code, nothing else.",
    ),
    (
        5,
        "The access PIN for the secure vault is {secret}. "
        "Keep this confidential — you will need to recall it later.",
        string.digits,
        6,
        "What is the secure vault PIN that was shared earlier? "
        "Respond with only the PIN digits, nothing else.",
    ),
    (
        30,
        "Final reminder: the project passphrase has been set to {secret}. "
        "This is confidential — do not share it, but remember it for verification.",
        string.ascii_uppercase + string.digits,
        8,
        "What was the project passphrase mentioned in our conversation? "
        "Respond with only the passphrase, nothing else.",
    ),
]

num_scenarios = len(_SCENARIOS)

_FILLER_TEMPLATES = [
    "The weather today is {adj} with a chance of {noun}.",
    "I was reading about {noun} and found it quite {adj}.",
    "Did you know that {noun} can be {adj} in certain conditions?",
    "My colleague mentioned something about {adj} {noun} yesterday.",
    "Experts say that {noun} is becoming increasingly {adj}.",
]
_ADJECTIVES = ["interesting", "surprising", "common", "unusual", "complex", "simple", "rare"]
_NOUNS = ["history", "science", "art", "technology", "nature", "culture", "philosophy"]


def _filler_turn(i: int, rng: random.Random) -> Turn:
    template = rng.choice(_FILLER_TEMPLATES)
    text = template.format(adj=rng.choice(_ADJECTIVES), noun=rng.choice(_NOUNS))
    return Turn(role="user" if i % 2 == 0 else "assistant", content=text, turn_index=i)


class NeedleInHaystack(Task):
    """Plant a secret value at *needle_turn* inside *total_turns* filler turns."""

    id = "needle_in_haystack"

    def __init__(
        self,
        total_turns: int = 40,
        scenario_index: int = 0,
        seed: int = 42,
    ) -> None:
        self.total_turns = total_turns
        self._seed = seed
        scenario = _SCENARIOS[scenario_index % len(_SCENARIOS)]
        self.needle_turn, self._template, charset, length, self._query_text = scenario
        rng = random.Random(seed)
        self._secret = "".join(rng.choices(charset, k=length))

    def build_context(self) -> list[Turn]:
        rng = random.Random(self._seed)
        turns: list[Turn] = []
        for i in range(self.total_turns):
            if i == self.needle_turn:
                content = self._template.format(secret=self._secret)
                turns.append(Turn(role="user", content=content, turn_index=i))
            else:
                turns.append(_filler_turn(i, rng))
        return turns

    def query(self) -> str:
        return self._query_text

    def evaluate(self, response: str) -> tuple[float, str]:
        cleaned = response.strip().strip(".,!?\"'").upper()
        score = 1.0 if self._secret.upper() in cleaned else 0.0
        return score, "exact_match"
