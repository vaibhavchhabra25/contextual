"""Needle-in-haystack task.

A secret code is planted at a configurable depth (turn index) inside a long
conversation of filler turns.  The evaluator checks exact-match retrieval.
"""

from __future__ import annotations

import random
import string

from harness.interface import Task
from harness.models import Turn

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
    """Plant a secret code at *needle_turn* inside *total_turns* filler turns."""

    id = "needle_in_haystack"

    def __init__(
        self,
        total_turns: int = 40,
        needle_turn: int = 10,
        seed: int = 42,
    ) -> None:
        self.total_turns = total_turns
        self.needle_turn = needle_turn
        self._rng = random.Random(seed)
        self._secret = self._make_secret()

    def _make_secret(self) -> str:
        chars = string.ascii_uppercase + string.digits
        return "".join(self._rng.choices(chars, k=8))

    def build_context(self) -> list[Turn]:
        turns: list[Turn] = []
        for i in range(self.total_turns):
            if i == self.needle_turn:
                content = (
                    f"Important: remember this secret code for later: {self._secret}. "
                    "Do not forget it — you will be asked to recall it."
                )
                turns.append(Turn(role="user", content=content, turn_index=i))
            else:
                turns.append(_filler_turn(i, self._rng))
        return turns

    def query(self) -> str:
        return "What was the secret code mentioned earlier in our conversation? Respond with only the code, nothing else."

    def evaluate(self, response: str) -> tuple[float, str]:
        # Exact match after stripping whitespace/punctuation
        cleaned = response.strip().strip(".,!?\"'").upper()
        score = 1.0 if self._secret in cleaned else 0.0
        return score, "exact_match"
