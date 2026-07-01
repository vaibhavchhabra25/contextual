#!/usr/bin/env python3
"""ctx — history inspection CLI for the versioned context engine.

Usage:
    python ctx.py log [--task needle_in_haystack]
    python ctx.py diff <turn_a> <turn_b> [--task ...]
    python ctx.py checkout <turn> [--task ...]

The CLI runs a task through the versioned engine first, then lets you inspect
the resulting context history.

Examples:
    python ctx.py log
    python ctx.py diff 5 20
    python ctx.py checkout 12
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from engine.cli import cmd_checkout, cmd_diff, cmd_log
from engine.strategy import VersionedContextEngine
from harness.tokenizer import annotate, count_turns
from tasks.agentic_session_replay import AgenticSessionReplay
from tasks.instruction_persistence import InstructionPersistence
from tasks.multi_hop_qa import MultiHopQA
from tasks.needle_in_haystack import NeedleInHaystack

console = Console()

_TASKS = {
    "needle_in_haystack": NeedleInHaystack(total_turns=40, needle_turn=10),
    "multi_hop_qa": MultiHopQA(total_turns=40, hop_a_turn=8, hop_b_turn=28),
    "agentic_session_replay": AgenticSessionReplay(),
    "instruction_persistence": InstructionPersistence(total_turns=40, instruction_turn=2),
}


def _build_engine(task_name: str) -> VersionedContextEngine:
    task = _TASKS[task_name]
    history = annotate(task.build_context())
    total_tokens = count_turns(history)

    engine = VersionedContextEngine(recency_window=8, verbose=False)
    # Run at full budget just to populate the store with history
    engine.compress(history, budget=total_tokens)

    console.print(
        f"[dim]Loaded task [bold]{task_name}[/bold] — "
        f"{len(history)} turns, {total_tokens} tokens[/dim]\n"
    )
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(prog="ctx", description="Context history inspector")
    parser.add_argument("command", choices=["log", "diff", "checkout"])
    parser.add_argument("args", nargs="*", help="turn indices for diff/checkout")
    parser.add_argument(
        "--task",
        default="needle_in_haystack",
        choices=list(_TASKS.keys()),
        help="Which task to load (default: needle_in_haystack)",
    )
    opts = parser.parse_args()

    engine = _build_engine(opts.task)
    store = engine.store

    if opts.command == "log":
        cmd_log(store)

    elif opts.command == "diff":
        if len(opts.args) < 2:
            console.print("[red]diff requires two turn indices: ctx diff <turn_a> <turn_b>[/red]")
            sys.exit(1)
        cmd_diff(store, int(opts.args[0]), int(opts.args[1]))

    elif opts.command == "checkout":
        if not opts.args:
            console.print("[red]checkout requires a turn index: ctx checkout <turn>[/red]")
            sys.exit(1)
        cmd_checkout(store, int(opts.args[0]))


if __name__ == "__main__":
    main()
