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
from tasks.agentic_session_replay import num_scenarios as _asr_n
from tasks.instruction_persistence import InstructionPersistence
from tasks.instruction_persistence import num_scenarios as _ip_n
from tasks.multi_hop_qa import MultiHopQA
from tasks.multi_hop_qa import num_scenarios as _mhq_n
from tasks.needle_in_haystack import NeedleInHaystack
from tasks.needle_in_haystack import num_scenarios as _nih_n

console = Console()

_TASK_NAMES = ["needle_in_haystack", "multi_hop_qa", "agentic_session_replay", "instruction_persistence"]
_NUM_SCENARIOS = {
    "needle_in_haystack": _nih_n,
    "multi_hop_qa": _mhq_n,
    "agentic_session_replay": _asr_n,
    "instruction_persistence": _ip_n,
}


def _make_task(task_name: str, scenario: int):
    if task_name == "needle_in_haystack":
        return NeedleInHaystack(total_turns=40, scenario_index=scenario)
    if task_name == "multi_hop_qa":
        return MultiHopQA(total_turns=40, hop_a_turn=8, hop_b_turn=28, scenario_index=scenario)
    if task_name == "agentic_session_replay":
        return AgenticSessionReplay(scenario_index=scenario)
    return InstructionPersistence(total_turns=40, instruction_turn=2, scenario_index=scenario)


def _build_engine(task_name: str, scenario: int) -> VersionedContextEngine:
    task = _make_task(task_name, scenario)
    history = annotate(task.build_context())
    total_tokens = count_turns(history)

    engine = VersionedContextEngine(recency_window=8, verbose=False)
    # Run at full budget just to populate the store with history
    engine.compress(history, budget=total_tokens)

    console.print(
        f"[dim]Loaded task [bold]{task_name}[/bold] scenario {scenario} — "
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
        choices=_TASK_NAMES,
        help="Which task to load (default: needle_in_haystack)",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=0,
        help="Scenario index within the task (default: 0)",
    )
    opts = parser.parse_args()

    n = _NUM_SCENARIOS[opts.task]
    if opts.scenario >= n:
        console.print(f"[red]--scenario {opts.scenario} out of range; {opts.task} has {n} scenario(s) (0–{n-1})[/red]")
        sys.exit(1)

    engine = _build_engine(opts.task, opts.scenario)
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
