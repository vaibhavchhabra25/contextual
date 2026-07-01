#!/usr/bin/env python3
"""Run the benchmark matrix and print comparison tables per task.

Usage:
    python run_benchmark.py
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from harness.interface import CompressionStrategy, Task
from harness.models import EvalResult
from harness.runner import run_once
from harness.tokenizer import annotate, count_turns
from strategies.naive_truncation import NaiveTruncation
from strategies.rolling_summarization import RollingSummarization
from strategies.semantic_retrieval import SemanticRetrieval
from tasks.agentic_session_replay import AgenticSessionReplay
from tasks.instruction_persistence import InstructionPersistence
from tasks.multi_hop_qa import MultiHopQA
from tasks.needle_in_haystack import NeedleInHaystack

console = Console()


def run_task(task: Task, strategies: list[CompressionStrategy]) -> None:
    original_tokens = count_turns(annotate(task.build_context()))
    budget_fractions = [1.0, 0.75, 0.5, 0.25]
    budgets = [max(50, int(original_tokens * f)) for f in budget_fractions]

    console.print(f"\n[bold yellow]══ Task: {task.id} ══[/bold yellow]")
    console.print(f"Original context: {original_tokens} tokens")

    # Print task-specific info
    if hasattr(task, "_secret"):
        console.print(f"Needle at turn {task.needle_turn}  |  Secret: {task._secret}")
    if hasattr(task, "_hop_a"):
        console.print(f"Hop A (turn {task.hop_a_turn}): {task._hop_a}")
        console.print(f"Hop B (turn {task.hop_b_turn}): {task._hop_b}")
        console.print(f"Query: {task._query}  |  Answer: {task._answer}")
    if hasattr(task, "_instruction"):
        console.print(f"Instruction at turn {task.instruction_turn}: {task._instruction[:80]}…")
        console.print(f"Constraint check: {task._check_desc}")

    results: dict[str, list[EvalResult]] = {s.id: [] for s in strategies}
    for strategy in strategies:
        console.print(f"\n  [cyan]{strategy.id}[/cyan]")
        for budget in budgets:
            r = run_once(task, strategy, budget, verbose=True)
            results[strategy.id].append(r)

    # Comparison table
    table = Table(title=f"Results — {task.id}")
    table.add_column("Budget %", justify="right")
    table.add_column("Budget (tok)", justify="right")
    for s in strategies:
        table.add_column(s.id, justify="center")

    for i, (budget, frac) in enumerate(zip(budgets, budget_fractions)):
        row = [f"{frac:.0%}", str(budget)]
        for s in strategies:
            r = results[s.id][i]
            icon = "✓" if r.score == 1.0 else "✗"
            row.append(f"{icon} ({r.tokens_after_compression} tok)")
        table.add_row(*row)

    console.print()
    console.print(table)

    console.print("\n[bold]First failure point:[/bold]")
    for s in strategies:
        for r, frac in zip(results[s.id], budget_fractions):
            if r.score < 1.0:
                console.print(f"  {s.id}: fails at {frac:.0%}")
                break
        else:
            console.print(f"  {s.id}: [green]never fails[/green]")


def main() -> None:
    strategies: list[CompressionStrategy] = [
        NaiveTruncation(),
        RollingSummarization(keep_last=10),
        SemanticRetrieval(keep_last=3),
    ]

    tasks: list[Task] = [
        NeedleInHaystack(total_turns=40, needle_turn=10),
        MultiHopQA(total_turns=40, hop_a_turn=8, hop_b_turn=28),
        AgenticSessionReplay(),
        InstructionPersistence(total_turns=40, instruction_turn=2),
    ]

    for task in tasks:
        # For semantic retrieval, inject the task's query as the hint
        for s in strategies:
            if isinstance(s, SemanticRetrieval):
                s.query_hint = task.query()
        run_task(task, strategies)


if __name__ == "__main__":
    main()
