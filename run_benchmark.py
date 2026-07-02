#!/usr/bin/env python3
"""Run the benchmark matrix and print comparison tables per task.

Usage:
    python run_benchmark.py                        # print tables only
    python run_benchmark.py --json results/run.json  # also save JSON
    python run_benchmark.py --seeds 3              # average over 3 seeds
    python run_benchmark.py --json results/run.json --plot  # save + plot
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from engine.strategy import VersionedContextEngine
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


def run_task(
    task: Task,
    strategies: list[CompressionStrategy],
    seeds: int = 1,
) -> list[EvalResult]:
    """Run all strategies × budgets for one task. Returns flat list of EvalResults."""
    original_tokens = count_turns(annotate(task.build_context()))
    budget_fractions = [1.0, 0.75, 0.5, 0.25]
    budgets = [max(50, int(original_tokens * f)) for f in budget_fractions]

    console.print(f"\n[bold yellow]══ Task: {task.id} ══[/bold yellow]")
    console.print(f"Original context: {original_tokens} tokens")

    if hasattr(task, "_secret"):
        console.print(f"Needle at turn {task.needle_turn}  |  Secret: {task._secret}")
    if hasattr(task, "_hop_a"):
        console.print(f"Hop A (turn {task.hop_a_turn}): {task._hop_a}")
        console.print(f"Hop B (turn {task.hop_b_turn}): {task._hop_b}")
        console.print(f"Query: {task._query}  |  Answer: {task._answer}")
    if hasattr(task, "_instruction"):
        console.print(f"Instruction at turn {task.instruction_turn}: {task._instruction[:80]}…")
        console.print(f"Constraint check: {task._check_desc}")

    all_results: list[EvalResult] = []

    # Accumulate scores across seeds then average
    # key: (strategy_id, budget) → list of EvalResult (one per seed)
    seed_results: dict[tuple[str, int], list[EvalResult]] = {}

    for strategy in strategies:
        console.print(f"\n  [cyan]{strategy.id}[/cyan]")
        for budget in budgets:
            key = (strategy.id, budget)
            seed_results[key] = []
            for seed in range(seeds):
                r = run_once(task, strategy, budget, verbose=(seed == 0))
                seed_results[key].append(r)

    # Average across seeds and build final results
    final: dict[str, list[EvalResult]] = {s.id: [] for s in strategies}
    for strategy in strategies:
        for budget, frac in zip(budgets, budget_fractions):
            key = (strategy.id, budget)
            runs = seed_results[key]
            avg_score = statistics.mean(r.score for r in runs)
            representative = runs[0]
            averaged = EvalResult(
                task_id=representative.task_id,
                strategy_id=representative.strategy_id,
                token_budget=budget,
                tokens_original=representative.tokens_original,
                tokens_after_compression=representative.tokens_after_compression,
                score=avg_score,
                score_label=representative.score_label + (f"_avg{seeds}" if seeds > 1 else ""),
                turn_token_log=representative.turn_token_log,
                notes=f"seeds={seeds}",
            )
            final[strategy.id].append(averaged)
            all_results.append(averaged)

    # Print comparison table
    table = Table(title=f"Results — {task.id}")
    table.add_column("Budget %", justify="right")
    table.add_column("Budget (tok)", justify="right")
    for s in strategies:
        table.add_column(s.id, justify="center")

    for i, (budget, frac) in enumerate(zip(budgets, budget_fractions)):
        row = [f"{frac:.0%}", str(budget)]
        for s in strategies:
            r = final[s.id][i]
            icon = "✓" if r.score == 1.0 else ("~" if r.score > 0 else "✗")
            row.append(f"{icon} ({r.tokens_after_compression} tok)")
        table.add_row(*row)

    console.print()
    console.print(table)

    console.print("\n[bold]First failure point:[/bold]")
    for s in strategies:
        for r, frac in zip(final[s.id], budget_fractions):
            if r.score < 1.0:
                console.print(f"  {s.id}: fails at {frac:.0%}")
                break
        else:
            console.print(f"  {s.id}: [green]never fails[/green]")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", metavar="PATH", help="Save results to JSON file")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to average over")
    parser.add_argument("--plot", action="store_true", help="Generate charts after saving JSON")
    opts = parser.parse_args()

    strategies: list[CompressionStrategy] = [
        NaiveTruncation(),
        RollingSummarization(keep_last=10),
        SemanticRetrieval(keep_last=3),
        VersionedContextEngine(recency_window=8),
    ]

    tasks: list[Task] = [
        NeedleInHaystack(total_turns=40, needle_turn=10),
        MultiHopQA(total_turns=40, hop_a_turn=8, hop_b_turn=28),
        AgenticSessionReplay(),
        InstructionPersistence(total_turns=40, instruction_turn=2),
    ]

    all_results: list[EvalResult] = []
    for task in tasks:
        for s in strategies:
            if isinstance(s, SemanticRetrieval):
                s.query_hint = task.query()
        all_results.extend(run_task(task, strategies, seeds=opts.seeds))

    if opts.json:
        out = Path(opts.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        console.print(f"\n[dim]Results saved to {out}[/dim]")

        if opts.plot:
            from plot_results import plot
            charts = plot([asdict(r) for r in all_results], out.parent / "charts")
            console.print(f"[dim]{len(charts)} charts saved[/dim]")


if __name__ == "__main__":
    main()
