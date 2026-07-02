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
from tasks.agentic_session_replay import num_scenarios as _asr_n
from tasks.instruction_persistence import InstructionPersistence
from tasks.instruction_persistence import num_scenarios as _ip_n
from tasks.multi_hop_qa import MultiHopQA
from tasks.multi_hop_qa import num_scenarios as _mhq_n
from tasks.needle_in_haystack import NeedleInHaystack
from tasks.needle_in_haystack import num_scenarios as _nih_n

console = Console()


def run_task(
    task_id: str,
    scenarios: list[Task],
    strategies: list[CompressionStrategy],
    seeds: int = 1,
) -> list[EvalResult]:
    """Run all scenarios × strategies × budgets for one task type.

    Scores are averaged across scenarios (and seeds) before printing.
    Returns flat list of averaged EvalResults.
    """
    budget_fractions = [1.0, 0.75, 0.5, 0.25]

    console.print(f"\n[bold yellow]══ Task: {task_id} ({len(scenarios)} scenario(s)) ══[/bold yellow]")

    # key: (strategy_id, budget_fraction_index) → flat list of scores from all scenarios × seeds
    accumulated: dict[tuple[str, int], list[float]] = {}
    representative: dict[tuple[str, int], EvalResult] = {}

    for sc_idx, task in enumerate(scenarios):
        original_tokens = count_turns(annotate(task.build_context()))
        budgets = [max(50, int(original_tokens * f)) for f in budget_fractions]
        console.print(f"\n  [dim]Scenario {sc_idx}: {original_tokens} tok[/dim]")

        if hasattr(task, "_secret"):
            console.print(f"    Needle @ turn {task.needle_turn}  |  Secret: {task._secret}")
        if hasattr(task, "_hop_a"):
            console.print(f"    Hop A (t{task.hop_a_turn}): {task._hop_a}")
            console.print(f"    Hop B (t{task.hop_b_turn}): {task._hop_b}")
            console.print(f"    Answer: {task._answer}")
        if hasattr(task, "_instruction"):
            console.print(f"    Instruction @ turn {task.instruction_turn}: {task._instruction[:60]}…")

        for strategy in strategies:
            if isinstance(strategy, SemanticRetrieval):
                strategy.query_hint = task.query()
            console.print(f"    [cyan]{strategy.id}[/cyan]")
            for fi, (budget, frac) in enumerate(zip(budgets, budget_fractions)):
                key = (strategy.id, fi)
                if key not in accumulated:
                    accumulated[key] = []
                for seed in range(seeds):
                    r = run_once(task, strategy, budget, verbose=(seed == 0 and sc_idx == 0))
                    accumulated[key].append(r.score)
                    if key not in representative:
                        representative[key] = r

    # Build averaged EvalResults
    all_results: list[EvalResult] = []
    final: dict[str, list[EvalResult]] = {s.id: [] for s in strategies}
    n_scenarios = len(scenarios)

    for strategy in strategies:
        for fi, frac in enumerate(budget_fractions):
            key = (strategy.id, fi)
            scores = accumulated[key]
            rep = representative[key]
            avg_score = statistics.mean(scores)
            label_suffix = ""
            if n_scenarios > 1:
                label_suffix += f"_sc{n_scenarios}"
            if seeds > 1:
                label_suffix += f"_avg{seeds}"
            averaged = EvalResult(
                task_id=task_id,
                strategy_id=strategy.id,
                token_budget=rep.token_budget,
                tokens_original=rep.tokens_original,
                tokens_after_compression=rep.tokens_after_compression,
                score=avg_score,
                score_label=rep.score_label + label_suffix,
                turn_token_log=rep.turn_token_log,
                notes=f"scenarios={n_scenarios} seeds={seeds}",
            )
            final[strategy.id].append(averaged)
            all_results.append(averaged)

    # Print comparison table
    table = Table(title=f"Results — {task_id}")
    table.add_column("Budget %", justify="right")
    table.add_column("Budget (tok)", justify="right")
    for s in strategies:
        table.add_column(s.id, justify="center")

    for i, frac in enumerate(budget_fractions):
        row = [f"{frac:.0%}", str(final[strategies[0].id][i].token_budget)]
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
                console.print(f"  {s.id}: fails at {frac:.0%} (score={r.score:.2f})")
                break
        else:
            console.print(f"  {s.id}: [green]never fails[/green]")

    return all_results


def main() -> None:
    _TASK_NAMES = ["needle_in_haystack", "multi_hop_qa", "agentic_session_replay", "instruction_persistence"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", metavar="PATH", help="Save results to JSON file")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to average over")
    parser.add_argument("--plot", action="store_true", help="Generate charts after saving JSON")
    parser.add_argument(
        "--task", metavar="NAME", action="append", dest="tasks",
        choices=_TASK_NAMES, help=f"Run only this task (repeatable). Choices: {_TASK_NAMES}",
    )
    opts = parser.parse_args()

    strategies: list[CompressionStrategy] = [
        NaiveTruncation(),
        RollingSummarization(keep_last=10),
        SemanticRetrieval(keep_last=3),
        VersionedContextEngine(recency_window=8),
    ]

    # Each task type is run across all its scenarios; scores are averaged.
    task_groups: list[tuple[str, list[Task]]] = [
        (
            "needle_in_haystack",
            [NeedleInHaystack(total_turns=40, scenario_index=i) for i in range(_nih_n)],
        ),
        (
            "multi_hop_qa",
            [MultiHopQA(total_turns=40, hop_a_turn=8, hop_b_turn=28, scenario_index=i) for i in range(_mhq_n)],
        ),
        (
            "agentic_session_replay",
            [AgenticSessionReplay(scenario_index=i) for i in range(_asr_n)],
        ),
        (
            "instruction_persistence",
            [InstructionPersistence(total_turns=40, instruction_turn=2, scenario_index=i) for i in range(_ip_n)],
        ),
    ]

    filter_tasks = set(opts.tasks) if opts.tasks else None

    all_results: list[EvalResult] = []
    for task_id, scenarios in task_groups:
        if filter_tasks and task_id not in filter_tasks:
            continue
        all_results.extend(run_task(task_id, scenarios, strategies, seeds=opts.seeds))

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
