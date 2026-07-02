#!/usr/bin/env python3
"""Generate accuracy-vs-token-budget curve plots from saved benchmark results.

Usage:
    python run_benchmark.py --json results/run.json   # save results first
    python plot_results.py results/run.json           # then plot
    python plot_results.py results/run.json --out charts/  # custom output dir
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_STRATEGY_STYLES: dict[str, dict] = {
    "naive_truncation":     {"color": "#e05252", "marker": "o",  "ls": "--"},
    "rolling_summarization":{"color": "#e08c52", "marker": "s",  "ls": "-."},
    "semantic_retrieval":   {"color": "#5294e0", "marker": "^",  "ls": "-"},
    "versioned_engine":     {"color": "#52c47a", "marker": "D",  "ls": "-"},
}
_DEFAULT_STYLE = {"color": "#999999", "marker": "x", "ls": ":"}


def _style(strategy_id: str) -> dict:
    return _STRATEGY_STYLES.get(strategy_id, _DEFAULT_STYLE)


def plot(results: list[dict], out_dir: Path) -> list[Path]:
    """Plot one chart per task. Returns list of saved file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r)

    saved: list[Path] = []
    for task_id, task_results in by_task.items():
        # Group by strategy within this task
        by_strategy: dict[str, list[dict]] = defaultdict(list)
        for r in task_results:
            by_strategy[r["strategy_id"]].append(r)

        fig, ax = plt.subplots(figsize=(7, 4.5))

        for strategy_id, runs in by_strategy.items():
            # Sort by budget fraction ascending
            runs = sorted(runs, key=lambda r: r["token_budget"])
            fractions = [r["tokens_after_compression"] / r["tokens_original"] for r in runs]
            scores = [r["score"] for r in runs]
            style = _style(strategy_id)
            ax.plot(
                fractions, scores,
                color=style["color"], marker=style["marker"],
                ls=style["ls"], linewidth=2, markersize=7,
                label=strategy_id.replace("_", " "),
            )

        ax.set_xlabel("Compression ratio (tokens after / tokens original)", fontsize=11)
        ax.set_ylabel("Score (0 = fail, 1 = pass)", fontsize=11)
        ax.set_title(f"Accuracy vs. compression — {task_id.replace('_', ' ')}", fontsize=12)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.05, 1.15)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()

        path = out_dir / f"{task_id}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
        print(f"  saved {path}")

    # Combined overview chart (all tasks as subplots)
    tasks = list(by_task.keys())
    ncols = 2
    nrows = (len(tasks) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for i, task_id in enumerate(tasks):
        ax = axes[i]
        by_strategy = defaultdict(list)
        for r in by_task[task_id]:
            by_strategy[r["strategy_id"]].append(r)

        for strategy_id, runs in by_strategy.items():
            runs = sorted(runs, key=lambda r: r["token_budget"])
            fractions = [r["tokens_after_compression"] / r["tokens_original"] for r in runs]
            scores = [r["score"] for r in runs]
            style = _style(strategy_id)
            ax.plot(fractions, scores,
                    color=style["color"], marker=style["marker"],
                    ls=style["ls"], linewidth=2, markersize=6,
                    label=strategy_id.replace("_", " "))

        ax.set_title(task_id.replace("_", " "), fontsize=11)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.05, 1.15)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.set_major_locator(mticker.FixedLocator([0, 0.5, 1.0]))
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlabel("Compression ratio", fontsize=9)
        ax.set_ylabel("Score", fontsize=9)

    # Hide unused subplots
    for j in range(len(tasks), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy vs. compression ratio — all tasks", fontsize=13, y=1.01)
    fig.tight_layout()

    overview_path = out_dir / "overview.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(overview_path)
    print(f"  saved {overview_path}")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_json", help="Path to results JSON file")
    parser.add_argument("--out", default="results/charts", help="Output directory for charts")
    opts = parser.parse_args()

    with open(opts.results_json) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {opts.results_json}")
    saved = plot(results, Path(opts.out))
    print(f"\n{len(saved)} charts saved to {opts.out}/")


if __name__ == "__main__":
    main()
