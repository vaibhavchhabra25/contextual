"""ctx log / diff / checkout — history inspection CLI for VersionedContextEngine.

Works like git log/diff/checkout but for context state over turns.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from engine.segment import SegmentStatus
from engine.store import SegmentStore

console = Console()

_STATUS_STYLE = {
    SegmentStatus.ACTIVE: "green",
    SegmentStatus.SUPERSEDED: "yellow",
    SegmentStatus.SUMMARIZED: "blue",
    SegmentStatus.DROPPED: "red",
}


# ── ctx log ────────────────────────────────────────────────────────────────────

def cmd_log(store: SegmentStore) -> None:
    """Show the history of context states, one row per turn."""
    if not store.snapshots:
        console.print("[dim]No history. Run a compression first.[/dim]")
        return

    table = Table(title="ctx log — context state history", show_lines=False)
    table.add_column("Turn", justify="right", style="cyan", width=6)
    table.add_column("Active", justify="right", width=8)
    table.add_column("Superseded", justify="right", width=12)
    table.add_column("Δ event", style="dim")

    prev_active: set[str] = set()
    for snap in store.snapshots:
        added = snap.active_ids - prev_active
        removed = prev_active - snap.active_ids
        delta_parts = []
        if added:
            delta_parts.append(f"[green]+{len(added)} seg[/green]")
        if removed:
            delta_parts.append(f"[red]-{len(removed)} seg[/red]")
        event_str = snap.event if snap.event else "  ".join(delta_parts) or "—"
        table.add_row(
            str(snap.turn_index),
            str(len(snap.active_ids)),
            str(len(snap.superseded_ids)),
            event_str,
        )
        prev_active = snap.active_ids

    console.print(table)

    # Summary legend
    segs = list(store.segments.values())
    by_status: dict[SegmentStatus, int] = {}
    for s in segs:
        by_status[s.status] = by_status.get(s.status, 0) + 1
    parts = [f"[{_STATUS_STYLE[k]}]{k.value}={v}[/{_STATUS_STYLE[k]}]"
             for k, v in by_status.items()]
    console.print("Final segment counts: " + "  ".join(parts))


# ── ctx diff ───────────────────────────────────────────────────────────────────

def cmd_diff(store: SegmentStore, turn_a: int, turn_b: int) -> None:
    """Show what changed between two turn snapshots."""
    snap_a = store.snapshot_at(turn_a)
    snap_b = store.snapshot_at(turn_b)

    if snap_a is None or snap_b is None:
        console.print(f"[red]Could not find snapshots for turns {turn_a} and {turn_b}.[/red]")
        return

    added_ids = snap_b.active_ids - snap_a.active_ids
    removed_ids = snap_a.active_ids - snap_b.active_ids
    newly_superseded = snap_b.superseded_ids - snap_a.superseded_ids
    kept_ids = snap_a.active_ids & snap_b.active_ids

    console.print(f"\n[bold]ctx diff turn{turn_a} → turn{turn_b}[/bold]\n")
    console.print(f"  Active segments:  {len(snap_a.active_ids)} → {len(snap_b.active_ids)}")
    console.print(f"  Superseded total: {len(snap_a.superseded_ids)} → {len(snap_b.superseded_ids)}\n")

    def _preview(seg_id: str, max_len: int = 80) -> str:
        seg = store.get(seg_id)
        if seg is None:
            return "(unknown)"
        preview = seg.content.replace("\n", " ")[:max_len]
        return preview + ("…" if len(seg.content) > max_len else "")

    if added_ids:
        console.print("[green bold]+ Added (newly active):[/green bold]")
        for sid in sorted(added_ids):
            seg = store.get(sid)
            tags = f" [{', '.join(seg.tags)}]" if seg and seg.tags else ""
            console.print(f"  [green]+[/green] [{sid}] turn={store.get(sid).created_turn}{tags}")
            console.print(f"      {_preview(sid)}")

    if removed_ids:
        console.print("\n[red bold]- Removed (no longer active):[/red bold]")
        for sid in sorted(removed_ids):
            seg = store.get(sid)
            status = f" ({seg.status.value})" if seg else ""
            console.print(f"  [red]-[/red] [{sid}] turn={store.get(sid).created_turn}{status}")
            console.print(f"      {_preview(sid)}")

    if newly_superseded:
        console.print("\n[yellow bold]~ Superseded in this range:[/yellow bold]")
        for sid in sorted(newly_superseded):
            seg = store.get(sid)
            by = f" → superseded_by={seg.superseded_by}" if seg and seg.superseded_by else ""
            console.print(f"  [yellow]~[/yellow] [{sid}] turn={store.get(sid).created_turn}{by}")
            console.print(f"      {_preview(sid)}")

    if not added_ids and not removed_ids and not newly_superseded:
        console.print("[dim]No changes between these turns.[/dim]")


# ── ctx checkout ───────────────────────────────────────────────────────────────

def cmd_checkout(store: SegmentStore, turn_index: int) -> list[str]:
    """Return the exact context (list of content strings) as it existed at turn_index."""
    snap = store.snapshot_at(turn_index)
    if snap is None:
        console.print(f"[red]No snapshot found at or before turn {turn_index}.[/red]")
        return []

    # Reconstruct in original insertion order
    contents: list[str] = []
    for seg_id in store._ordered_ids:
        if seg_id in snap.active_ids:
            seg = store.get(seg_id)
            if seg:
                contents.append(seg.content)

    console.print(f"\n[bold]ctx checkout turn{turn_index}[/bold]")
    console.print(f"[dim]{len(snap.active_ids)} active segments at this point[/dim]\n")

    for i, content in enumerate(contents):
        seg_id = [sid for sid in store._ordered_ids if sid in snap.active_ids][i]
        seg = store.get(seg_id)
        tags_str = f"  tags=[{', '.join(seg.tags)}]" if seg and seg.tags else ""
        turn_str = f"turn={seg.created_turn}" if seg else ""
        console.print(Panel(
            content[:300] + ("…" if len(content) > 300 else ""),
            title=f"[{seg_id}] {turn_str}{tags_str}",
            title_align="left",
            border_style="dim",
        ))

    return contents
