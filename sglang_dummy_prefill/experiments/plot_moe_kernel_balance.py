#!/usr/bin/env python3
"""Visualize MoE kernel balance data from recorder dumps.

moe_times:          [decode_steps, world_size]          — per-step total forward time (ms)
local_token_counts: [decode_steps, num_layers, world_size] — per-layer local token counts
batch_sizes:        [decode_steps, world_size]           — batch size per rank per step
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_data(path: str):
    data = torch.load(path, map_location="cpu", weights_only=False)
    moe_times = data["moe_times"].cpu().float().numpy()
    local_token_counts = data["local_token_counts"].cpu().int().numpy()
    timestamps = data.get("timestamps")
    if timestamps is not None:
        timestamps = timestamps.cpu().double().numpy()

    phase_times = {}
    for key in ("attn_times", "ag_times", "ar_times", "fwd_times"):
        t = data.get(key)
        if t is not None:
            phase_times[key] = t.cpu().float().numpy()

    print(f"Loaded: {path}")
    print(f"  moe_times shape    = {moe_times.shape} (steps, layers, ranks)")
    print(f"  local_token_counts = {local_token_counts.shape} (steps, layers, ranks)")
    if timestamps is not None:
        print(f"  timestamps shape   = {timestamps.shape} (steps, ranks)")
    for key, arr in phase_times.items():
        print(f"  {key} shape      = {arr.shape}")
    print(f"  num_total_steps    = {data['num_total_steps']}")
    print(f"  num_decode_steps   = {data['num_decode_steps']}")
    return moe_times, local_token_counts, timestamps, phase_times


def filter_peak_batch(
    moe_times, batch_sizes, local_token_counts, timestamps, peak_pct: float
):
    """Keep only the contiguous time range where global batch size >= peak_pct * peak.

    Global batch size = sum of batch_sizes across all ranks for each step.
    We find the first and last step meeting the threshold, then slice all arrays
    to that contiguous range.
    """
    if batch_sizes is None or peak_pct <= 0:
        return moe_times, batch_sizes, local_token_counts, timestamps

    global_batch = batch_sizes.sum(axis=1)  # [steps]
    peak = global_batch.max()
    threshold = peak_pct * peak
    mask = global_batch >= threshold
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(
            f"  WARNING: No steps meet {peak_pct:.0%} of peak batch ({peak}). Keeping all."
        )
        return moe_times, batch_sizes, local_token_counts, timestamps

    start, end = indices[0], indices[-1] + 1  # contiguous range
    kept = end - start
    total = len(global_batch)
    print(
        f"  Peak filter: global_batch peak={peak}, threshold={threshold:.0f} "
        f"({peak_pct:.0%}), keeping steps [{start}:{end}] ({kept}/{total} steps)"
    )

    moe_times = moe_times[start:end]
    batch_sizes = batch_sizes[start:end]
    if local_token_counts is not None:
        local_token_counts = local_token_counts[start:end]
    if timestamps is not None:
        timestamps = timestamps[start:end]
    return moe_times, batch_sizes, local_token_counts, timestamps


# ── Timing plots (moe_times: [steps, ranks]) ────────────────────────────


def plot_step_time_timeline(
    moe_times: np.ndarray, output_dir: Path, elapsed: np.ndarray | None = None
):
    """Time series of per-step forward time for each rank."""
    num_steps, world_size = moe_times.shape
    fig, ax = plt.subplots(figsize=(14, 5))
    if elapsed is not None:
        x = elapsed[:, 0]  # rank-0 elapsed seconds
        xlabel = "Time (s)"
    else:
        x = np.arange(num_steps)
        xlabel = "Decode Step"
    for rank in range(world_size):
        ax.plot(x, moe_times[:, rank], alpha=0.6, linewidth=0.5, label=f"Rank {rank}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Forward Time (ms)")
    ax.set_title(
        f"Per-Step Forward Time ({num_steps} decode steps, {world_size} ranks)"
    )
    ax.legend(fontsize=7, ncol=min(world_size, 4), loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "step_time_timeline.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_avg_time_per_rank(moe_times: np.ndarray, output_dir: Path):
    """Bar chart of average forward time per rank."""
    avg = moe_times.mean(axis=0)
    world_size = len(avg)
    fig, ax = plt.subplots(figsize=(8, 4))
    ranks = np.arange(world_size)
    colors = plt.cm.RdYlGn_r((avg - avg.min()) / max(avg.max() - avg.min(), 1e-6))
    ax.bar(ranks, avg, color=colors)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Avg Forward Time (ms)")
    ax.set_title(
        f"Avg Forward Time per Rank (CV={avg.std() / max(avg.mean(), 1e-6):.4f})"
    )
    ax.set_xticks(ranks)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = "avg_time_per_rank.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_time_imbalance_timeline(
    moe_times: np.ndarray, output_dir: Path, elapsed: np.ndarray | None = None
):
    """Time series of max/min ratio across ranks per step."""
    mins = moe_times.min(axis=1)
    maxs = moe_times.max(axis=1)
    ratio = np.where(mins > 0, maxs / mins, 1.0)
    num_steps = len(ratio)

    fig, ax = plt.subplots(figsize=(14, 4))
    if elapsed is not None:
        x = elapsed[:, 0]
        xlabel = "Time (s)"
    else:
        x = np.arange(num_steps)
        xlabel = "Decode Step"
    ax.plot(x, ratio, linewidth=0.5, color="steelblue")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Max/Min Time Ratio")
    ax.set_title("Forward Time Imbalance (max/min across ranks)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = "time_imbalance_timeline.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Per-layer per-rank MoE time heatmap ──────────────────────────────────


def plot_avg_heatmap_moe_times(moe_times: np.ndarray, output_dir: Path):
    """Heatmap of average MoE compute time (ms) per layer per rank.

    moe_times: [steps, layers, ranks]
    """
    avg = moe_times.astype(np.float64).mean(axis=0)  # [layers, ranks]
    num_layers, world_size = avg.shape

    fig, ax = plt.subplots(
        figsize=(max(10, num_layers * 0.15), max(4, world_size * 0.4))
    )
    im = ax.imshow(avg.T, aspect="auto", cmap="inferno", interpolation="nearest")
    ax.set_xlabel("Layer ID")
    ax.set_ylabel("Rank")
    ax.set_title(f"MoE Compute Time (ms) — Avg over {moe_times.shape[0]} Decode Steps")
    ax.set_xticks(np.arange(0, num_layers, max(1, num_layers // 20)))
    ax.set_yticks(np.arange(world_size))
    fig.colorbar(im, ax=ax, label="Time (ms)")
    fig.tight_layout()
    fname = "moe_times_avg_heatmap.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Local token count plots (local_token_counts: [steps, layers, ranks]) ─


def plot_avg_heatmap_local_tokens(local_token_counts: np.ndarray, output_dir: Path):
    """Heatmap of average local token counts per layer per rank."""
    avg = local_token_counts.astype(np.float64).mean(axis=0)
    num_layers, world_size = avg.shape

    fig, ax = plt.subplots(
        figsize=(max(10, num_layers * 0.15), max(4, world_size * 0.4))
    )
    im = ax.imshow(avg.T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Layer ID")
    ax.set_ylabel("Rank")
    ax.set_title(
        f"Local Token Count - Avg over {local_token_counts.shape[0]} Decode Steps"
    )
    ax.set_xticks(np.arange(0, num_layers, max(1, num_layers // 20)))
    ax.set_yticks(np.arange(world_size))
    fig.colorbar(im, ax=ax, label="Token-Expert Pairs")
    fig.tight_layout()
    fname = "local_tokens_avg_heatmap.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_rank_imbalance_local_tokens(local_token_counts: np.ndarray, output_dir: Path):
    """Bar chart of CV of local token counts across ranks per layer."""
    avg = local_token_counts.astype(np.float64).mean(axis=0)
    mean = avg.mean(axis=1)
    std = avg.std(axis=1)
    cv = np.where(mean > 0, std / mean, 0.0)
    num_layers = cv.shape[0]

    fig, ax = plt.subplots(figsize=(8, max(5, num_layers * 0.12)))
    layers = np.arange(num_layers)
    colors = plt.cm.RdYlGn_r(cv / max(cv.max(), 1e-6))
    ax.barh(layers, cv, color=colors)
    ax.set_xlabel("Coefficient of Variation (std / mean)")
    ax.set_ylabel("Layer ID")
    ax.set_title("Local Token Count Imbalance Across Ranks")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fname = "local_tokens_rank_imbalance.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_step_rank_heatmap_local_tokens(
    local_token_counts: np.ndarray,
    output_dir: Path,
    elapsed: np.ndarray | None = None,
):
    """Heatmap of total local tokens per step per rank (summed across layers).

    X-axis: time (s) or decode step, Y-axis: rank, color: total local token count.
    """
    # local_token_counts: [steps, layers, ranks]
    total_per_step_rank = local_token_counts.astype(np.float64).sum(
        axis=1
    )  # [steps, ranks]
    num_steps, world_size = total_per_step_rank.shape

    fig, ax = plt.subplots(figsize=(14, max(3, world_size * 0.5)))

    if elapsed is not None:
        t = elapsed[:, 0]  # rank-0 elapsed seconds
        # Use pcolormesh for non-uniform x-axis (time)
        # Create bin edges: midpoints between timestamps, plus boundaries
        t_edges = np.empty(num_steps + 1)
        t_edges[0] = t[0]
        t_edges[-1] = t[-1] + (t[-1] - t[-2]) if num_steps > 1 else t[-1] + 0.1
        t_edges[1:-1] = (t[:-1] + t[1:]) / 2
        rank_edges = np.arange(world_size + 1) - 0.5
        im = ax.pcolormesh(
            t_edges,
            rank_edges,
            total_per_step_rank.T,
            cmap="inferno",
            shading="flat",
        )
        xlabel = "Time (s)"
    else:
        im = ax.imshow(
            total_per_step_rank.T,
            aspect="auto",
            cmap="inferno",
            interpolation="nearest",
            origin="lower",
        )
        xlabel = "Decode Step"

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rank")
    ax.set_title(
        f"Local Token Count per Step per Rank "
        f"(summed across {local_token_counts.shape[1]} layers, {num_steps} steps)"
    )
    ax.set_yticks(np.arange(world_size))
    fig.colorbar(im, ax=ax, label="Total Local Tokens (all layers)")
    fig.tight_layout()
    fname = "local_tokens_step_rank_heatmap.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_cumulative_tokens_timeline(
    local_token_counts: np.ndarray,
    output_dir: Path,
    elapsed: np.ndarray | None = None,
):
    """Cumulative total local tokens processed per rank over time.

    Each rank gets a line. Y-axis is cumulative sum of tokens (summed across
    all MoE layers). Divergence between lines shows workload imbalance.
    """
    # local_token_counts: [steps, layers, ranks]
    # Sum across layers → [steps, ranks], then cumsum over steps
    per_step_rank = local_token_counts.astype(np.float64).sum(axis=1)  # [steps, ranks]
    cumulative = np.cumsum(per_step_rank, axis=0)  # [steps, ranks]
    num_steps, world_size = cumulative.shape

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

    if elapsed is not None:
        xlabel = "Time (s)"
    else:
        xlabel = "Decode Step"

    # Top: cumulative lines
    ax = axes[0]
    cmap = plt.cm.tab10
    for rank in range(world_size):
        if elapsed is not None:
            x = elapsed[:, rank]  # per-rank timestamps
        else:
            x = np.arange(num_steps)
        ax.plot(
            x,
            cumulative[:, rank],
            linewidth=1.2,
            alpha=0.8,
            color=cmap(rank / max(world_size - 1, 1)),
            label=f"Rank {rank}",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative Local Tokens")
    ax.set_title(
        f"Cumulative Local MoE Tokens per Rank ({num_steps} steps, {world_size} ranks)"
    )
    ax.legend(fontsize=7, ncol=min(world_size, 4), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Bottom: gap between max and min rank (absolute divergence)
    ax2 = axes[1]
    if elapsed is not None:
        x_bottom = elapsed[:, 0]
    else:
        x_bottom = np.arange(num_steps)
    gap = cumulative.max(axis=1) - cumulative.min(axis=1)
    gap_pct = 100.0 * gap / np.maximum(cumulative.mean(axis=1), 1.0)
    ax2.fill_between(x_bottom, gap_pct, alpha=0.4, color="coral")
    ax2.plot(x_bottom, gap_pct, linewidth=1, color="red")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Max-Min Gap (%)")
    ax2.set_title("Cumulative Workload Divergence (max-min as % of mean)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = "cumulative_tokens_timeline.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Phase breakdown plots ────────────────────────────────────────────────


def plot_phase_breakdown_per_rank(
    moe_times: np.ndarray, phase_times: dict, output_dir: Path
):
    """Stacked bar chart of average per-step time breakdown by rank.

    Each bar shows the four phases: attn (blue), ag (orange), moe (green), ar (red).
    moe_times: [steps, layers, ranks]
    phase_times: dict with attn_times/ag_times/ar_times, each [steps, layers, ranks]
    """
    attn = phase_times.get("attn_times")
    ag = phase_times.get("ag_times")
    ar = phase_times.get("ar_times")
    if attn is None or ag is None or ar is None:
        print("  Skipping phase breakdown (missing phase data)")
        return

    world_size = moe_times.shape[2]
    ranks = np.arange(world_size)

    fwd = phase_times.get("fwd_times")

    avg_moe = moe_times.sum(axis=1).mean(axis=0)
    avg_attn = attn.sum(axis=1).mean(axis=0)
    avg_ag = ag.sum(axis=1).mean(axis=0)
    avg_ar = ar.sum(axis=1).mean(axis=0)
    if fwd is not None:
        avg_fwd = fwd.sum(axis=1).mean(axis=0)
        avg_other = np.maximum(0, avg_fwd - avg_moe - avg_attn - avg_ag - avg_ar)
    else:
        avg_other = np.zeros(world_size)

    fig, ax = plt.subplots(figsize=(max(8, world_size * 0.6), 5))
    bar_width = 0.6

    bottom = np.zeros(world_size)
    ax.bar(ranks, avg_ag, bar_width, bottom=bottom, label="All-Gather", color="#ff7f0e")
    bottom += avg_ag
    ax.bar(
        ranks, avg_attn, bar_width, bottom=bottom, label="Attention", color="#1f77b4"
    )
    bottom += avg_attn
    ax.bar(ranks, avg_ar, bar_width, bottom=bottom, label="All-Reduce", color="#d62728")
    bottom += avg_ar
    ax.bar(ranks, avg_moe, bar_width, bottom=bottom, label="MoE FFN", color="#2ca02c")
    bottom += avg_moe
    if fwd is not None and avg_other.sum() > 0:
        ax.bar(
            ranks, avg_other, bar_width, bottom=bottom,
            label="Other (LN, gate, scatter…)", color="#9467bd",
        )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Avg Per-Step Time (ms)")
    ax.set_title(
        f"Phase Breakdown per Rank ({moe_times.shape[0]} decode steps, "
        f"summed across {moe_times.shape[1]} layers)"
    )
    ax.set_xticks(ranks)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = "phase_breakdown_per_rank.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Plot MoE kernel balance data from .pt dump files"
    )
    parser.add_argument(
        "input", nargs="+", help="Path(s) to .pt file(s), supports glob"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for plots (default: same dir as input)",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=10,
        help="Number of initial steps to skip as warmup (default: 10)",
    )
    parser.add_argument(
        "--peak-pct",
        type=float,
        default=0,
        help="Keep only the contiguous time range where global batch size "
        ">= this fraction of peak (e.g. 0.9 for 90%%). 0 = disabled.",
    )
    args = parser.parse_args()

    paths = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        for entry in expanded:
            p = Path(entry)
            if p.is_dir():
                pts = sorted(p.glob("moe_kernel_balance_*.pt"))
                paths.extend(str(f) for f in pts)
            else:
                paths.append(entry)
    if not paths:
        parser.error("No matching .pt files found")

    for pt_path_str in sorted(paths):
        pt_path = Path(pt_path_str)
        output_dir = Path(args.output_dir) if args.output_dir else pt_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        moe_times, local_token_counts, timestamps, phase_times = load_data(str(pt_path))

        warmup = args.warmup
        if warmup > 0 and moe_times.shape[0] > warmup:
            print(f"  Skipping first {warmup} steps (warmup)")
            moe_times = moe_times[warmup:]
            if local_token_counts is not None:
                local_token_counts = local_token_counts[warmup:]
            if timestamps is not None:
                timestamps = timestamps[warmup:]
            for key in list(phase_times):
                phase_times[key] = phase_times[key][warmup:]

        if args.peak_pct > 0 and local_token_counts is not None:
            global_batch = local_token_counts.sum(axis=(1, 2))
            peak = global_batch.max()
            threshold = args.peak_pct * peak
            indices = np.where(global_batch >= threshold)[0]
            if len(indices) > 0:
                start, end = indices[0], indices[-1] + 1
                print(
                    f"  Peak filter: keeping steps [{start}:{end}] "
                    f"({end - start}/{len(global_batch)} steps)"
                )
                moe_times = moe_times[start:end]
                local_token_counts = local_token_counts[start:end]
                if timestamps is not None:
                    timestamps = timestamps[start:end]
                for key in list(phase_times):
                    phase_times[key] = phase_times[key][start:end]

        if moe_times.sum() == 0:
            print("  All times are zero, skipping.")
            continue

        elapsed = None
        if timestamps is not None:
            elapsed = timestamps - timestamps[0, 0]

        moe_times_per_step = moe_times.sum(axis=1)

        print(f"  Generating plots in {output_dir}/")
        plot_step_time_timeline(moe_times_per_step, output_dir, elapsed)
        plot_avg_time_per_rank(moe_times_per_step, output_dir)
        plot_time_imbalance_timeline(moe_times_per_step, output_dir, elapsed)
        plot_avg_heatmap_moe_times(moe_times, output_dir)

        if local_token_counts is not None:
            plot_avg_heatmap_local_tokens(local_token_counts, output_dir)
            plot_rank_imbalance_local_tokens(local_token_counts, output_dir)
            plot_step_rank_heatmap_local_tokens(local_token_counts, output_dir, elapsed)
            plot_cumulative_tokens_timeline(local_token_counts, output_dir, elapsed)

        if phase_times:
            plot_phase_breakdown_per_rank(moe_times, phase_times, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
