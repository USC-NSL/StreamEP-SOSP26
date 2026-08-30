#!/usr/bin/env python3
"""Compare recorder dumps across multiple experiments.

moe_times:          [decode_steps, num_layers, world_size] — per-layer MoE compute time (ms)
local_token_counts: [decode_steps, num_layers, world_size] — per-layer local token counts
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def filter_peak_batch(data: dict, peak_pct: float) -> dict:
    ltok = data["local_token_counts"]
    global_batch = ltok.sum(axis=(1, 2))
    peak = global_batch.max()
    threshold = peak_pct * peak
    mask = global_batch >= threshold
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(f"  WARNING: No steps meet {peak_pct:.0%} of peak. Keeping all.")
        return data

    start, end = indices[0], indices[-1] + 1
    print(
        f"  Peak filter: global_batch peak={peak}, threshold={threshold:.0f} "
        f"({peak_pct:.0%}), keeping steps [{start}:{end}] ({end - start}/{len(global_batch)} steps)"
    )

    filtered = {k: v for k, v in data.items()}
    filtered["moe_times"] = data["moe_times"][start:end]
    filtered["local_token_counts"] = data["local_token_counts"][start:end]
    if "timestamps" in data:
        filtered["timestamps"] = data["timestamps"][start:end]
    for key in ("attn_times", "ag_times", "ar_times"):
        if key in data:
            filtered[key] = data[key][start:end]
    return filtered


def _resolve_kernel_balance_path(path_or_dir: str) -> str | None:
    if path_or_dir.endswith(".pt") and os.path.exists(path_or_dir):
        return path_or_dir
    if os.path.isdir(path_or_dir):
        recorder_dir = os.path.join(path_or_dir, "recorder_raw")
        search_dir = recorder_dir if os.path.isdir(recorder_dir) else path_or_dir
    else:
        search_dir = os.path.dirname(path_or_dir) or "."
    matches = sorted(glob.glob(os.path.join(search_dir, "moe_kernel_balance_*.pt")))
    return matches[-1] if matches else None


def load_experiment(path_or_dir: str):
    pt_path = _resolve_kernel_balance_path(path_or_dir)
    if pt_path is None:
        print(f"  WARNING: No moe_kernel_balance_*.pt found in {path_or_dir}")
        return None

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    # moe_times: [steps, layers, ranks], local_token_counts: [steps, layers, ranks]
    moe_times = data["moe_times"].cpu().float().numpy()
    local_token_counts = data["local_token_counts"].cpu().int().numpy()
    timestamps = data.get("timestamps")
    if timestamps is not None:
        timestamps = timestamps.cpu().double().numpy()

    result = {
        "pt_path": pt_path,
        "moe_times": moe_times,
        "local_token_counts": local_token_counts,
    }
    if timestamps is not None:
        result["timestamps"] = timestamps
    for key in ("attn_times", "ag_times", "ar_times"):
        t = data.get(key)
        if t is not None:
            result[key] = t.cpu().float().numpy()
    return result


def plot_per_rank_cdf(experiments, values_fn, xlabel, title, output_path):
    """CDF where each rank of each experiment is a separate line.

    values_fn(data) should return array of shape [steps, layers, ranks].
    Each rank's CDF is computed over the flattened (steps × layers) values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, data) in enumerate(experiments):
        color = COLORS[i % len(COLORS)]
        arr = values_fn(data)
        if arr is None or arr.size == 0:
            continue
        num_steps, num_layers, world_size = arr.shape
        all_vals = arr.reshape(-1)
        label_text = (
            f"{label} (P50={np.percentile(all_vals, 50):.1f}, "
            f"P99={np.percentile(all_vals, 99):.1f})"
        )
        for rank in range(world_size):
            rank_vals = arr[:, :, rank].reshape(-1)
            rank_vals = rank_vals[rank_vals > 0]
            if len(rank_vals) == 0:
                continue
            sorted_vals = np.sort(rank_vals)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            lbl = label_text if rank == 0 else None
            ax.plot(sorted_vals, cdf, linewidth=0.8, color=color, alpha=0.5, label=lbl)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_timeline_grid(experiments, output_path, metric="local_token_counts"):
    n = len(experiments)
    rows, cols = (1, n) if n <= 2 else (2, 2)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    for i, (label, data) in enumerate(experiments):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        color = COLORS[i % len(COLORS)]

        has_ts = "timestamps" in data
        if has_ts:
            ts = data["timestamps"]
            t0 = ts[0, 0]
            x = ts[:, 0] - t0
            xlabel = "Time (s)"
        else:
            x = np.arange(data["local_token_counts"].shape[0])
            xlabel = "Decode Step"

        if metric == "local_token_counts":
            ltok = data["local_token_counts"].astype(np.float64)
            per_step_rank = ltok.mean(axis=1)
        else:
            per_step_rank = data["moe_times"].mean(axis=1)

        for rank in range(per_step_rank.shape[1]):
            ax.plot(x, per_step_rank[:, rank], linewidth=0.7, alpha=0.25, color=color)

        mean_series = per_step_rank.mean(axis=1)
        ax.plot(x, mean_series, linewidth=2, color=color, label="rank mean")

        if len(mean_series) > 50:
            window = max(len(mean_series) // 50, 10)
            rolling = np.convolve(mean_series, np.ones(window) / window, mode="valid")
            ax.plot(
                x[window - 1 :],
                rolling,
                linewidth=2.5,
                color="black",
                alpha=0.8,
                label=f"rolling avg (w={window})",
            )

        ylabel = (
            "Local Tokens (avg over layers)"
            if metric == "local_token_counts"
            else "MoE Time (ms, avg over layers)"
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].set_visible(False)

    title = (
        "Local Token Count Timeline"
        if metric == "local_token_counts"
        else "MoE Compute Time Timeline"
    )
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_cumulative_tokens_comparison(experiments, output_path):
    n = len(experiments)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n), squeeze=False)

    for i, (label, data) in enumerate(experiments):
        ax = axes[i][0]
        ltok = data["local_token_counts"]
        if ltok is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(label)
            continue

        per_step_rank = ltok.astype(np.float64).sum(axis=1)
        cumulative = np.cumsum(per_step_rank, axis=0)
        num_steps, world_size = cumulative.shape

        has_ts = "timestamps" in data
        if has_ts:
            ts = data["timestamps"]
            t0 = ts[0, 0]
            xlabel = "Time (s)"
        else:
            xlabel = "Decode Step"

        cmap = plt.cm.tab10
        for rank in range(world_size):
            x_rank = (ts[:, rank] - t0) if has_ts else np.arange(num_steps)
            ax.plot(
                x_rank,
                cumulative[:, rank],
                linewidth=1.2,
                alpha=0.8,
                color=cmap(rank / max(world_size - 1, 1)),
                label=f"Rank {rank}",
            )

        final = cumulative[-1]
        gap_pct = 100.0 * (final.max() - final.min()) / max(final.mean(), 1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative Local Tokens")
        ax.set_title(f"{label}  (final max-min gap: {gap_pct:.2f}%)")
        ax.legend(fontsize=7, ncol=min(world_size, 4), loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cumulative Local MoE Tokens per Rank", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_heatmap_summary(experiments, output_path, metric="local_token_counts"):
    """Side-by-side layer×rank heatmaps for all experiments in a single PNG."""
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)

    if metric == "local_token_counts":
        cmap = "viridis"
        cb_label = "Token-Expert Pairs"
        suptitle = "Local Token Count — Avg per Layer per Rank"
    else:
        cmap = "inferno"
        cb_label = "Time (ms)"
        suptitle = "MoE Compute Time (ms) — Avg per Layer per Rank"

    for i, (label, data) in enumerate(experiments):
        ax = axes[0][i]
        arr = data[metric].astype(np.float64)
        avg = arr.mean(axis=0)  # [layers, ranks]
        num_layers, world_size = avg.shape

        im = ax.imshow(avg.T, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xlabel("Layer ID")
        if i == 0:
            ax.set_ylabel("Rank")
        else:
            ax.set_yticklabels([])
        ax.set_title(label)
        ax.set_xticks(np.arange(0, num_layers, max(1, num_layers // 10)))
        ax.set_yticks(np.arange(world_size))
        fig.colorbar(im, ax=ax, label=cb_label, shrink=0.8)

    fig.suptitle(suptitle, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_phase_breakdown_comparison(experiments, output_path):
    """Grouped bar chart comparing phase breakdown across experiments.

    Each experiment gets a stacked bar showing average per-step time
    split into ag, attn, ar, moe, and other (from fwd_times) phases.
    When fwd_times is available, "other" = fwd_total - (attn+ag+moe+ar)
    captures layernorm, router/gate, dp_scatter, etc.
    """
    has_phases = all(
        "attn_times" in d and "ag_times" in d and "ar_times" in d
        for _, d in experiments
    )
    if not has_phases:
        print("  Skipping phase breakdown comparison (missing phase data)")
        return

    has_fwd = all("fwd_times" in d for _, d in experiments)

    n = len(experiments)
    labels = []
    avg_ag_vals = []
    avg_attn_vals = []
    avg_ar_vals = []
    avg_moe_vals = []
    avg_other_vals = []

    for label, data in experiments:
        labels.append(label)
        avg_moe_vals.append(data["moe_times"].sum(axis=1).mean())
        avg_attn_vals.append(data["attn_times"].sum(axis=1).mean())
        avg_ag_vals.append(data["ag_times"].sum(axis=1).mean())
        avg_ar_vals.append(data["ar_times"].sum(axis=1).mean())
        if has_fwd:
            fwd_total = data["fwd_times"].sum(axis=1).mean()
            four_phases = (
                data["moe_times"].sum(axis=1).mean()
                + data["attn_times"].sum(axis=1).mean()
                + data["ag_times"].sum(axis=1).mean()
                + data["ar_times"].sum(axis=1).mean()
            )
            avg_other_vals.append(max(0, fwd_total - four_phases))
        else:
            avg_other_vals.append(0)

    avg_ag = np.array(avg_ag_vals)
    avg_attn = np.array(avg_attn_vals)
    avg_ar = np.array(avg_ar_vals)
    avg_moe = np.array(avg_moe_vals)
    avg_other = np.array(avg_other_vals)

    x = np.arange(n)
    bar_width = 0.5
    fig, ax = plt.subplots(figsize=(max(6, n * 2), 5))

    bottom = np.zeros(n)
    ax.bar(x, avg_ag, bar_width, bottom=bottom, label="All-Gather", color="#ff7f0e")
    bottom += avg_ag
    ax.bar(x, avg_attn, bar_width, bottom=bottom, label="Attention", color="#1f77b4")
    bottom += avg_attn
    ax.bar(x, avg_ar, bar_width, bottom=bottom, label="All-Reduce", color="#d62728")
    bottom += avg_ar
    ax.bar(x, avg_moe, bar_width, bottom=bottom, label="MoE FFN", color="#2ca02c")
    bottom += avg_moe
    if has_fwd and avg_other.sum() > 0:
        ax.bar(
            x, avg_other, bar_width, bottom=bottom,
            label="Other (LN, gate, scatter…)", color="#9467bd",
        )

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Avg Per-Step Time (ms, summed across layers)")
    title = "Phase Breakdown Comparison"
    if has_fwd:
        title += " (full layer-forward coverage)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare recorder dumps across experiments"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help='Each entry is "Label:path_to_experiment_dir_or_pt"',
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--peak-pct",
        type=float,
        default=0,
        help="Keep only the contiguous time range where global batch size "
        ">= this fraction of peak (e.g. 0.9 for 90%%). 0 = disabled.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = []
    for spec in args.experiments:
        if ":" not in spec:
            print(f"ERROR: Expected 'Label:path', got '{spec}'")
            sys.exit(1)
        label, path_or_dir = spec.split(":", 1)
        print(f"Loading {label} from {path_or_dir}...")
        data = load_experiment(path_or_dir)
        if data is not None:
            if args.peak_pct > 0:
                data = filter_peak_batch(data, args.peak_pct)
            experiments.append((label, data))
            s = data["moe_times"].shape
            print(
                f"  source={data['pt_path']}, steps={s[0]}, layers={s[1]}, ranks={s[2]}"
            )

    if not experiments:
        print("No data loaded.")
        sys.exit(1)

    print("\nPlotting MoE local token count CDF (per-layer per-rank)...")
    plot_per_rank_cdf(
        experiments,
        lambda d: d["local_token_counts"].astype(np.float64),
        xlabel="Local Token-Expert Pairs (per MoE layer step)",
        title="MoE Local Token Count CDF — Per Layer Per Rank",
        output_path=os.path.join(args.output_dir, "cdf_moe_step_batch_size.png"),
    )

    print("Plotting MoE compute time CDF (per-layer per-rank)...")
    plot_per_rank_cdf(
        experiments,
        lambda d: d["moe_times"],
        xlabel="MoE Compute Time (ms, per layer step)",
        title="MoE Compute Time CDF — Per Layer Per Rank",
        output_path=os.path.join(args.output_dir, "cdf_moe_step_exec_time.png"),
    )

    print("Plotting local token count timeline grid...")
    plot_timeline_grid(
        experiments,
        output_path=os.path.join(args.output_dir, "timeline_local_tokens.png"),
        metric="local_token_counts",
    )

    print("Plotting MoE compute time timeline grid...")
    plot_timeline_grid(
        experiments,
        output_path=os.path.join(args.output_dir, "timeline_moe_time.png"),
        metric="moe_times",
    )

    print("Plotting cumulative token comparison...")
    plot_cumulative_tokens_comparison(
        experiments,
        output_path=os.path.join(args.output_dir, "cumulative_tokens_comparison.png"),
    )

    print("Plotting local token count heatmap summary...")
    plot_heatmap_summary(
        experiments,
        output_path=os.path.join(args.output_dir, "heatmap_local_tokens.png"),
        metric="local_token_counts",
    )

    print("Plotting MoE compute time heatmap summary...")
    plot_heatmap_summary(
        experiments,
        output_path=os.path.join(args.output_dir, "heatmap_moe_times.png"),
        metric="moe_times",
    )

    print("Plotting phase breakdown comparison...")
    plot_phase_breakdown_comparison(
        experiments,
        output_path=os.path.join(args.output_dir, "phase_breakdown_comparison.png"),
    )

    print(f"\nAll comparison plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
