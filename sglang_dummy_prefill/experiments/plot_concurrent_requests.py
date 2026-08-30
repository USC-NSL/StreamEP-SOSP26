#!/usr/bin/env python3
"""
Plot global running request timeline for 5 configs using REAL recorded data.
SGLang: batch_size_timeline .pt files (per-step local batch sizes per rank).
AMoE: #running requests from DPScheduler log.
"""
import glob, re
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

REPO = Path("/home/yizhuoliang/sglang-fake-prefill")

def parse_amoe(log_path):
    times, running = [], []
    for line in open(log_path):
        m = re.search(r"(\d+\.\d+)\s*-\s*\[INFO\].*#running requests:\s*(\d+)", line)
        if m:
            times.append(float(m.group(1))); running.append(int(m.group(2)))
    if not times: return np.array([]), np.array([])
    t0 = times[0]
    return np.array([t - t0 for t in times]), np.array(running)

def load_bs(recorder_dir, mode="ep"):
    pts = sorted(glob.glob(str(Path(recorder_dir) / "batch_size_timeline_*.pt")))
    if not pts: return np.array([])
    all_local = [torch.load(f, weights_only=False)["local_batch_sizes"] for f in pts]
    if mode == "ep":
        n = min(len(bs) for bs in all_local)
        g = np.zeros(n)
        for bs in all_local: g += np.array(bs[:n])
        return g
    return np.array(all_local[0])

def get_itl(server_log):
    itls = []
    for line in open(server_log):
        m = re.search(r"'median':\s*([\d.]+)", line)
        if m: itls.append(float(m.group(1)))
    return np.median(itls[5:-5]) if len(itls) > 10 else (np.median(itls) if itls else 0.5)

def get_bench_duration(bench_log):
    """Extract wall-clock benchmark duration from bench.log."""
    for line in open(bench_log):
        m = re.search(r"Benchmark duration \(s\):\s*([\d.]+)", line)
        if m: return float(m.group(1))
    return None

def main():
    amoe_t, amoe_c = parse_amoe(REPO / "amoe.txt")

    data = [
        ("sgl-009/exp1_legal", "ep", "SGLang EP16 (legal)", "#1f77b4", "-"),
        ("sgl-009/exp2_balanced_legal", "ep", "SGLang EP16 (balanced legal)", "#2ca02c", "-"),
        ("sgl-010/exp1_legal", "pp", "SGLang PP4×TP2 (legal)", "#ff7f0e", "--"),
        ("sgl-010/exp2_balanced_legal", "pp", "SGLang PP4×TP2 (balanced legal)", "#d62728", "--"),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))

    for subdir, mode, label, color, ls in data:
        bs = load_bs(REPO / f"experiments/{subdir}/recorder_raw", mode=mode)
        if len(bs) == 0:
            print(f"{label}: NO DATA"); continue
        # Use wall-clock duration from bench.log for accurate time axis
        # (avoids PP micro-batch inflation: PP4 records 4x steps per global decode step)
        wall_dur = get_bench_duration(REPO / f"experiments/{subdir}/bench.log")
        t = np.linspace(0, wall_dur, len(bs))
        ax.plot(t, bs, label=label, color=color, linestyle=ls, linewidth=1.5, alpha=0.85)
        print(f"{label}: {len(bs)} steps, wall={wall_dur:.0f}s, peak={bs.max():.0f}")

    if len(amoe_t) > 0:
        ax.plot(amoe_t, amoe_c, label="AMoE EP16 (legal)", color="#9467bd",
                linestyle="-.", linewidth=2, marker="o", markersize=5)
        print(f"AMoE: {len(amoe_t)} pts, t=[{amoe_t[0]:.0f},{amoe_t[-1]:.0f}]s, peak={amoe_c.max()}")

    ax.set_xlabel("Time (seconds)", fontsize=13)
    ax.set_ylabel("Global Running Requests", fontsize=13)
    ax.set_title("Global Running Requests Timeline (Recorded)\n"
                 "10k reqs, rps=2000, input∈[256,512], output∈[256,512], fake-prefill", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = REPO / "experiments/plots/concurrent_requests_timeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()

if __name__ == "__main__":
    main()
