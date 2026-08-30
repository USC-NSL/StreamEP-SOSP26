#!/usr/bin/env python3
"""Plot in-flight request timelines from SGLang server logs.

Usage: python plot_sglang_inflight_timeline.py <results_dir> [profile_filter]
  results_dir should contain sglang_*/ subdirectories, each with logs/server_head.log
  profile_filter (optional): e.g. "ep16" to only plot ep16 runs (default: all)
"""
import re, os, sys, glob
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <results_dir> [profile_filter]")
    sys.exit(1)

BASE = sys.argv[1]
PROFILE_FILTER = sys.argv[2] if len(sys.argv) > 2 else None

DIRS = sorted(glob.glob(os.path.join(BASE, "sglang_*")))

LOG_PATTERN = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] from Detokenizer Manager,.*?"
    r"In-flight requests: (\d+),\s*Waiting requests: (\w+)"
)
LOG_PATTERN_OLD = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] from Detokenizer Manager,.*?"
    r"In-flight requests: (\d+)"
)

def label_from_dir(d):
    name = os.path.basename(d).replace("sglang_", "")
    profile, workload = name.split("-", 1)
    parts = workload.split("_")
    return f"{profile} {parts[0]} ({parts[1]})"

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
          "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"]

plotted = 0
for d in DIRS:
    if PROFILE_FILTER and PROFILE_FILTER not in os.path.basename(d):
        continue

    log_path = os.path.join(d, "logs", "server_head.log")
    if not os.path.isfile(log_path):
        continue

    with open(log_path) as f:
        content = f.read()

    timestamps, inflight, waiting = [], [], []
    has_waiting = False
    for m in LOG_PATTERN.finditer(content):
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        timestamps.append(ts)
        inflight.append(int(m.group(2)))
        w = m.group(3)
        if w != "None":
            has_waiting = True
            waiting.append(int(w))
        else:
            waiting.append(0)

    if not timestamps:
        for m in LOG_PATTERN_OLD.finditer(content):
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            timestamps.append(ts)
            inflight.append(int(m.group(2)))
        has_waiting = False

    if not timestamps:
        continue

    t0 = timestamps[0]
    rel = [(t - t0).total_seconds() for t in timestamps]
    c = colors[plotted % len(colors)]
    label = label_from_dir(d)

    axes[0].plot(rel, inflight, label=label, linewidth=1.5, color=c)
    if has_waiting:
        axes[1].plot(rel, waiting, label=label, linewidth=1.5, color=c)
    plotted += 1

profile_tag = PROFILE_FILTER or "all"
axes[0].set_ylabel("In-flight Requests", fontsize=11)
axes[0].set_title(f"SGLang In-Flight Request Timelines ({profile_tag})", fontsize=13)
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Time since first Detokenizer log (s)", fontsize=11)
axes[1].set_ylabel("Waiting Requests", fontsize=11)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
out = os.path.join(BASE, f"sglang_inflight_timeline_{profile_tag}.png")
fig.savefig(out, dpi=180)
print(f"Saved to {out}")
