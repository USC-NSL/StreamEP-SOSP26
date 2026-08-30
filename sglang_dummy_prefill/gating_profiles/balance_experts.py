#!/usr/bin/env python3
"""
Balance MoE expert assignment across 16 GPUs via per-layer 1-to-1 index remapping.

For each layer, finds a bijection expert_old -> expert_new such that the 16 GPU groups
(each with 8 consecutive experts) have roughly equal total token counts.

Algorithm: LPT (Longest Processing Time first) greedy scheduling.
"""

import json
import heapq
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NUM_EXPERTS = 128
NUM_GPUS = 16
EXPERTS_PER_GPU = NUM_EXPERTS // NUM_GPUS  # 8

EXPERT_COLS = [
    "expert_logical_k0",
    "expert_logical_k1",
    "expert_logical_k2",
    "expert_logical_k3",
]

PARQUET_DIR = Path("gating_profiles")
OUTPUT_DIR = Path("gating_profiles/balanced_output")


def count_experts_per_layer(df: pd.DataFrame) -> dict:
    """Count total token assignments per expert per layer using np.bincount."""
    layers_arr = df["layer"].values
    unique_layers = np.unique(layers_arr)
    result = {}

    for layer in unique_layers:
        mask = layers_arr == layer
        counts = np.zeros(NUM_EXPERTS, dtype=np.int64)
        for col in EXPERT_COLS:
            vals = df[col].values[mask]
            counts += np.bincount(vals, minlength=NUM_EXPERTS)
        result[int(layer)] = counts

    return result


def greedy_balance(expert_counts: np.ndarray) -> np.ndarray:
    """
    Find a permutation that balances expert counts across NUM_GPUS groups of EXPERTS_PER_GPU.

    Uses LPT (Longest Processing Time first) greedy algorithm.
    Returns: mapping array where mapping[old_expert_id] = new_expert_id
    """
    assert len(expert_counts) == NUM_EXPERTS

    # Sort experts by count descending (LPT)
    sorted_experts = np.argsort(-expert_counts)

    # Min-heap: (current_sum, gpu_id)
    # Separate capacity tracking to enforce exactly EXPERTS_PER_GPU per GPU
    heap = [(0, gpu_id) for gpu_id in range(NUM_GPUS)]
    heapq.heapify(heap)
    gpu_assigned: dict[int, list[int]] = {g: [] for g in range(NUM_GPUS)}

    for old_expert in sorted_experts:
        # Pop until we find a GPU that isn't full
        deferred = []
        while True:
            current_sum, gpu_id = heapq.heappop(heap)
            if len(gpu_assigned[gpu_id]) < EXPERTS_PER_GPU:
                break
            deferred.append((current_sum, gpu_id))
        # Push back deferred (full) GPUs
        for item in deferred:
            heapq.heappush(heap, item)

        gpu_assigned[gpu_id].append(int(old_expert))
        heapq.heappush(heap, (current_sum + int(expert_counts[old_expert]), gpu_id))

    # Build the permutation mapping
    mapping = np.zeros(NUM_EXPERTS, dtype=np.int32)
    for gpu_id, assigned in gpu_assigned.items():
        assert len(assigned) == EXPERTS_PER_GPU, (
            f"GPU {gpu_id} got {len(assigned)} experts"
        )
        for slot_idx, old_expert in enumerate(assigned):
            new_expert = gpu_id * EXPERTS_PER_GPU + slot_idx
            mapping[old_expert] = new_expert

    return mapping


def compute_gpu_loads(expert_counts: np.ndarray) -> np.ndarray:
    """Compute per-GPU total token load from expert counts."""
    return expert_counts.reshape(NUM_GPUS, EXPERTS_PER_GPU).sum(axis=1)


def plot_heatmaps(
    before_loads: np.ndarray,
    after_loads: np.ndarray,
    layers: list,
    dataset_name: str,
    output_path: Path,
):
    """Plot before/after heatmaps side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), sharey=True)

    # Same color scale for both
    vmin = min(before_loads.min(), after_loads.min())
    vmax = max(before_loads.max(), after_loads.max())

    gpu_labels = [
        f"{g * EXPERTS_PER_GPU}-{g * EXPERTS_PER_GPU + EXPERTS_PER_GPU - 1}"
        for g in range(NUM_GPUS)
    ]

    im1 = ax1.imshow(
        before_loads.T,
        aspect="auto",
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax1.set_title(f"{dataset_name}\nBEFORE balancing", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("GPU Rank (Expert Range)", fontsize=12)
    ax1.set_yticks(range(NUM_GPUS))
    ax1.set_yticklabels(gpu_labels, fontsize=9)
    tick_step = max(1, len(layers) // 12)
    tick_positions = list(range(0, len(layers), tick_step))
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([layers[i] for i in tick_positions], fontsize=9)

    im2 = ax2.imshow(
        after_loads.T,
        aspect="auto",
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax2.set_title(f"{dataset_name}\nAFTER balancing", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([layers[i] for i in tick_positions], fontsize=9)

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="Token Count")

    # Stats
    before_cv = np.std(before_loads, axis=1) / np.mean(before_loads, axis=1)
    after_cv = np.std(after_loads, axis=1) / np.mean(after_loads, axis=1)
    before_gap = before_loads.max(axis=1) - before_loads.min(axis=1)
    after_gap = after_loads.max(axis=1) - after_loads.min(axis=1)

    stats_text = (
        f"Avg max-min gap: {before_gap.mean():,.0f} → {after_gap.mean():,.0f}  |  "
        f"Avg CV: {before_cv.mean():.4f} → {after_cv.mean():.4f}"
    )
    fig.text(
        0.5,
        0.01,
        stats_text,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved heatmap: {output_path}")


def process_dataset(parquet_path: Path, output_dir: Path):
    """Process a single dataset parquet file."""
    dataset_name = parquet_path.stem.replace("gating_", "")
    print(f"\n{'=' * 60}")
    print(f"Processing: {dataset_name}")
    print(f"{'=' * 60}")

    df = pd.read_parquet(parquet_path)
    print(f"  Shape: {df.shape}")

    expert_counts_per_layer = count_experts_per_layer(df)
    layers = sorted(expert_counts_per_layer.keys())
    print(f"  Layers: {len(layers)} ({min(layers)}..{max(layers)})")

    num_layers = len(layers)
    before_loads = np.zeros((num_layers, NUM_GPUS), dtype=np.int64)
    after_loads = np.zeros((num_layers, NUM_GPUS), dtype=np.int64)
    mappings = {}

    for i, layer in enumerate(layers):
        counts = expert_counts_per_layer[layer]

        # Before: original grouping
        before_loads[i] = compute_gpu_loads(counts)

        # Find balanced mapping
        mapping = greedy_balance(counts)
        mappings[int(layer)] = mapping.tolist()

        # After: apply remapping then compute loads
        remapped_counts = np.zeros_like(counts)
        for old_id in range(NUM_EXPERTS):
            remapped_counts[mapping[old_id]] = counts[old_id]
        after_loads[i] = compute_gpu_loads(remapped_counts)

    # Print stats
    before_cv = np.std(before_loads, axis=1) / np.mean(before_loads, axis=1)
    after_cv = np.std(after_loads, axis=1) / np.mean(after_loads, axis=1)
    before_gap = before_loads.max(axis=1) - before_loads.min(axis=1)
    after_gap = after_loads.max(axis=1) - after_loads.min(axis=1)

    print(
        f"  Before — avg CV: {before_cv.mean():.4f}, avg max-min gap: {before_gap.mean():,.0f}"
    )
    print(
        f"  After  — avg CV: {after_cv.mean():.4f}, avg max-min gap: {after_gap.mean():,.0f}"
    )
    print(
        f"  Improvement — CV: {(1 - after_cv.mean() / before_cv.mean()) * 100:.1f}%, "
        f"gap: {(1 - after_gap.mean() / before_gap.mean()) * 100:.1f}%"
    )

    # Save mapping
    mapping_path = output_dir / f"mapping_{dataset_name}.json"
    with open(mapping_path, "w") as f:
        json.dump(mappings, f)
    print(f"  Saved mapping: {mapping_path}")

    # Build remapped parquet: apply per-layer mapping to expert columns
    # Pre-build a (num_layers, NUM_EXPERTS) lookup table for vectorized remapping
    max_layer = max(layers)
    mapping_lut = np.zeros((max_layer + 1, NUM_EXPERTS), dtype=np.int32)
    for layer, m in mappings.items():
        mapping_lut[layer] = m

    layer_vals = df["layer"].values
    remapped_df = df.copy()
    for col in EXPERT_COLS:
        old_vals = df[col].values
        # Fancy-index: for each row, mapping_lut[layer_of_row, old_expert_of_row]
        remapped_df[col] = mapping_lut[layer_vals, old_vals]

    parquet_out_path = output_dir / f"balanced_{dataset_name}.parquet"
    remapped_df.to_parquet(parquet_out_path, index=False)
    print(f"  Saved balanced parquet: {parquet_out_path}")

    # Plot heatmaps
    heatmap_path = output_dir / f"heatmap_{dataset_name}.png"
    plot_heatmaps(before_loads, after_loads, layers, dataset_name, heatmap_path)

    return mappings


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(PARQUET_DIR.glob("gating_*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    for pf in parquet_files:
        process_dataset(pf, OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
