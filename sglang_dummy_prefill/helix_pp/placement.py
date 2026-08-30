"""
Helix-style ILP placement for pipeline-parallel LLM serving.

Ported from: Helix-ASPLOS25/simulator/initial_layout/ilp_layout/ilp_layout.py

The original Helix ILP solves a max-flow problem over a heterogeneous cluster
graph with source/sink nodes, per-edge flow variables, and edge-switch
constraints that model which GPU-to-GPU links are active.  The objective is
to maximize total token throughput (flow from source to sink).

For a **single pipeline** (no model replication), the formulation simplifies
dramatically: the throughput equals the *minimum* stage throughput across the
pipeline, and we want to maximize that minimum.

ILP Formulation (Helix-simplified, single pipeline)
----------------------------------------------------

Sets:
    N       = set of GPUs  {0, 1, ..., n-1}
    K_i     = {1, 2, ..., max_layers_i}   feasible layer counts for GPU i

Decision variables:
    b[i][k] ∈ {0,1}     GPU i holds exactly k layers   (binary)
    Z       ≥ 0          pipeline throughput lower bound (continuous)

Parameters:
    L                    total number of model layers
    T[i][k]              throughput (tokens/s) of GPU i when holding k layers

Objective:
    maximize  Z

Constraints:
    (C1) Σ_k  b[i][k]           = 1       ∀ i ∈ N       (one config per GPU)
    (C2) Σ_i Σ_k  k · b[i][k]  = L                     (all layers assigned)
    (C3) Z  ≤  Σ_k T[i][k] · b[i][k]     ∀ i ∈ N       (bottleneck bound)
    (C4) b[i][k] ∈ {0, 1}                                (binary)
    (C5) Z ≥ 0                                            (non-negativity)

The throughput model T[i][k] can be:
  (a) Profiled: loaded from CSV, like Helix's model_manager
  (b) Proportional: T[i][k] = compute_factor[i] / k
      (assumes linear scaling: k layers takes k× the time of 1 layer,
       and compute_factor captures relative SM capacity)

Connection to original Helix ILP:
  - Helix's b_{ik} variables (step 2.2)  →  our b[i][k]
  - Helix's s_i (start layer, step 2.1)  →  implicit (computed from partition)
  - Helix's f_{ij} (edge flows, step 2.3) →  collapsed to single Z
  - Helix's d_{ij} (edge switches, step 2.4) →  removed (single path)
  - Helix's node throughput constraint (step 5) →  our C3
  - Helix's model placement constraint (step 3) →  our C1 + C2
  - Helix's objective max Σ f_{source,j} →  our max Z
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csc_array


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def compute_pp_partition(
    num_layers: int,
    gpu_compute_factors: List[float],
    *,
    gpu_max_layers: Optional[List[int]] = None,
    throughput_profiles: Optional[Dict[int, Dict[int, float]]] = None,
    min_layers_per_gpu: int = 1,
) -> List[int]:
    """Compute an optimal PP layer partition for heterogeneous GPUs.

    This is the main entry point.  It runs the Helix-style ILP and returns
    a partition list suitable for ``SGLANG_PP_LAYER_PARTITION``.

    Parameters
    ----------
    num_layers : int
        Total number of hidden layers in the model.
    gpu_compute_factors : list[float]
        Relative compute capacity of each GPU.  1.0 = full speed,
        0.5 = half speed (e.g. 50% SM via CUDA MPS).  Length = pp_size.
    gpu_max_layers : list[int], optional
        Maximum layers each GPU can hold (memory constraint).
        Defaults to ``num_layers`` for every GPU.
    throughput_profiles : dict[int, dict[int, float]], optional
        If provided, maps ``{gpu_idx: {layer_count: throughput}}``.
        Overrides the proportional throughput model for profiled GPUs.
    min_layers_per_gpu : int
        Minimum layers any GPU must hold.  Default 1.

    Returns
    -------
    list[int]
        Layer counts per GPU, e.g. ``[4, 4, 2, 2]``.
        Sum equals ``num_layers``.

    Raises
    ------
    ValueError
        If the problem is infeasible (not enough total capacity).

    Examples
    --------
    >>> compute_pp_partition(12, [1.0, 1.0, 0.5, 0.5])
    [4, 4, 2, 2]
    >>> compute_pp_partition(36, [1.0, 1.0, 1.0, 1.0])
    [9, 9, 9, 9]
    """
    n_gpus = len(gpu_compute_factors)

    if gpu_max_layers is None:
        gpu_max_layers = [num_layers] * n_gpus

    # Clamp max_layers so no GPU claims more than available
    gpu_max_layers = [min(m, num_layers) for m in gpu_max_layers]

    # Build throughput table T[i][k] for each GPU i and layer count k
    throughput_table: Dict[int, Dict[int, float]] = {}
    for i in range(n_gpus):
        throughput_table[i] = {}
        for k in range(min_layers_per_gpu, gpu_max_layers[i] + 1):
            if (
                throughput_profiles
                and i in throughput_profiles
                and k in throughput_profiles[i]
            ):
                # Use profiled throughput (like Helix's model_manager CSV data)
                throughput_table[i][k] = throughput_profiles[i][k]
            else:
                # Proportional model: T[i][k] = compute_factor[i] / k
                # This mirrors Helix's layer_count_2_throughput computation
                # where throughput = min(inference_throughput, nic_throughput)
                # For intra-node (uniform PCIe), NIC is not the bottleneck.
                throughput_table[i][k] = gpu_compute_factors[i] / k
        if not throughput_table[i]:
            raise ValueError(
                f"GPU {i}: no feasible layer count in "
                f"[{min_layers_per_gpu}, {gpu_max_layers[i]}]"
            )

    partition = solve_placement_ilp(
        num_layers=num_layers,
        num_gpus=n_gpus,
        throughput_table=throughput_table,
        min_layers_per_gpu=min_layers_per_gpu,
        gpu_max_layers=gpu_max_layers,
    )
    return partition


def solve_placement_ilp(
    num_layers: int,
    num_gpus: int,
    throughput_table: Dict[int, Dict[int, float]],
    min_layers_per_gpu: int,
    gpu_max_layers: List[int],
) -> List[int]:
    """Solve the Helix-style placement ILP using scipy.optimize.milp.

    This function implements the MILP formulation described in the module
    docstring.  It uses the HiGHS solver bundled with scipy.

    Parameters
    ----------
    num_layers : int
        Total model layers (L).
    num_gpus : int
        Number of GPUs / PP stages (N).
    throughput_table : dict
        ``{gpu_idx: {layer_count: throughput}}``
    min_layers_per_gpu : int
        Minimum layers per GPU.
    gpu_max_layers : list[int]
        Maximum layers per GPU.

    Returns
    -------
    list[int]
        Optimal layer partition.
    """
    # ---------------------------------------------------------------
    # Variable layout:
    #   b[i][k]  for each GPU i and each feasible k
    #   Z        (single continuous variable, appended at the end)
    #
    # We flatten b[i][k] into a 1-D vector.
    # ---------------------------------------------------------------

    # Build index mapping: var_idx[(i, k)] → position in variable vector
    var_idx: Dict[Tuple[int, int], int] = {}
    idx = 0
    for i in range(num_gpus):
        for k in range(min_layers_per_gpu, gpu_max_layers[i] + 1):
            var_idx[(i, k)] = idx
            idx += 1
    num_b_vars = idx
    z_idx = num_b_vars  # Z is the last variable
    num_vars = num_b_vars + 1

    # ---------------------------------------------------------------
    # Objective: maximize Z  →  minimize -Z
    # ---------------------------------------------------------------
    c = np.zeros(num_vars)
    c[z_idx] = -1.0  # maximize Z ↔ minimize -Z

    # ---------------------------------------------------------------
    # Variable bounds and integrality
    # ---------------------------------------------------------------
    # b[i][k] ∈ {0, 1}  →  bounds [0, 1], integrality = 1
    # Z ≥ 0              →  bounds [0, ∞], integrality = 0
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    ub[z_idx] = np.inf  # Z has no upper bound
    integrality = np.ones(num_vars, dtype=int)
    integrality[z_idx] = 0  # Z is continuous

    # ---------------------------------------------------------------
    # Constraints — build as lists then convert to sparse
    # ---------------------------------------------------------------
    A_rows = []
    b_lower = []
    b_upper = []

    # (C1) One-hot: Σ_k b[i][k] = 1  for each GPU i
    for i in range(num_gpus):
        row = np.zeros(num_vars)
        for k in range(min_layers_per_gpu, gpu_max_layers[i] + 1):
            row[var_idx[(i, k)]] = 1.0
        A_rows.append(row)
        b_lower.append(1.0)
        b_upper.append(1.0)

    # (C2) Total layers: Σ_i Σ_k k·b[i][k] = L
    row = np.zeros(num_vars)
    for i in range(num_gpus):
        for k in range(min_layers_per_gpu, gpu_max_layers[i] + 1):
            row[var_idx[(i, k)]] = float(k)
    A_rows.append(row)
    b_lower.append(float(num_layers))
    b_upper.append(float(num_layers))

    # (C3) Bottleneck: Z ≤ Σ_k T[i][k]·b[i][k]  for each GPU i
    #      ↔  Z - Σ_k T[i][k]·b[i][k] ≤ 0
    for i in range(num_gpus):
        row = np.zeros(num_vars)
        row[z_idx] = 1.0
        for k in range(min_layers_per_gpu, gpu_max_layers[i] + 1):
            row[var_idx[(i, k)]] = -throughput_table[i][k]
        A_rows.append(row)
        b_lower.append(-np.inf)
        b_upper.append(0.0)

    # Convert to sparse matrix
    A_eq_ub = np.array(A_rows)
    constraints = LinearConstraint(
        A=csc_array(A_eq_ub),
        lb=np.array(b_lower),
        ub=np.array(b_upper),
    )

    from scipy.optimize import Bounds as ScipyBounds

    bounds = ScipyBounds(lb=lb, ub=ub)

    # ---------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------
    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"disp": False, "time_limit": 60.0},
    )

    if not result.success:
        raise ValueError(
            f"ILP infeasible or solver failed: {result.message}\n"
            f"  num_layers={num_layers}, num_gpus={num_gpus}, "
            f"gpu_max_layers={gpu_max_layers}"
        )

    # ---------------------------------------------------------------
    # Extract solution
    # ---------------------------------------------------------------
    x = result.x
    partition = []
    for i in range(num_gpus):
        assigned_k = 0
        for k in range(min_layers_per_gpu, gpu_max_layers[i] + 1):
            if round(x[var_idx[(i, k)]]) == 1:
                assigned_k = k
                break
        if assigned_k == 0:
            raise ValueError(f"GPU {i}: no layer assignment found in ILP solution")
        partition.append(assigned_k)

    # Sanity check
    assert sum(partition) == num_layers, (
        f"Partition {partition} sums to {sum(partition)}, expected {num_layers}"
    )

    optimal_throughput = x[z_idx]
    print(
        f"[helix_pp] ILP solved: partition={partition}, "
        f"bottleneck_throughput={optimal_throughput:.6f}"
    )

    return partition


# ---------------------------------------------------------------------------
#  Utility: generate the env var string
# ---------------------------------------------------------------------------


def partition_to_env(partition: List[int]) -> str:
    """Convert a partition list to ``SGLANG_PP_LAYER_PARTITION`` value.

    >>> partition_to_env([4, 4, 2, 2])
    '4,4,2,2'
    """
    return ",".join(str(k) for k in partition)


# ---------------------------------------------------------------------------
#  Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Helix-style ILP placement for SGLang PP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 12 layers, 4 GPUs: GPUs 0,1 full speed, GPUs 2,3 half speed
  python -m helix_pp.placement --num-layers 12 --compute-factors 1.0 1.0 0.5 0.5

  # 36 layers, 4 identical GPUs
  python -m helix_pp.placement --num-layers 36 --compute-factors 1.0 1.0 1.0 1.0

  # 12 layers with memory constraints (GPU 2,3 can hold max 4 layers each)
  python -m helix_pp.placement --num-layers 12 --compute-factors 1.0 1.0 0.5 0.5 \\
      --max-layers 12 12 4 4
        """,
    )
    parser.add_argument(
        "--num-layers", type=int, required=True, help="Total model hidden layers"
    )
    parser.add_argument(
        "--compute-factors",
        type=float,
        nargs="+",
        required=True,
        help="Relative compute capacity per GPU (1.0 = full)",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        nargs="+",
        default=None,
        help="Max layers per GPU (memory constraint)",
    )
    parser.add_argument(
        "--min-layers", type=int, default=1, help="Min layers per GPU (default: 1)"
    )
    parser.add_argument(
        "--throughput-json",
        type=str,
        default=None,
        help="JSON file with profiled throughputs: "
        "{gpu_idx: {layer_count: throughput}}",
    )
    args = parser.parse_args()

    profiles = None
    if args.throughput_json:
        with open(args.throughput_json) as f:
            raw = json.load(f)
        profiles = {
            int(k): {int(kk): float(vv) for kk, vv in v.items()} for k, v in raw.items()
        }

    partition = compute_pp_partition(
        num_layers=args.num_layers,
        gpu_compute_factors=args.compute_factors,
        gpu_max_layers=args.max_layers,
        throughput_profiles=profiles,
        min_layers_per_gpu=args.min_layers,
    )

    env_val = partition_to_env(partition)
    print(f"\nResult:")
    print(f"  Partition:                    {partition}")
    print(f"  SGLANG_PP_LAYER_PARTITION:    {env_val}")
    print(f"\nUsage:")
    print(f"  export SGLANG_PP_LAYER_PARTITION={env_val}")
