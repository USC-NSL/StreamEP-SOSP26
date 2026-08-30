"""
helix_pp — Helix-style ILP pipeline placement for SGLang
=========================================================

Ported from the ASPLOS'25 paper:
  "Helix: Serving Large Language Models over Heterogeneous GPUs
   and Network via Max-Flow"  (Mei et al., 2025)

Source: Helix-ASPLOS25/simulator/initial_layout/ilp_layout/ilp_layout.py

This module extracts Helix's core ILP placement logic — the "control plane"
that decides how many layers each GPU should hold in a pipeline-parallel
configuration.  It outputs a partition string compatible with SGLang's
``SGLANG_PP_LAYER_PARTITION`` environment variable.

Simplifications vs. the full Helix ILP:
  - Single pipeline only (no multi-path / model-replication topologies)
  - No partial inference (each layer lives on exactly one GPU)
  - Uniform intra-node network (no per-link bandwidth modeling)
  - Throughput model: profiled or proportional to SM capacity

Usage::

    from helix_pp import compute_pp_partition

    partition = compute_pp_partition(
        num_layers=12,
        gpu_compute_factors=[1.0, 1.0, 0.5, 0.5],
    )
    # partition == [4, 4, 2, 2]
    # export SGLANG_PP_LAYER_PARTITION=4,4,2,2
"""

from helix_pp.placement import compute_pp_partition, solve_placement_ilp
from helix_pp.full_placement import solve_full_helix, FullHelixSolution

__all__ = [
    "compute_pp_partition",
    "solve_placement_ilp",
    "solve_full_helix",
    "FullHelixSolution",
]
