"""
Full Helix ILP placement with partial inference (layer-splitting) support.

Faithfully ported from: Helix-ASPLOS25/simulator/initial_layout/ilp_layout/ilp_layout.py

Unlike placement.py (single-pipeline simplification), this implements the COMPLETE
Helix max-flow ILP including:
  - Source/sink flow graph with arbitrary topology
  - Per-edge flow and switch variables
  - Partial inference: a layer range on GPU j can OVERLAP with GPU i
    (edge i→j enabled when start_j ≤ end_i < end_j)
  - Multi-path pipelines (different requests can take different routes)

This module is a pure configurator — it does NOT integrate with sglang's PP
(which only supports contiguous non-overlapping partitions). Use it to see what
Helix's full solver recommends for a given heterogeneous cluster.

Variable mapping to original (ilp_layout.py):
    start_i     → var_node_start[f"start_{i}"]          (step 2.1)
    hold_i_k    → var_node_hold_layer[i][f"hold_{i}_{k}"]  (step 2.2)
    flow_i_j    → var_flow[f"flow_{i}_{j}"]             (step 2.3)
    switch_i_j  → var_edge_switch[f"switch_{i}_{j}"]    (step 2.4)
    cond1_i_j   → tmp_var_compute_edge_cond1[...]        (step 2.5, partial only)
    cond2_i_j   → tmp_var_compute_edge_cond2[...]        (step 2.5, partial only)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csc_array

NodeId = Union[int, str]


@dataclass
class NodeSpec:
    max_layers: int
    layer_count_to_throughput: Dict[int, float]
    connected_to: List[NodeId] = field(default_factory=list)


@dataclass
class EdgeSpec:
    from_id: NodeId
    to_id: NodeId
    throughput: float


@dataclass
class FullHelixSolution:
    node_layers: Dict[int, Tuple[int, int]]
    edge_flows: Dict[Tuple[NodeId, NodeId], float]
    total_flow: float
    uses_partial_inference: bool


class _VarRegistry:
    """Maps named variables to indices in a flat scipy vector."""

    def __init__(self):
        self._next = 0
        self._names: Dict[str, int] = {}
        self._lb: List[float] = []
        self._ub: List[float] = []
        self._integrality: List[int] = []

    @property
    def n(self) -> int:
        return self._next

    def add_int(self, name: str, lb: int, ub: int) -> int:
        idx = self._alloc(name, float(lb), float(ub), 1)
        return idx

    def add_binary(self, name: str) -> int:
        return self._alloc(name, 0.0, 1.0, 1)

    def add_continuous(self, name: str, lb: float = 0.0, ub: float = np.inf) -> int:
        return self._alloc(name, lb, ub, 0)

    def idx(self, name: str) -> int:
        return self._names[name]

    def _alloc(self, name: str, lb: float, ub: float, integ: int) -> int:
        assert name not in self._names, f"duplicate variable: {name}"
        i = self._next
        self._names[name] = i
        self._lb.append(lb)
        self._ub.append(ub)
        self._integrality.append(integ)
        self._next += 1
        return i

    def arrays(self):
        return (
            np.array(self._lb),
            np.array(self._ub),
            np.array(self._integrality, dtype=int),
        )


class _ConstraintBuilder:
    """Accumulates Ax ∈ [lb, ub] rows for scipy LinearConstraint."""

    def __init__(self, num_vars: int):
        self._nv = num_vars
        self._rows: List[np.ndarray] = []
        self._lb: List[float] = []
        self._ub: List[float] = []

    def add(self, coeffs: Dict[int, float], lb: float, ub: float):
        row = np.zeros(self._nv)
        for idx, c in coeffs.items():
            row[idx] = c
        self._rows.append(row)
        self._lb.append(lb)
        self._ub.append(ub)

    def eq(self, coeffs: Dict[int, float], rhs: float):
        self.add(coeffs, rhs, rhs)

    def leq(self, coeffs: Dict[int, float], rhs: float):
        self.add(coeffs, -np.inf, rhs)

    def geq(self, coeffs: Dict[int, float], rhs: float):
        self.add(coeffs, rhs, np.inf)

    def build(self) -> LinearConstraint:
        A = np.array(self._rows) if self._rows else np.zeros((0, self._nv))
        return LinearConstraint(
            A=csc_array(A),
            lb=np.array(self._lb) if self._lb else np.array([]),
            ub=np.array(self._ub) if self._ub else np.array([]),
        )

    @property
    def count(self) -> int:
        return len(self._rows)


def solve_full_helix(
    num_layers: int,
    nodes: List[NodeSpec],
    edges: List[EdgeSpec],
    *,
    allow_partial_inference: bool = True,
    remove_redundant: bool = True,
    time_limit: float = 120.0,
) -> FullHelixSolution:
    """Solve the full Helix max-flow ILP.

    Parameters
    ----------
    num_layers :
        Total model layers (L).
    nodes :
        Compute node specs. Index in list = node_id (0-based).
        Each must have ``max_layers`` and ``layer_count_to_throughput``.
        ``connected_to`` is auto-populated from ``edges`` if empty.
    edges :
        Network links. Must include source→node and node→sink edges.
        Use ``build_fully_connected_edges()`` for convenience.
    allow_partial_inference :
        If True, GPU j can start from a layer already covered by GPU i
        (edge enabled when start_j ≤ end_i < end_j).
        If False, edges only enabled when end_i == start_j (strict pipeline).
    remove_redundant :
        Skip constraints implied by the maximization objective (matching
        original Helix's ``remove_redundant`` flag).
    time_limit :
        Solver time limit in seconds.

    Returns
    -------
    FullHelixSolution
    """
    n_compute = len(nodes)
    L = num_layers

    # Build adjacency from edges
    adjacency: Dict[NodeId, List[NodeId]] = {i: [] for i in range(n_compute)}
    adjacency["source"] = []
    adjacency["sink"] = []
    link_map: Dict[Tuple[NodeId, NodeId], EdgeSpec] = {}
    for e in edges:
        link_map[(e.from_id, e.to_id)] = e
        if e.to_id not in adjacency.get(e.from_id, []):
            adjacency.setdefault(e.from_id, []).append(e.to_id)
        if e.from_id not in adjacency.get(e.to_id, []):
            adjacency.setdefault(e.to_id, []).append(e.from_id)

    for i, ns in enumerate(nodes):
        ns.connected_to = list(adjacency.get(i, []))

    # --- Step 1: Variables ---
    V = _VarRegistry()

    # start_i (step 2.1): starting layer index per node
    for i in range(n_compute):
        ub = L - 1 if not remove_redundant else L
        V.add_int(f"start_{i}", 0, ub)

    # hold_i_k (step 2.2): binary, node i holds exactly k layers
    for i in range(n_compute):
        for k in range(1, nodes[i].max_layers + 1):
            V.add_binary(f"hold_{i}_{k}")

    # Enumerate directed edges for flow/switch variables (steps 2.3-2.4)
    directed_edges: List[Tuple[NodeId, NodeId]] = []
    for a, b in link_map:
        if a != "sink" and b != "source":
            directed_edges.append((a, b))
        if b != "sink" and a != "source":
            directed_edges.append((b, a))
    directed_edges = list(set(directed_edges))

    for a, b in directed_edges:
        V.add_continuous(f"flow_{a}_{b}")
        V.add_binary(f"switch_{a}_{b}")

    # cond1/cond2 for partial inference (step 2.5)
    compute_edges = [
        (a, b)
        for (a, b) in directed_edges
        if a != "source" and b != "sink" and a != "sink" and b != "source"
    ]
    if allow_partial_inference:
        for a, b in compute_edges:
            V.add_binary(f"cond1_{a}_{b}")
            V.add_binary(f"cond2_{a}_{b}")

    num_vars = V.n
    C = _ConstraintBuilder(num_vars)

    # Helper: end_layer_expr coefficients for node i
    # end_i = start_i + Σ_k k * hold_i_k
    def end_coeffs(i: int) -> Dict[int, float]:
        d = {V.idx(f"start_{i}"): 1.0}
        for k in range(1, nodes[i].max_layers + 1):
            d[V.idx(f"hold_{i}_{k}")] = float(k)
        return d

    # --- Step 3: Model placement constraints ---
    for i in range(n_compute):
        # (3.1) One-hot: Σ_k hold_i_k = 1
        C.eq(
            {V.idx(f"hold_{i}_{k}"): 1.0 for k in range(1, nodes[i].max_layers + 1)},
            1.0,
        )

        # (3.2) End layer ≤ L: start_i + Σ_k k*hold_i_k ≤ L
        C.leq(end_coeffs(i), float(L))

    # --- Step 4: Flow conservation ---
    for i in range(n_compute):
        flow_in = {}
        flow_out = {}
        for nb in nodes[i].connected_to:
            if nb != "sink" and f"flow_{nb}_{i}" in V._names:
                flow_in[V.idx(f"flow_{nb}_{i}")] = 1.0
            if nb != "source" and f"flow_{i}_{nb}" in V._names:
                flow_out[V.idx(f"flow_{i}_{nb}")] = 1.0
        # Σ flow_in - Σ flow_out = 0
        combined = {}
        for idx, c in flow_in.items():
            combined[idx] = combined.get(idx, 0.0) + c
        for idx, c in flow_out.items():
            combined[idx] = combined.get(idx, 0.0) - c
        C.eq(combined, 0.0)

    # --- Step 5: Node throughput constraint ---
    for i in range(n_compute):
        # Σ flow_in ≤ Σ_k throughput(k) * hold_i_k
        # → Σ flow_in - Σ_k T[k]*hold_i_k ≤ 0
        row = {}
        for nb in nodes[i].connected_to:
            if nb != "sink" and f"flow_{nb}_{i}" in V._names:
                row[V.idx(f"flow_{nb}_{i}")] = 1.0
        for k in range(1, nodes[i].max_layers + 1):
            tp = nodes[i].layer_count_to_throughput.get(k, 0.0)
            row[V.idx(f"hold_{i}_{k}")] = row.get(V.idx(f"hold_{i}_{k}"), 0.0) - tp
        C.leq(row, 0.0)

    # --- Step 6: Edge switch constraints ---
    for a, b in directed_edges:
        sw = V.idx(f"switch_{a}_{b}")

        if a == "source":
            # switch = (start_b == 0)
            # Prop 1: b=1 iff start=0
            si = V.idx(f"start_{b}")
            if not remove_redundant:
                # d >= 1 - s_i
                C.geq({sw: 1.0, si: 1.0}, 1.0)
            # s_i <= L*(1-d)
            C.leq({si: 1.0, sw: float(L)}, float(L))

        elif b == "sink":
            # switch = (end_a == L)
            # Prop 2: b=1 iff end=L
            ec = end_coeffs(a)
            if not remove_redundant:
                # (L-1)*(d+1) >= end_a → (L-1)*d - end_a >= -(L-1)
                row = {sw: float(L - 1)}
                for idx, c in ec.items():
                    row[idx] = row.get(idx, 0.0) - c
                C.geq(row, -float(L - 1))
            # L*d <= end_a → L*d - end_a <= 0
            row2 = {sw: float(L)}
            for idx, c in ec.items():
                row2[idx] = row2.get(idx, 0.0) - c
            C.leq(row2, 0.0)

        else:
            # Compute-to-compute edge
            if allow_partial_inference:
                # switch = cond(start_b ≤ end_a) AND cond(end_a < end_b)
                c1 = V.idx(f"cond1_{a}_{b}")
                c2 = V.idx(f"cond2_{a}_{b}")
                M = L + 1

                # Condition 1: end_a - start_b >= 0 → cond1=1
                # end_a = start_a + Σ k*hold_a_k
                ea = end_coeffs(a)
                sb = V.idx(f"start_{b}")

                # Prop 3: (M)*cond1 >= (end_a - start_b) + 1
                if not remove_redundant:
                    row = {c1: float(M)}
                    for idx, c in ea.items():
                        row[idx] = row.get(idx, 0.0) - c
                    row[sb] = row.get(sb, 0.0) + 1.0
                    C.geq(row, 1.0)
                # M*(1-cond1) >= start_b - end_a → M - M*cond1 - start_b + end_a >= 0
                row2 = {c1: -float(M), sb: -1.0}
                for idx, c in ea.items():
                    row2[idx] = row2.get(idx, 0.0) + c
                C.geq(row2, -float(M))

                # Condition 2: end_b - end_a > 0 → cond2=1
                eb = end_coeffs(b)
                # Prop 4: M*cond2 >= end_b - end_a
                if not remove_redundant:
                    row3 = {c2: float(M)}
                    for idx, c in eb.items():
                        row3[idx] = row3.get(idx, 0.0) - c
                    for idx, c in ea.items():
                        row3[idx] = row3.get(idx, 0.0) + c
                    C.geq(row3, 0.0)
                # end_b - end_a >= 1 - M*(1-cond2)
                row4 = {c2: -float(M)}
                for idx, c in eb.items():
                    row4[idx] = row4.get(idx, 0.0) + c
                for idx, c in ea.items():
                    row4[idx] = row4.get(idx, 0.0) - c
                C.geq(row4, 1.0 - float(M))

                # Prop 5: switch = cond1 AND cond2
                if not remove_redundant:
                    # cond1 + cond2 - switch <= 1
                    C.leq({c1: 1.0, c2: 1.0, sw: -1.0}, 1.0)
                # 2*switch <= cond1 + cond2
                C.leq({sw: 2.0, c1: -1.0, c2: -1.0}, 0.0)

            else:
                # No partial: switch = (end_a == start_b)
                # Prop 6: b=0 if a≠0 where a = start_b - end_a
                ea = end_coeffs(a)
                sb = V.idx(f"start_{b}")

                # L*switch ≤ L + (start_b - end_a)
                row = {sw: float(L), sb: -1.0}
                for idx, c in ea.items():
                    row[idx] = row.get(idx, 0.0) + c
                C.leq(row, float(L))

                # L*switch ≤ L - (start_b - end_a)
                row2 = {sw: float(L), sb: 1.0}
                for idx, c in ea.items():
                    row2[idx] = row2.get(idx, 0.0) - c
                C.leq(row2, float(L))

    # --- Step 7: Edge flow constraint ---
    for a, b in directed_edges:
        # flow_a_b ≤ throughput * switch_a_b
        tp = _edge_throughput(a, b, link_map)
        C.leq({V.idx(f"flow_{a}_{b}"): 1.0, V.idx(f"switch_{a}_{b}"): -tp}, 0.0)

    # --- Objective: maximize Σ flow_source_j → minimize -Σ flow_source_j ---
    obj = np.zeros(num_vars)
    for j in range(n_compute):
        fname = f"flow_source_{j}"
        if fname in V._names:
            obj[V.idx(fname)] = -1.0

    lb, ub, integrality = V.arrays()
    from scipy.optimize import Bounds as ScipyBounds

    constraints = C.build()
    bounds = ScipyBounds(lb=lb, ub=ub)

    result = milp(
        c=obj,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"disp": False, "time_limit": time_limit},
    )

    if not result.success:
        raise ValueError(f"ILP solver failed: {result.message}")

    x = result.x

    # --- Extract solution ---
    node_layers = {}
    for i in range(n_compute):
        s = round(x[V.idx(f"start_{i}")])
        num_k = 0
        for k in range(1, nodes[i].max_layers + 1):
            if round(x[V.idx(f"hold_{i}_{k}")]) == 1:
                num_k = k
                break
        node_layers[i] = (s, s + num_k)

    edge_flows = {}
    for a, b in directed_edges:
        f_val = x[V.idx(f"flow_{a}_{b}")]
        if f_val > 1e-6:
            edge_flows[(a, b)] = f_val

    total_flow = -result.fun

    partial_used = False
    for (a, b), fv in edge_flows.items():
        if a == "source" or b == "sink" or a == "sink" or b == "source":
            continue
        if fv > 1e-6:
            si_b = node_layers[b][0]
            ei_a = node_layers[a][1]
            if ei_a != si_b:
                partial_used = True
                break

    sol = FullHelixSolution(
        node_layers=node_layers,
        edge_flows=edge_flows,
        total_flow=total_flow,
        uses_partial_inference=partial_used,
    )

    _print_solution(sol, n_compute, L)
    return sol


def _edge_throughput(
    a: NodeId, b: NodeId, link_map: Dict[Tuple[NodeId, NodeId], EdgeSpec]
) -> float:
    if (a, b) in link_map:
        return link_map[(a, b)].throughput
    if (b, a) in link_map:
        return link_map[(b, a)].throughput
    raise KeyError(f"No edge between {a} and {b}")


def build_fully_connected_edges(
    n_nodes: int,
    num_layers: int,
    *,
    source_sink_throughput: float = 1e6,
    inter_node_throughput: float = 1e6,
) -> List[EdgeSpec]:
    """Build a fully-connected graph with source and sink.

    Convenience for cases where network is not the bottleneck
    (e.g., single-node intra-PCIe).
    """
    edges = []
    for i in range(n_nodes):
        edges.append(EdgeSpec("source", i, source_sink_throughput))
        edges.append(EdgeSpec(i, "sink", source_sink_throughput))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edges.append(EdgeSpec(i, j, inter_node_throughput))
    return edges


def build_node_specs(
    gpu_compute_factors: List[float],
    num_layers: int,
    max_layers_per_gpu: Optional[int] = None,
) -> List[NodeSpec]:
    """Build NodeSpec list from simple compute factor array."""
    ml = max_layers_per_gpu or num_layers
    specs = []
    for cf in gpu_compute_factors:
        tp = {k: cf / k for k in range(1, ml + 1)}
        specs.append(NodeSpec(max_layers=ml, layer_count_to_throughput=tp))
    return specs


def _print_solution(sol: FullHelixSolution, n: int, L: int):
    print(f"[helix_pp/full] Total max-flow: {sol.total_flow:.6f}")
    print(f"[helix_pp/full] Partial inference used: {sol.uses_partial_inference}")
    print(f"[helix_pp/full] Node assignments:")
    for i in range(n):
        s, e = sol.node_layers[i]
        print(f"  GPU {i}: layers [{s}, {e})  ({e - s} layers)")
    if sol.edge_flows:
        active = [(a, b, f) for (a, b), f in sol.edge_flows.items() if f > 0.01]
        active.sort(key=lambda x: (-x[2]))
        print(f"[helix_pp/full] Active edges ({len(active)}):")
        for a, b, f in active[:20]:
            print(f"  {a} → {b}: flow={f:.4f}")


# --- CLI ---
if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Full Helix ILP (with partial inference)"
    )
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--compute-factors", type=float, nargs="+", required=True)
    parser.add_argument(
        "--partial",
        action="store_true",
        default=False,
        help="Enable partial inference (layer splitting)",
    )
    parser.add_argument("--no-partial", dest="partial", action="store_false")
    parser.add_argument(
        "--inter-node-tp",
        type=float,
        default=1e6,
        help="Inter-node edge throughput (default: very high)",
    )
    parser.add_argument("--time-limit", type=float, default=120.0)
    args = parser.parse_args()

    specs = build_node_specs(args.compute_factors, args.num_layers)
    edges = build_fully_connected_edges(
        len(specs),
        args.num_layers,
        inter_node_throughput=args.inter_node_tp,
    )

    sol = solve_full_helix(
        num_layers=args.num_layers,
        nodes=specs,
        edges=edges,
        allow_partial_inference=args.partial,
        time_limit=args.time_limit,
    )

    print(
        f"\nPartition (for reference only — partial inference not supported in sglang):"
    )
    partition = [
        sol.node_layers[i][1] - sol.node_layers[i][0] for i in range(len(specs))
    ]
    print(f"  {partition}")
