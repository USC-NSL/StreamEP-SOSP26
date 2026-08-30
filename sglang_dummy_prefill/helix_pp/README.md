# helix_pp — Helix ILP Pipeline Placement for SGLang

Ported from [Helix (ASPLOS'25)](https://arxiv.org/abs/2406.01566):
*"Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow"*
(Mei et al., CMU, ASPLOS 2025)

Source: `Helix-ASPLOS25/simulator/initial_layout/ilp_layout/ilp_layout.py`

## The Problem

LLM serving clusters are increasingly heterogeneous — GPUs differ in VRAM, compute throughput, and network bandwidth. Given a model with L transformer layers and N GPUs with different capabilities, how should layers be assigned to GPUs to maximize serving throughput?

Naively assigning equal layers per GPU bottlenecks the pipeline at the slowest stage. Helix formulates this as a **max-flow MILP** that takes a cluster description as input and outputs the optimal layer-to-GPU mapping.

### Inputs to the ILP

The solver needs three things: what GPUs you have, how they're connected, and what model you're serving.

#### 1. Per-GPU throughput profile: `T[i][k]`

The most critical input. For each GPU `i` and each possible layer count `k`, `T[i][k]` is the throughput (tokens/second) that GPU `i` can sustain when it holds `k` layers.

This captures GPU heterogeneity: an A100 processing 4 layers has different throughput than an RTX 3090 processing 4 layers. More layers per GPU means more computation per token, so throughput drops as `k` increases.

**Where this comes from** (two options):

- **Profiled** (what Helix does in the paper): Run the actual model on each GPU type with varying layer counts, measure latency per token at different batch sizes, take the throughput at peak batch. The original Helix loads these from CSV files (`prompt_bs2time.csv`, `decode_bs2time.csv`) produced by `llm_sys/utils/model_evaluator.py`. The throughput is also capped by the GPU's NIC speed: `T[i][k] = min(compute_throughput(i, k), nic_throughput(i))`.

- **Proportional model** (what we use for simplicity): Assume throughput scales linearly with compute capacity and inversely with layer count: `T[i][k] = compute_factor[i] / k`. The `compute_factor` is a single float capturing the GPU's relative speed (1.0 for full-speed A6000, 0.5 for an A6000 throttled to 50% SM via CUDA MPS). This is a good approximation when network is not the bottleneck and you don't have profiling data.

```python
# Proportional: just pass relative compute power
partition = compute_pp_partition(12, gpu_compute_factors=[1.0, 1.0, 0.5, 0.5])

# Profiled: pass actual measured throughputs per layer count
partition = compute_pp_partition(12, gpu_compute_factors=[1.0, 1.0, 0.5, 0.5],
    throughput_profiles={
        0: {1: 100.0, 2: 55.0, 3: 38.0, 4: 30.0},  # measured tok/s for GPU 0
        1: {1: 100.0, 2: 55.0, 3: 38.0, 4: 30.0},  # GPU 1 (same type)
        2: {1:  50.0, 2: 28.0, 3: 19.0, 4: 15.0},  # GPU 2 (slower)
        3: {1:  50.0, 2: 28.0, 3: 19.0, 4: 15.0},  # GPU 3 (slower)
    })
```

#### 2. Per-GPU memory constraint: `max_layers[i]`

How many layers GPU `i` can hold, determined by its VRAM. A GPU with 24GB VRAM might fit 6 layers of a large MoE model, while a 48GB GPU fits 12. This bounds the `hold_i_k` variables to `k ∈ [1, max_layers[i]]`.

Defaults to `num_layers` (no memory constraint) if not specified.

#### 3. Network topology and bandwidth (full solver only)

For the full solver, you describe how GPUs are connected:

```python
edges = [
    EdgeSpec("source", 0, throughput=1e6),  # source can send to GPU 0
    EdgeSpec(0, 1, throughput=12.5),         # GPU 0 → GPU 1 at 12.5 tok/s
    EdgeSpec(1, "sink", throughput=1e6),     # GPU 1 can deliver to sink
    ...
]
```

Edge throughput = `link_bandwidth / activation_size`. For example, a 100 Gbps InfiniBand link transferring 8KB activations per token gives `100e9 / 8 / 8192 ≈ 1.5M tok/s`. For intra-node PCIe, bandwidth is high enough to not bottleneck, so use a large number.

The simplified solver skips this entirely (assumes infinite bandwidth).

Convenience helper for fully-connected topologies:
```python
edges = build_fully_connected_edges(n_nodes=4, num_layers=12, inter_node_throughput=0.5)
```

#### 4. Model specification

Just the total layer count `L` (e.g., 36 for GPT-OSS-120B, or 12 for our hacked version). The solver is model-agnostic — it doesn't care about hidden dimensions, MoE structure, etc. Those details are baked into the throughput profile `T[i][k]`.

### What the Solver Decides (Outputs)

Given the above inputs, the solver finds:
- **Per-GPU layer range**: `[start_i, end_i)` — which layers each GPU holds
- **Per-edge flow** (full solver): how many tokens/s flow through each link
- **Total throughput**: the maximum achievable serving throughput

### What Makes This Hard

1. **Heterogeneous compute**: `T[i][k]` varies per GPU type — the solver must balance layer counts to equalize stage latencies
2. **Heterogeneous network**: inter-GPU bandwidth limits activation transfer speed between pipeline stages
3. **Multi-path routing**: different requests can take different routes through the cluster (not just one pipeline)
4. **Partial inference**: a layer range on GPU j can *overlap* with GPU i — GPU i computes layers [2,6), GPU j computes layers [4,8), and the edge i→j is valid because `start_j ≤ end_i < end_j`

## Two Implementations

This module provides two solvers at different fidelity levels:

### `placement.py` — Simplified (Single Pipeline)

For the common case: one linear pipeline, no layer splitting, uniform network.

**Assumptions**:
- Single pipeline path: GPU₀ → GPU₁ → ... → GPUₙ
- Each layer on exactly one GPU, contiguous ranges, no overlap
- Network is not a bottleneck (intra-node PCIe / NVLink)
- Throughput = min(stage throughput) — bottleneck determines pipeline speed

**ILP formulation**:
```
maximize  Z

subject to:
  (C1)  Σ_k b[i][k]           = 1      ∀ i     one config per GPU
  (C2)  Σ_i Σ_k k·b[i][k]    = L              all layers assigned
  (C3)  Z ≤ Σ_k T[i][k]·b[i][k]  ∀ i         bottleneck bound
        b[i][k] ∈ {0,1},  Z ≥ 0
```

**Output**: layer counts per GPU → feeds directly into `SGLANG_PP_LAYER_PARTITION`.

**Use when**: single-node PP, uniform interconnect, you just need the partition string.

### `full_placement.py` — Full Helix (Max-Flow with Partial Inference)

Faithful port of the complete Helix MILP including all features from the paper.

**No simplifying assumptions** — models the full network graph:
- Source/sink flow network with arbitrary topology
- Per-edge flow variables with bandwidth constraints
- Multi-path routing (different requests take different GPU chains)
- Partial inference: layer ranges can overlap between adjacent GPUs
- Edge switch constraints with big-M linearization (Props 3-6 from the paper)

**Variables** (mapped to original `ilp_layout.py`):

| Variable | Original name | Type | Meaning |
|----------|--------------|------|---------|
| `start_i` | `var_node_start` (step 2.1) | integer | Starting layer index for GPU i |
| `hold_i_k` | `var_node_hold_layer` (step 2.2) | binary | GPU i holds exactly k layers |
| `flow_a_b` | `var_flow` (step 2.3) | continuous | Token flow on edge a→b |
| `switch_a_b` | `var_edge_switch` (step 2.4) | binary | Edge a→b is active |
| `cond1_a_b` | `tmp_var_compute_edge_cond1` (step 2.5) | binary | `end_a ≥ start_b` (partial) |
| `cond2_a_b` | `tmp_var_compute_edge_cond2` (step 2.5) | binary | `end_b > end_a` (partial) |

**Edge switch cases** (step 6):
- Source → i: enabled iff `start_i == 0` (Prop 1)
- i → Sink: enabled iff `end_i == L` (Prop 2)
- i → j (partial): enabled iff `start_j ≤ end_i < end_j` (Props 3-5)
- i → j (no partial): enabled iff `end_i == start_j` (Prop 6)

**Objective**: `maximize Σ flow(source → j)` — total throughput across all pipeline paths.

**Output**: per-GPU layer ranges `[start, end)`, per-edge flow values, total max-flow, whether partial inference was used.

**Use when**: analyzing what Helix's full solver recommends, studying multi-path / partial inference effects, comparing full vs simplified solutions.

### Comparison

| | `placement.py` (simplified) | `full_placement.py` (full) |
|---|---|---|
| Topology | Single linear pipeline | Arbitrary source/sink flow graph |
| Layer overlap | No (contiguous, disjoint) | Yes (partial inference) |
| Multi-path | No (one pipeline) | Yes (flow splits across paths) |
| Network modeling | None (assumes fast) | Per-edge bandwidth constraints |
| Variables | O(N × max_k) | O(N × max_k + E²) |
| Solve time | <1s for 8 GPUs | Seconds to minutes |
| Output | `[4, 4, 2, 2]` partition list | Layer ranges + edge flows + total flow |
| SGLang compatible | Yes (`SGLANG_PP_LAYER_PARTITION`) | No (informational only) |

**When they agree**: on a single node with uniform PCIe, both produce the same layer distribution. The simplified version is faster and directly usable.

**When they diverge**: with bandwidth-constrained or multi-node topologies, the full version may find multi-path solutions (replicate model on each GPU for independent serving) or partial-inference routes that the simplified version cannot express.

### Example: Same Cluster, Both Solvers

```
4 GPUs: [1.0, 1.0, 0.5, 0.5] compute factors, 12 layers, max 4 layers/GPU

Simplified (placement.py):
  Partition: [4, 4, 2, 2]
  Bottleneck throughput: 0.25

Full (full_placement.py, no partial):
  GPU 0: [4,8)   GPU 1: [8,12)   GPU 2: [2,4)   GPU 3: [0,2)
  Pipeline: GPU3 → GPU2 → GPU0 → GPU1 → sink
  Max flow: 0.25
  Partition: [4, 4, 2, 2]  ← same layer counts, different ordering
```

Both agree on 4,4,2,2 — but the full version also tells you the pipeline *order* and flow routing.

## Usage

### Simplified — for SGLang PP

```python
from helix_pp import compute_pp_partition

partition = compute_pp_partition(
    num_layers=12,
    gpu_compute_factors=[1.0, 1.0, 0.5, 0.5],
)
# → [4, 4, 2, 2]
# export SGLANG_PP_LAYER_PARTITION=4,4,2,2
```

```bash
python -m helix_pp.placement --num-layers 12 --compute-factors 1.0 1.0 0.5 0.5
```

### Full — for analysis

```python
from helix_pp.full_placement import (
    solve_full_helix, build_node_specs, build_fully_connected_edges, NodeSpec, EdgeSpec,
)

specs = build_node_specs([1.0, 1.0, 0.5, 0.5], num_layers=12, max_layers_per_gpu=4)
edges = build_fully_connected_edges(4, 12, inter_node_throughput=0.5)

sol = solve_full_helix(12, specs, edges, allow_partial_inference=True)
# sol.node_layers:  {0: (4,8), 1: (8,12), 2: (2,4), 3: (0,2)}
# sol.edge_flows:   {('source',3): 0.25, (3,2): 0.25, ...}
# sol.total_flow:   0.25
# sol.uses_partial_inference: False
```

```bash
python -m helix_pp.full_placement \
    --num-layers 12 --compute-factors 1.0 1.0 0.5 0.5 \
    --partial --inter-node-tp 0.5
```

### Integration with SGLang

```bash
export SGLANG_PP_LAYER_PARTITION=$(python -m helix_pp.placement \
    --num-layers 12 --compute-factors 1.0 1.0 0.5 0.5 \
    2>/dev/null | grep "SGLANG_PP_LAYER_PARTITION:" | awk '{print $2}')

python -m sglang.launch_server \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --load-format dummy \
    --pp-size 4 --tp-size 1 \
    --json-model-override-args '{"num_hidden_layers": 12}' \
    --enable-fake-prefill ...
```

SGLang reads `SGLANG_PP_LAYER_PARTITION` in `python/sglang/srt/distributed/utils.py:get_pp_indices()` and assigns layers accordingly. No core code changes needed.

## Dependencies

- Python ≥ 3.8
- scipy ≥ 1.9 (for `scipy.optimize.milp` — uses the HiGHS solver)
- numpy

No Gurobi license required. The original Helix uses Gurobi; this port replaces it with scipy's bundled HiGHS solver.

## References

- Paper: https://arxiv.org/abs/2406.01566
- Original code: `Helix-ASPLOS25/simulator/initial_layout/ilp_layout/ilp_layout.py`
- SGLang PP partition hook: `python/sglang/srt/distributed/utils.py:get_pp_indices()`
