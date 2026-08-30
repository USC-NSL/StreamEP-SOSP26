#!/usr/bin/env bash
# config.sh — SHARED cluster / network / workload config for the throughput-vs-ITL
# reproduction. Sourced by run_{head,worker}_{streaminfer,sglang}.sh. System-specific
# settings (conda env, repo, ports, parallelism, mem) live in each run_*_<system>.sh.
# Every value is overridable from the environment; edit here to adapt to a cluster.

# ── Cluster topology ──────────────────────────────────────────────────────────
HEAD_IP="${HEAD_IP:-10.0.0.1}"                  # head node's cluster-network IP (sgpu6)
N_NODE="${N_NODE:-4}"
N_GPU_PER_NODE="${N_GPU_PER_NODE:-2}"
WORLD_SIZE="${WORLD_SIZE:-$(( N_NODE * N_GPU_PER_NODE ))}"   # 8 GPUs
WORKER_HOSTS="${WORKER_HOSTS:-sgpu7 sgpu8 sgpu9}"            # ranks 1..N (head=sgpu6=rank0)

# ── Cluster network (RoCE / NCCL) ─────────────────────────────────────────────
HOST_IFNAME="${HOST_IFNAME:-ens1f1np1}"
# all nodes expose the ens1f1np1 RoCE NIC as mlx5_1 (MLNX_OFED 24.10)
NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_1}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"

# ── Model depth — 1/2 of gpt-oss's 36 layers, for the 1/2-size (8-GPU) cluster ─
# Per-GPU work is what the halving preserves: 36 layers / 16 GPUs == 18 / 8.
export NUM_LAYERS="${NUM_LAYERS:-18}"

# ── Benchmark sweep (BOTH systems) — sharegpt, 5 rates ────────────────────────
RATES="${RATES:-25 50 100 150 200}"             # requests/sec levels
export BENCH_TIME="${BENCH_TIME:-200}"          # seconds of sending per rate (exported:
                                                # parse_results.py derives prompted counts
                                                # = rate*BENCH_TIME for the success-rate column)
BENCH_MAX_CONTEXT_LEN="${BENCH_MAX_CONTEXT_LEN:-2048}"
BENCH_MIN_IN="${BENCH_MIN_IN:-256}";  BENCH_MAX_IN="${BENCH_MAX_IN:-512}"
BENCH_MIN_OUT="${BENCH_MIN_OUT:-256}"; BENCH_MAX_OUT="${BENCH_MAX_OUT:-512}"

# ── Paths ─────────────────────────────────────────────────────────────────────
MINICONDA="${MINICONDA:-$HOME/miniconda3}"
_UTIL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "$_UTIL_DIR/../.." && pwd)}"    # StreamEP-Artifact
RESULTS_BASE="${RESULTS_BASE:-$_UTIL_DIR/results}"                    # per-system subdir

# ── Expert-gating profile (BOTH systems) — replayed routing outcomes ──────────
# Same parquet (DisagMoE profile format) for both systems: StreamInfer takes it
# via --gate-profile-file, sglang via --profile-driven-gate-path. Must exist on
# every node at this path (sglang ranks read it locally).
GATE_PROFILE="${GATE_PROFILE:-$ARTIFACT_ROOT/gating_profiles/balanced_gptoss120b_sharegpt_200.parquet}"
