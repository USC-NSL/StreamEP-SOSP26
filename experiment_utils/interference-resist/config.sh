#!/usr/bin/env bash
# config.sh — SHARED cluster / network / workload / interference config for the
# network-interference-tolerance reproduction. Sourced by run_{head,worker}_{streaminfer,sglang}.sh
# and lib_interference.sh. System-specific settings live in each run_*_<system>.sh.
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

# ── Benchmark — ONE fixed rate; interference is the independent variable ───────
# 75 rps: from the throughput-itl sweep BOTH systems are stable at this rate with
# no interference, so any degradation here is attributable to the network noise
# (sglang already self-collapses at >=150 rps even without interference).
RATE="${RATE:-75}"                              # requests/sec (fixed across all conditions)
BENCH_TIME="${BENCH_TIME:-120}"                 # seconds of load per condition
BENCH_MAX_CONTEXT_LEN="${BENCH_MAX_CONTEXT_LEN:-2048}"
BENCH_MIN_IN="${BENCH_MIN_IN:-256}";  BENCH_MAX_IN="${BENCH_MAX_IN:-512}"
BENCH_MIN_OUT="${BENCH_MIN_OUT:-256}"; BENCH_MAX_OUT="${BENCH_MAX_OUT:-512}"

# ── Interference conditions ───────────────────────────────────────────────────
# The 4 interference modes from the paper (+ a no-interference baseline). Each replays
# the `aws_hpc_metal` De Sensi et al. cloud-noise trace as trace-driven **RDMA** traffic
# (UCX rc) between specific node pairs — reproducing the trace's available-BW pattern on
# the datapath links that NCCL uses. Point-to-point flows are ONE-directional (one
# generator); "bidirectional" runs two generator sets per link. Nodes are indexed n0..nK
# in IFACE_NODES order (n0 = head) and paired (n0,n1) (n2,n3) ... for the all-links modes.
#   none            — baseline, no interference
#   single-link     — n0 -> n1
#   single-link-2x  — n0 -> n1, trace intensity doubled (--congestion-multiplier 2)
#   all-links       — n0 -> n1, n2 -> n3, ...   (half the nodes drive the other half)
#   bidir-all-links — n0 <-> n1, n2 <-> n3, ... (both above, plus the reverse flows)
CONDITIONS="${CONDITIONS:-none single-link single-link-2x all-links bidir-all-links}"

# Interference nodes as host:datapath-ip, indexed n0..n3 in this order (n0 = head).
IFACE_NODES="${IFACE_NODES:-sgpu6:10.0.0.1,sgpu7:10.0.0.2,sgpu8:10.0.0.3,sgpu9:10.0.0.4}"
LINK_CAP_GBPS="${LINK_CAP_GBPS:-200}"           # RoCE link capacity — scales the trace rates
IFACE_TRACE="${IFACE_TRACE:-aws_hpc_metal}"     # De Sensi et al. trace replayed for every mode
IFACE_STREAMS="${IFACE_STREAMS:-8}"             # parallel UCX streams per flow (saturates a link)
IFACE_WARMUP="${IFACE_WARMUP:-25}"              # seconds to let the flows connect+ramp before benchmarking

IFACE_EXTRA_MULT="${IFACE_EXTRA_MULT:-6}"

# Send-pipeline depth per stream: 64 x 64KB msgs in flight + 8MB burst. The tool's
# defaults (16 / 1MB) pace fine on an idle link but silently collapse to ~1MB-in-flight
# per stream once contention inflates RTT — the generator then yields to NCCL instead of
# contending (achieved << offered). Deep pipelines keep achieved ~= offered under load,
# like a real backlogged co-tenant transfer.
IFACE_MAX_OUTSTANDING="${IFACE_MAX_OUTSTANDING:-64}"
IFACE_BURST_BYTES="${IFACE_BURST_BYTES:-8388608}"

# ── UCX / RDMA for the interference generator (see lib_interference.sh, README) ─
# The generator must use rc (RDMA), not TCP, or it won't fair-share against NCCL on
# a lossless RoCE link. These are exported so the tool's set_ucx_env picks them up.
export UCX_LIB_DIR="${UCX_LIB_DIR:-$HOME/ucx118/lib}"          # pinned UCX>=1.18 bundle (1.16 tcp_sockcm is broken)
export UCX_NET_DEV_LIST="${UCX_NET_DEV_LIST:-mlx5_1:1,ens1f1np1}"
export UCX_IB_GID_INDEX="${UCX_IB_GID_INDEX:-3}"
export UCX_SOCKADDR_TLS_PRIORITY="${UCX_SOCKADDR_TLS_PRIORITY:-tcp,sockcm,rdmacm}"

# ── Paths ─────────────────────────────────────────────────────────────────────
MINICONDA="${MINICONDA:-$HOME/miniconda3}"
_UTIL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(cd "$_UTIL_DIR/../.." && pwd)}"    # StreamEP-Artifact
INTERFERENCE_DIR="${INTERFERENCE_DIR:-$ARTIFACT_ROOT/interference_gen}"
RESULTS_BASE="${RESULTS_BASE:-$_UTIL_DIR/results}"                    # per-system subdir

# ── Expert-gating profile (BOTH systems) — replayed routing outcomes ──────────
# Same parquet (DisagMoE profile format) for both systems: StreamInfer takes it
# via --gate-profile-file, sglang via --profile-driven-gate-path. Must exist on
# every node at this path (sglang ranks read it locally).
GATE_PROFILE="${GATE_PROFILE:-$ARTIFACT_ROOT/gating_profiles/balanced_gptoss120b_sharegpt_200.parquet}"
