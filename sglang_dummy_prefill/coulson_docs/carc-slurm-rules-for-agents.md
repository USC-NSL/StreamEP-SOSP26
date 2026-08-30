# CARC SLURM Cluster Rules for Agents

Rules and constraints for running multi-node SGLang on CARC SLURM allocations. **Read this before doing anything.**

---

## 1. Cluster Topology

CARC SLURM provides batch-allocated multi-node jobs. Node hostnames and IPs change every allocation.

**Example allocation (4 nodes × 2 GPUs = 8 GPUs):**

| Node Rank | Hostname | eth0 IP | ib0 IP | GPUs | Role |
|-----------|----------|---------|--------|------|------|
| 0 | b04-13 | 10.125.75.190 | 10.125.137.190 | 2× A100-80GB-PCIe | **Head** |
| 1 | b05-12 | 10.125.75.210 | 10.125.137.210 | 2× A100-80GB-PCIe | Worker |
| 2 | b05-14 | 10.125.75.212 | 10.125.137.212 | 2× A100-80GB-PCIe | Worker |
| 3 | b10-14 | 10.125.76.4 | 10.125.138.4 | 2× A100-80GB-PCIe | Worker |

**Important:** IB IPs may span different /24 ranges but are within the same /19 subnet. Always verify with `ssh <node> 'ip addr show ib0'`.

---

## 2. Architecture: Login Node vs Compute Nodes

| Component | Login Node (`carcai`) | Compute Nodes (b0x-xx) |
|-----------|----------------------|------------------------|
| tmux | ✅ Available | ✅ Via `module load tmux` |
| GPUs | 4× A100-SXM4-80GB | 2× A100-80GB-PCIe |
| `module` command | ✅ Works directly | Requires `source /etc/profile.d/modules.sh` |
| Port access to compute | Ports on compute nodes reachable directly | Bind with `--host 0.0.0.0` |
| SSH to compute | ✅ | ✅ (between allocated nodes) |
| Role | Run orchestrator scripts, benchmarks | Run SGLang server processes |

**Key rule:** Orchestrator scripts (run_*.sh) run from carcai. They use tmux to manage SSH sessions to compute nodes. tmux is also available on compute nodes via `module load tmux`.

---

## 3. Filesystem Layout

**Both /home1/ and /scratch1/ are NFS-shared across all nodes.** No rsync needed.

| Path | Type | Capacity | Use for |
|------|------|----------|---------|
| `/home1/yizhuoli/` | NFS home | **Very limited quota** | Code repos, small configs only |
| `/scratch1/yizhuoli/` | BeeGFS scratch | ~1.3 PB shared | Conda envs, model weights, HF cache, logs, temp files |

### Hard Rules

- **NEVER** write large files under `/home1/`. You will blow the quota.
- **NEVER** use `/tmp` for anything large. It's a small local tmpfs.
- **ALWAYS** use `/scratch1/yizhuoli/` for heavy data:
  - Conda environments → `/scratch1/yizhuoli/conda-envs/`
  - HuggingFace cache → `/scratch1/yizhuoli/hf_cache`
  - Pip cache → `/scratch1/yizhuoli/pip_cache`
  - Benchmark results → `/scratch1/yizhuoli/bench_results/`
  - Temp files → `/scratch1/yizhuoli/tmp/`
- **DO** use `/tmp/triton_cache` for Triton JIT cache (must be node-local, see Section 7)

### What Lives Where

```
/home1/yizhuoli/
├── sglang-fake-prefill/    # Main repo (NFS-shared, visible on all nodes)
│   ├── gating_profiles/    # Small parquet files
│   ├── coulson_docs/       # Docs
│   └── experiments/        # Scripts + git-ignored run data
└── miniconda3/             # Conda base installation

/scratch1/yizhuoli/
├── conda-envs/sglang-fp/  # The SGLang conda environment
├── hf_cache/               # HuggingFace model downloads
├── bench_results/          # Benchmark output
├── pip_cache/
├── torch_cache/
└── tmp/                    # TMPDIR target
```

---

## 4. Hardware

- **GPUs**: 2× NVIDIA A100-80GB-PCIe per node (SM80, PCIe Gen4)
- **GPU topology**: SYS (cross-NUMA PCIe, no NVLink between GPUs)
- **CPU**: AMD EPYC 7513 32-Core (2 sockets per node)
- **RAM**: 252 GB per node
- **Network**: InfiniBand 200 Gb/s (native IB, not RoCE), single HCA `mlx5_0`
- **IB interface**: `ib0` (10.125.128.0/19 subnet)
- **OS**: Rocky Linux 8.10
- **No sudo access**

### Implications for SGLang

| Feature | Status | Reason |
|---------|--------|--------|
| DeepEP | ❌ Cannot use | Requires NVLink (PCIe only) |
| mooncake-nccl | ✅ Use this | Standard NCCL all-reduce over IB |
| TP within node | ✅ Works (PCIe) | Slower than NVLink but functional |
| PP across nodes | ✅ Works (IB) | Pipeline stages on different nodes |

---

## 5. Environment Setup

**For launch scripts (on compute nodes):**

```bash
source /etc/profile.d/modules.sh
module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda
eval "$(/home1/yizhuoli/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch1/yizhuoli/conda-envs/sglang-fp
export HF_HOME=/scratch1/yizhuoli/hf_cache
export TMPDIR=/scratch1/yizhuoli/tmp
export PYTHONPATH=/home1/yizhuoli/sglang-fake-prefill/python:$PYTHONPATH
```

**For orchestrator scripts (on carcai login node):**

```bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda 2>/dev/null || true
eval "$(/home1/yizhuoli/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch1/yizhuoli/conda-envs/sglang-fp
export PYTHONPATH=/home1/yizhuoli/sglang-fake-prefill/python:$PYTHONPATH
export HF_HOME=/scratch1/yizhuoli/hf_cache
export TMPDIR=/scratch1/yizhuoli/tmp
```

### Why `source /etc/profile.d/modules.sh`?

On compute nodes, `module` is not available in non-login shells. The SSH commands from orchestrator scripts run non-login shells, so `module` would fail without this explicit source.

### If You Need New Python Packages

```bash
pip install --cache-dir /scratch1/yizhuoli/pip_cache <package>
```

Since the conda env is on NFS, installing once makes it available on all nodes.

---

## 6. NCCL / Network Configuration

All multi-node SGLang launches require these environment variables:

```bash
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export SGLANG_LOCAL_IP_NIC=ib0
export TRITON_CACHE_DIR=/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

| Variable | Why |
|----------|-----|
| `NCCL_SOCKET_IFNAME=ib0` | Use InfiniBand interface for NCCL (not eth0) |
| `NCCL_IB_HCA=mlx5_0` | Mellanox HCA device for RDMA |
| `GLOO_SOCKET_IFNAME=ib0` | PyTorch distributed (Gloo) uses same interface |
| `SGLANG_LOCAL_IP_NIC=ib0` | SGLang internal ZMQ/broadcast uses IB IP (requires `netifaces` package) |
| `TRITON_CACHE_DIR=/tmp/triton_cache` | Node-local Triton cache to avoid NFS corruption |
| `NCCL_DEBUG=WARN` | Suppress verbose logs; set to `INFO` for debugging |

**NOT needed:** `NCCL_IB_GID_INDEX` (that's for RoCE v2; CARC uses native InfiniBand).

The launch scripts in `experiments/scripts/carc-slurm-8/` already set all of these.

---

## 7. Running SGLang Server

### Available Configurations

| Config | tp-size | pp-size | dp-size | Backend | Total GPUs |
|--------|---------|---------|---------|---------|------------|
| EP8 | 8 | — | 8 | mooncake-nccl | 8 (4×2) |
| PP4×TP2 | 2 | 4 | — | — | 8 (4×2) |

### Launch Scripts

Located in `experiments/scripts/carc-slurm-8/`:

| Script | Mode | Arguments |
|--------|------|-----------|
| `launch_head_ep.sh` | EP8, no recorder | `<dist_init_addr> <gating_profile> <log_file> [mem_frac] [cuda_graph_max_bs]` |
| `launch_worker_ep.sh` | EP8, no recorder | `<node_rank> <dist_init_addr> <gating_profile> <log_file> [mem_frac] [cuda_graph_max_bs]` |
| `launch_head_ep_record.sh` | EP8, with recorder | `<dist_init_addr> <gating_profile> <log_file> <record_dir> [mem_frac] [cuda_graph_max_bs]` |
| `launch_worker_ep_record.sh` | EP8, with recorder | `<node_rank> <dist_init_addr> <gating_profile> <log_file> <record_dir> [mem_frac] [cuda_graph_max_bs]` |
| `launch_head_pptp.sh` | PP4×TP2, no recorder | same as EP head |
| `launch_worker_pptp.sh` | PP4×TP2, no recorder | same as EP worker |
| `launch_head_pptp_record.sh` | PP4×TP2, with recorder | same as EP head record |
| `launch_worker_pptp_record.sh` | PP4×TP2, with recorder | same as EP worker record |

### Orchestrator Scripts (run from carcai)

| Script | Mode | Arguments |
|--------|------|-----------|
| `run_ep8_quick_test.sh` | EP8, 50 requests | `<head_host> <dist_init_addr> <worker1> <worker2> <worker3>` |
| `run_ep8_skew_comparison.sh` | EP8, 4 profiles | same |
| `run_pptp8_skew_comparison.sh` | PP4×TP2, 4 profiles | same |

### Launching Manually (from carcai)

```bash
SCRIPT_DIR=/home1/yizhuoli/sglang-fake-prefill/experiments/scripts/carc-slurm-8

# Head node (rank 0) — tmux on carcai, SSH to compute
tmux new-session -d -s sglang-head \
  "ssh b04-13 'bash ${SCRIPT_DIR}/launch_head_ep.sh \
   10.125.137.190:25000 ./gating_profiles/gating_gptoss120b_sharegpt_200.parquet \
   experiments/my-test/server_head.log'"

sleep 3

# Workers (ranks 1-3) — tmux on carcai, SSH to each worker
tmux new-session -d -s sglang-w1 \
  "ssh b05-12 'bash ${SCRIPT_DIR}/launch_worker_ep.sh 1 \
   10.125.137.190:25000 ./gating_profiles/gating_gptoss120b_sharegpt_200.parquet \
   experiments/my-test/server_w1.log'"

tmux new-session -d -s sglang-w2 \
  "ssh b05-14 'bash ${SCRIPT_DIR}/launch_worker_ep.sh 2 \
   10.125.137.190:25000 ./gating_profiles/gating_gptoss120b_sharegpt_200.parquet \
   experiments/my-test/server_w2.log'"

tmux new-session -d -s sglang-w3 \
  "ssh b10-14 'bash ${SCRIPT_DIR}/launch_worker_ep.sh 3 \
   10.125.137.190:25000 ./gating_profiles/gating_gptoss120b_sharegpt_200.parquet \
   experiments/my-test/server_w3.log'"
```

### Key Parameters

| Parameter | EP8 | PP4×TP2 | Notes |
|-----------|-----|---------|-------|
| `--mem-fraction-static` | 0.85 | 0.85 | A100 80GB has plenty of headroom |
| `--load-format dummy` | ✅ | ✅ | Always use for benchmarking |
| `--moe-a2a-backend` | mooncake-nccl | — (not applicable) | PP doesn't use EP |
| `--enable-dp-attention` | ✅ | — | EP8 uses DP-attention |
| `--moe-runner-backend` | triton | triton | Required for A100 |
| `--disable-custom-all-reduce` | — | ✅ Required | PCIe SYS topology breaks IPC graph buffers |
| `--dist-init-addr` | head IB IP:25000 | head IB IP:25000 | **Must be IB IP, not eth0** |

### Server Startup Time

- With `--load-format dummy`: ~3-5 minutes (CUDA graph capture dominates)
- Health check: `curl -sf http://<head_host>:30000/health` — returns HTTP 200 with empty body when ready

---

## 8. Killing the Server

Kill all SGLang processes on **all allocated nodes**:

```bash
for n in b04-13 b05-12 b05-14 b10-14; do
    ssh "$n" 'pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null' 2>/dev/null || true
done
for s in sglang-head sglang-w1 sglang-w2 sglang-w3; do
    tmux kill-session -t "$s" 2>/dev/null || true
done
sleep 5
```

**Always kill and wait 5s before starting a new server** — leftover processes hold GPU memory.

---

## 9. Running Benchmarks

Run from carcai (login node), connecting to the head node:

```bash
eval "$(/home1/yizhuoli/miniconda3/bin/conda shell.bash hook)"
conda activate /scratch1/yizhuoli/conda-envs/sglang-fp
export PYTHONPATH=/home1/yizhuoli/sglang-fake-prefill/python:$PYTHONPATH

cd /home1/yizhuoli/sglang-fake-prefill
python -m sglang.bench_serving \
    --backend sglang \
    --host b04-13 --port 30000 \
    --model lmsys/gpt-oss-120b-bf16 \
    --dataset-name random \
    --random-input-len 128 --random-output-len 512 \
    --random-range-ratio 0.5 \
    --num-prompts 2000 --request-rate 500 \
    --seed 1 --warmup-requests 1 \
    2>&1 | tee experiments/<EXP_ID>/bench.log
```

---

## 10. Available Gating Profiles

```
gating_profiles/gating_gptoss120b_sharegpt_200.parquet   # General-purpose (ShareGPT)
gating_profiles/gating_math_gsm8k_200.parquet            # Math (GSM8K)
gating_profiles/gating_legal_court_opinions_200.parquet   # Legal (court opinions)
gating_profiles/gating_chinese_zhihu_200.parquet          # Chinese (Zhihu Q&A)
```

Always pair with `--disable-radix-cache --chunked-prefill-size -1`.

---

## 11. Recorder (MoE Kernel Balance + Expert Distribution)

### Enabling

Use the `_record` launch scripts, or add manually:

```bash
--expert-distribution-recorder-mode stat
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/path/to/output/dir
```

### HTTP Workflow

```bash
curl -X POST http://<head_host>:30000/start_expert_distribution_record
# ... run benchmark ...
curl -X POST http://<head_host>:30000/stop_expert_distribution_record
curl -X POST http://<head_host>:30000/dump_expert_distribution_record
```

### NFS Advantage

Since the filesystem is shared, recorder `.pt` files from all ranks are immediately visible. **No rsync needed** (unlike sphere-16 and aisys-303).

---

## 12. Common Pitfalls

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Running orchestrator from compute node without module | `tmux: command not found` | `module load tmux` or run from carcai |
| Server binds to localhost only | Health check / bench unreachable from carcai | Head scripts use `--host 0.0.0.0` |
| Triton/Inductor cache on NFS | Race conditions, ESTALE errors from concurrent JIT | `TRITON_CACHE_DIR=/tmp/triton_cache` + `TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache` |
| Missing `source /etc/profile.d/modules.sh` | `module: command not found` on compute nodes | Add to every script that runs on compute |
| Using eth0 IP for `--dist-init-addr` | NCCL can't connect (wrong subnet) | Always use the `ib0` IP |
| Using default Triton cache on NFS | ESTALE errors, segfaults from corrupted .so | Set `TRITON_CACHE_DIR=/tmp/triton_cache` |
| Forgetting `SGLANG_LOCAL_IP_NIC=ib0` | ZMQ/broadcast uses eth0, cross-node fails | Set it; requires `netifaces` package |
| Missing `netifaces` package | `SGLANG_LOCAL_IP_NIC` silently falls back to eth0 | `pip install --cache-dir /scratch1/yizhuoli/pip_cache netifaces` |
| Using `--moe-a2a-backend deepep` | DeepEP requires NVLink (PCIe only here) | Use `mooncake-nccl` |
| Writing HF cache to `~/.cache/` | Fills home quota | Set `HF_HOME=/scratch1/yizhuoli/hf_cache` |
| Not killing old server before new one | Port conflict or GPU memory exhaustion | Full kill command (Section 8) + wait 5s |
| Using `--mem-fraction-static 0.95` | OOM during CUDA graph capture | Use 0.85 |

---

## 13. Determining Node IPs for New Allocations

Each SLURM allocation gets different nodes with different IPs. To set up:

```bash
# Check which nodes are allocated
squeue -u yizhuoli    # (from login node)

# Get IB IPs for each node
for node in <node1> <node2> <node3> <node4>; do
    echo -n "$node: "
    ssh "$node" "ip addr show ib0 | grep 'inet ' | awk '{print \$2}'"
done
```

Use the first node's IB IP (with port 25000) as `--dist-init-addr`.

---

## 14. Experiment Rules

See `coulson_docs/rules-for-experiments.md`:

- Every experiment needs a unique ID: `sgl-<number>` for SGLang experiments.
- All logs, metrics, recorder dumps go to `experiments/<EXP_ID>/` (git-ignored).
- Plots and plotting scripts stay in `experiments/` (tracked by git).
- Before each experiment, clear pycache: `find /home1/yizhuoli/sglang-fake-prefill/python -name "__pycache__" -type d -exec rm -rf {} +`

---

## 15. Quick Reference

```bash
# Full env setup (on carcai)
eval "$(/home1/yizhuoli/miniconda3/bin/conda shell.bash hook)" && conda activate /scratch1/yizhuoli/conda-envs/sglang-fp && export PYTHONPATH=/home1/yizhuoli/sglang-fake-prefill/python:$PYTHONPATH HF_HOME=/scratch1/yizhuoli/hf_cache TMPDIR=/scratch1/yizhuoli/tmp

# Check GPU status on all nodes
for n in b04-13 b05-12 b05-14 b10-14; do echo "=== $n ==="; ssh "$n" nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader; done

# Check if server is running
curl -sf http://b04-13:30000/health && echo "UP" || echo "DOWN"

# Kill all SGLang on all nodes
for n in b04-13 b05-12 b05-14 b10-14; do ssh "$n" 'pkill -9 -f "sglang"; pkill -9 -f "torch._inductor"' 2>/dev/null || true; done; for s in sglang-head sglang-w1 sglang-w2 sglang-w3; do tmux kill-session -t "$s" 2>/dev/null || true; done; sleep 5

# Get IB IPs for current allocation
for n in b04-13 b05-12 b05-14 b10-14; do echo -n "$n: "; ssh "$n" "hostname -I" 2>/dev/null; done

# Clear pycache (NFS shared — one command is enough)
find /home1/yizhuoli/sglang-fake-prefill/python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# EP8 quick test
bash experiments/scripts/carc-slurm-8/run_ep8_quick_test.sh b04-13 10.125.137.190:25000 b05-12 b05-14 b10-14

# Full EP8 experiment
bash experiments/scripts/carc-slurm-8/run_ep8_skew_comparison.sh b04-13 10.125.137.190:25000 b05-12 b05-14 b10-14

# Full PP4×TP2 experiment
bash experiments/scripts/carc-slurm-8/run_pptp8_skew_comparison.sh b04-13 10.125.137.190:25000 b05-12 b05-14 b10-14

# Repo root
cd /home1/yizhuoli/sglang-fake-prefill
```
