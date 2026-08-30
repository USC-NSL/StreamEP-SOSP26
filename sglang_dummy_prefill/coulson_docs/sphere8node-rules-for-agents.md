# Sphere-16 Cluster Rules for Agents

Rules and constraints for running SGLang on the sphere-16 L40S cluster. **Read this before doing anything.**

---

## 1. Cluster Topology

| Node Rank | Hostname | RoCE IP | GPUs | Role |
|-----------|----------|---------|------|------|
| 0 | sgpu0 | 10.0.0.1 | 2× L40S (46 GB each) | **Head node** — this is where you run commands, API server runs here |
| 1 | sgpu2 | 10.0.0.2 | 2× L40S (46 GB each) | Worker |
| 2 | sgpu3 | 10.0.0.3 | 2× L40S (46 GB each) | Worker |
| 3 | sgpu4 | 10.0.0.4 | 2× L40S (46 GB each) | Worker |
| 4 | sgpu6 | 10.0.0.5 | 2× L40S (46 GB each) | Worker |
| 5 | sgpu7 | 10.0.0.6 | 2× L40S (46 GB each) | Worker |
| 6 | sgpu8 | 10.0.0.7 | 2× L40S (46 GB each) | Worker |
| 7 | sgpu9 | 10.0.0.8 | 2× L40S (46 GB each) | Worker |

**Total: 8 nodes × 2 GPUs = 16× NVIDIA L40S GPUs**

- OS: Ubuntu 24.04 LTS (image: `cuda126-ubuntu2404`)
- Inter-node: RoCE v2 via `ens1f1np1` (Mellanox ConnectX, `mlx5_1`)
- Intra-node: PCIe (no NVLink — L40S does not have NVLink)
- Cluster definition: `experiments/scripts/sphere-16/cluster-definition.py`

---

## 2. Filesystem

**Filesystem is NOT shared across nodes.** Each node has its own local disk.

| Path | Scope | Notes |
|------|-------|-------|
| `/home/yizhuoliang/` | Per-node, **not shared** | Code repos, configs, conda base |
| `/home/yizhuoliang/sglang-fake-prefill/` | Must exist on **all 8 nodes** | Main repo |
| `/home/yizhuoliang/miniconda3/` | Must exist on **all 8 nodes** | Conda installation |

### Hard Rules

- **NEVER** assume files written on sgpu0 are visible on other nodes.
- **ALWAYS rsync** after modifying code, configs, or gating profiles:
  ```bash
  WORKERS=(sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9)
  for w in "${WORKERS[@]}"; do
    rsync -az --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
      /home/yizhuoliang/sglang-fake-prefill/ "$w":/home/yizhuoliang/sglang-fake-prefill/
  done
  ```
- **ALWAYS clear `__pycache__`** on all nodes after rsync to avoid stale bytecode:
  ```bash
  for n in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
    if [ "$n" = "sgpu0" ]; then
      find /home/yizhuoliang/sglang-fake-prefill/python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    else
      ssh "$n" 'find /home/yizhuoliang/sglang-fake-prefill/python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null' &
    fi
  done
  wait
  ```
- **ALWAYS clean old experiment data** on head AND workers before re-running an experiment:
  ```bash
  rm -rf /home/yizhuoliang/sglang-fake-prefill/experiments/<EXP_ID>
  for w in "${WORKERS[@]}"; do
    ssh "$w" "rm -rf /home/yizhuoliang/sglang-fake-prefill/experiments/<EXP_ID>" &
  done
  wait
  ```
- The `sglang-fp` conda env is installed locally on each node. If you `pip install` something, you must repeat on all nodes.

---

## 3. SSH Access

Passwordless SSH is configured from sgpu0 to all workers:

```bash
ssh sgpu2 '<command>'
ssh sgpu3 '<command>'
ssh sgpu4 '<command>'
ssh sgpu6 '<command>'
ssh sgpu7 '<command>'
ssh sgpu8 '<command>'
ssh sgpu9 '<command>'
```

Short hostnames are resolved via `/etc/hosts` or DNS.

---

## 4. Environment Setup

**Copy-paste this block at the start of any shell session or script:**

```bash
eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
```

### What This Gives You

| Component | Detail |
|-----------|--------|
| Python | 3.12 |
| SGLang | Installed from local repo (`pip install -e .`) |
| Conda env name | `sglang-fp` |
| Conda env path | `/home/yizhuoliang/miniconda3/envs/sglang-fp/` |

### No Module System

This cluster does **not** use `module load`. CUDA is bundled with PyTorch inside the conda env.

### If You Need New Python Packages

```bash
# Install on all nodes
for n in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  if [ "$n" = "sgpu0" ]; then
    eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)" && conda activate sglang-fp && pip install <package>
  else
    ssh "$n" 'eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)" && conda activate sglang-fp && pip install <package>' &
  fi
done
wait
```

Do **not** create new conda envs unless specifically asked.

---

## 5. NCCL / Network Configuration

All multi-node SGLang launches require these environment variables:

```bash
export NCCL_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_HCA=mlx5_1
export GLOO_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

| Variable | Why |
|----------|-----|
| `NCCL_SOCKET_IFNAME=ens1f1np1` | Use the RoCE network interface (10.0.0.x subnet) |
| `NCCL_IB_HCA=mlx5_1` | Mellanox HCA device for RDMA |
| `GLOO_SOCKET_IFNAME=ens1f1np1` | Gloo (PyTorch distributed) uses the same interface |
| `NCCL_IB_GID_INDEX=3` | GID index for RoCE v2 |
| `NCCL_DEBUG=WARN` | Suppress verbose NCCL logs; set to `INFO` for debugging |

The launch scripts in `experiments/scripts/sphere-16/` already set all of these.

---

## 6. Running SGLang Server

### Repo and Branch

```
Repo: /home/yizhuoliang/sglang-fake-prefill/
Branch: fake_prefill_coul
```

### Available Launch Scripts (sphere-16)

Located in `experiments/scripts/sphere-16/`:

| Script | Mode | Arguments |
|--------|------|-----------|
| `launch_head_ep_record.sh` | EP16 with recorder | `<gating_profile> <log_file> <record_dir> [mem_frac] [max_running_reqs] [cuda_graph_max_bs]` |
| `launch_worker_ep_record.sh` | EP16 with recorder | `<rank> <gating_profile> <log_file> <record_dir> [mem_frac] [max_running_reqs] [cuda_graph_max_bs]` |
| `run_ep16_skew_comparison.sh` | Full 4-profile experiment | (self-contained, edit EXP_ID inside) |

### Standard Launch: EP16 (DP-attention, mooncake-nccl)

Run everything from sgpu0. Workers are sgpu2/3/4/6/7/8/9 at ranks 1-7.

```bash
WORKERS=(sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9)
WORKER_RANKS=(1 2 3 4 5 6 7)
PROFILE="gating_gptoss120b_sharegpt_200.parquet"
EXP_DIR="experiments/my-exp"
mkdir -p "$EXP_DIR/recorder_raw"

# Head node (rank 0)
tmux new-session -d -s sglang-head \
  "bash experiments/scripts/sphere-16/launch_head_ep_record.sh \
   ./gating_profiles/$PROFILE $EXP_DIR/server_head.log $EXP_DIR/recorder_raw"

sleep 3

# Workers (ranks 1-7)
for i in "${!WORKERS[@]}"; do
  w="${WORKERS[$i]}"
  rank="${WORKER_RANKS[$i]}"
  tmux new-session -d -s "sglang-w$((i+1))" \
    "ssh $w 'bash /home/yizhuoliang/sglang-fake-prefill/experiments/scripts/sphere-16/launch_worker_ep_record.sh \
     $rank ./gating_profiles/$PROFILE $EXP_DIR/server_w${rank}.log $EXP_DIR/recorder_raw'"
done
```

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--model-path` | `lmsys/gpt-oss-120b-bf16` | GPToss 120B model |
| `--load-format dummy` | — | Random weights, skip download. Always use for benchmarking. |
| `--tp-size 16` | — | All 16 GPUs in one TP group |
| `--dp-size 16` | — | DP attention across all GPUs |
| `--enable-dp-attention` | — | Each GPU handles different requests for attention |
| `--moe-a2a-backend mooncake-nccl` | — | Standard EP with NCCL all-reduce (no Mooncake C++ runtime) |
| `--dist-init-addr` | `10.0.0.1:25000` | Head node (sgpu0) RoCE IP |
| `--mem-fraction-static 0.70` | — | Safe for EP16 |
| `--moe-runner-backend triton` | — | Must be triton on L40S |
| `--enable-fake-prefill` | — | Skip real prefill for decode-only benchmarking |
| `--profile-driven-gate-path` | `./gating_profiles/<file>` | Pre-profiled expert routing |
| `--disable-radix-cache` | — | Required with profile-driven gating |
| `--chunked-prefill-size -1` | — | Required with profile-driven gating |
| `--trust-remote-code` | — | Required for GPToss model |

### Server Port and Health

- Port: `30000` (default)
- Startup time with `--load-format dummy`: ~3-5 minutes (CUDA graph capture dominates)
- Health check: `curl -sf http://localhost:30000/health` — returns HTTP 200 with **empty body** when ready

---

## 7. Killing the Server

SGLang spawns many processes. Kill aggressively on **all 8 nodes**:

```bash
pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null
for w in sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  ssh "$w" 'pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null' 2>/dev/null || true
done
for s in sglang-head sglang-w1 sglang-w2 sglang-w3 sglang-w4 sglang-w5 sglang-w6 sglang-w7; do
  tmux kill-session -t "$s" 2>/dev/null || true
done
sleep 5
```

**Always kill and wait 5s before starting a new server** — leftover processes hold GPU memory.

---

## 8. Running Benchmarks

Run from sgpu0 (head node) after the server is healthy:

```bash
eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp

python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model lmsys/gpt-oss-120b-bf16 \
    --dataset-name random \
    --random-input-len 128 --random-output-len 512 \
    --random-range-ratio 0.5 \
    --num-prompts 8000 --request-rate 2000 \
    --seed 1 --warmup-requests 1 \
    2>&1 | tee experiments/<EXP_ID>/bench.log
```

---

## 9. Available Gating Profiles

```
gating_profiles/gating_gptoss120b_sharegpt_200.parquet   # General-purpose (ShareGPT)
gating_profiles/gating_math_gsm8k_200.parquet            # Math (GSM8K)
gating_profiles/gating_legal_court_opinions_200.parquet   # Legal (court opinions)
gating_profiles/gating_chinese_zhihu_200.parquet          # Chinese (Zhihu Q&A)
```

Each profile contains ~200 sequences with per-token per-layer expert routing from real model runs. Used with `--profile-driven-gate-path`. **Always** pair with `--disable-radix-cache --chunked-prefill-size -1`.

---

## 10. Recorder (MoE Kernel Balance + Expert Distribution)

### Enabling

Add `--expert-distribution-recorder-mode stat` to the server launch. This enables both the MoE kernel balance recorder (phase timing) and the expert distribution recorder.

### Start / Stop / Dump via HTTP

```bash
curl -X POST http://localhost:30000/start_expert_distribution_record
# ... run benchmark ...
curl -X POST http://localhost:30000/stop_expert_distribution_record
curl -X POST http://localhost:30000/dump_expert_distribution_record
```

### Collecting from workers

Recorder `.pt` files are saved to `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` on each node. After dump, rsync from workers:

```bash
for w in sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  rsync -az "$w:/home/yizhuoliang/sglang-fake-prefill/$EXP_DIR/recorder_raw/" "$EXP_DIR/recorder_raw/" 2>/dev/null || true
done
```

### Important: Recorder adds overhead

The recorder's `capture_step()` calls `torch.cuda.synchronize()` every decode step, which breaks the overlap scheduler's CPU-GPU pipelining. With the recorder enabled, ITL is inflated by ~40% compared to production. The GPU phase timing percentages are still accurate — only the absolute ITL/throughput numbers are affected.

---

## 11. Common Pitfalls

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Editing code but not rsyncing to workers | Workers run stale code, mysterious failures | Always rsync after edits |
| Not clearing `__pycache__` after rsync | Workers use stale bytecode even after rsync | Always clear pycache on all nodes |
| Not cleaning old experiment data before rerun | Run script skips "already completed" profiles | Delete `experiments/<EXP_ID>` on head AND workers |
| Not killing old server before new one | Port conflict or GPU memory exhaustion | Full kill command (Section 7) + wait 5s |
| Using `--moe-runner-backend` other than `triton` | Other backends not installed on L40S | Always `triton` |
| Health check expecting response body | SGLang returns HTTP 200 with **empty body** | Check HTTP status code, not body content |
| Starting workers before head node | Workers can't connect to `dist-init-addr` | Start head first, workers within ~30s after |
| Forgetting NCCL env vars on manual launch | Hangs at distributed init or uses wrong interface | Use the launch scripts, or export all 6 vars |

---

## 12. Experiment Rules

See `coulson_docs/rules-for-experiments.md`:

- Every experiment needs a unique ID: `sgl-<number>` for SGLang experiments.
- All logs, metrics, recorder dumps go to `experiments/<EXP_ID>/` on the head node (git-ignored).
- Plots and plotting scripts go in `experiments/` (not git-ignored).

---

## 13. Quick Reference

```bash
# Full env setup (run on sgpu0)
eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)" && conda activate sglang-fp

# Check GPU status on all nodes
for n in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  if [ "$n" = "sgpu0" ]; then nvidia-smi; else ssh "$n" nvidia-smi; fi
done

# Check if server is running
curl -sf http://localhost:30000/health && echo "UP" || echo "DOWN"

# Kill all SGLang on all nodes
pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor" 2>/dev/null
for w in sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  ssh "$w" 'pkill -9 -f "sglang"; pkill -9 -f "torch._inductor"' 2>/dev/null || true
done
sleep 5

# Rsync code to all workers
for w in sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  rsync -az --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    /home/yizhuoliang/sglang-fake-prefill/ "$w":/home/yizhuoliang/sglang-fake-prefill/ &
done
wait

# Clear pycache on all nodes
for n in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  if [ "$n" = "sgpu0" ]; then
    find /home/yizhuoliang/sglang-fake-prefill/python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
  else
    ssh "$n" 'find /home/yizhuoliang/sglang-fake-prefill/python -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null' &
  fi
done
wait

# Repo root
cd /home/yizhuoliang/sglang-fake-prefill
```
