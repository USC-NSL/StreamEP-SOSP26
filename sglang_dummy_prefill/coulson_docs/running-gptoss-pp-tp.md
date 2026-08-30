# Running GPToss 120B Benchmarks on 4×2 L40S Cluster (TP2×PP4)

## Cluster Overview

| Node | Hostname | RoCE IP | Role |
|------|----------|---------|------|
| 0 | sgpu4 | 10.0.0.1 | Head (API server) |
| 1 | sgpu6 | 10.0.0.2 | Worker |
| 2 | sgpu7 | 10.0.0.3 | Worker |
| 3 | sgpu8 | 10.0.0.4 | Worker |

Each node has **2× NVIDIA L40S GPUs**. We use **TP=2** (within node) and **PP=4** (across nodes).

## Prerequisites

- Conda environment `sglang-fp` with Python 3.12 must be installed on **all 4 nodes**
- The `sglang-fake-prefill` repo (branch `fake_prefill_coul`) must be present at `/home/yizhuoliang/sglang-fake-prefill` on all nodes
- Filesystem is **NOT shared** — any code changes must be rsynced manually:
  ```bash
  rsync -avz --exclude '.git' --exclude '__pycache__' \
    /home/yizhuoliang/sglang-fake-prefill/ sgpu6:/home/yizhuoliang/sglang-fake-prefill/
  rsync -avz --exclude '.git' --exclude '__pycache__' \
    /home/yizhuoliang/sglang-fake-prefill/ sgpu7:/home/yizhuoliang/sglang-fake-prefill/
  rsync -avz --exclude '.git' --exclude '__pycache__' \
    /home/yizhuoliang/sglang-fake-prefill/ sgpu8:/home/yizhuoliang/sglang-fake-prefill/
  ```
- SSH access from sgpu4 to sgpu6/sgpu7/sgpu8 must work without password

## Gating Profiles

Two profiles are available in `gating_profiles/`:

| Profile | File | Description |
|---------|------|-------------|
| General | `gating_gptoss120b_200.parquet` | General-purpose gating profile |
| Math | `gating_math_gsm8k_200.parquet` | Math/GSM8K topic-specific profile |

## Step 1: Launch the Server

### Using launch scripts (recommended)

The scripts handle conda activation, NCCL config, and all server flags.

**Head node (run on sgpu4):**
```bash
# In separate tmux sessions:
tmux new-session -d -s sglang-head \
  "bash /home/yizhuoliang/sglang-fake-prefill/launch_head_pp.sh \
   ./gating_profiles/gating_gptoss120b_200.parquet server_head.log"

tmux new-session -d -s sglang-w1 \
  "ssh sgpu6 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_pp.sh 1 \
   ./gating_profiles/gating_gptoss120b_200.parquet server_w1.log'"

tmux new-session -d -s sglang-w2 \
  "ssh sgpu7 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_pp.sh 2 \
   ./gating_profiles/gating_gptoss120b_200.parquet server_w2.log'"

tmux new-session -d -s sglang-w3 \
  "ssh sgpu8 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_pp.sh 3 \
   ./gating_profiles/gating_gptoss120b_200.parquet server_w3.log'"
```

Replace `gating_gptoss120b_200.parquet` with `gating_math_gsm8k_200.parquet` for the math profile.

### Manual launch (if needed)

On each node, set environment first:
```bash
eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
cd /home/yizhuoliang/sglang-fake-prefill

export NCCL_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_HCA=mlx5_1
export GLOO_SOCKET_IFNAME=ens1f1np1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
```

Then on each node (change `--node-rank` for each):
```bash
python -m sglang.launch_server \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --load-format dummy \
    --tp-size 2 \
    --pp-size 4 \
    --nnodes 4 \
    --node-rank <0|1|2|3> \
    --dist-init-addr 10.0.0.1:25000 \
    --enable-fake-prefill \
    --profile-driven-gate-path ./gating_profiles/<profile>.parquet \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --mem-fraction-static 0.85 \
    --trust-remote-code \
    --log-level-http warning \
    --moe-runner-backend triton \
    --log-level warning
```

## Step 2: Wait for Server Ready

The server takes ~3-5 minutes to start (model loading + CUDA graph capture).

Check readiness:
```bash
curl -v http://localhost:30000/health
# Returns HTTP 200 with empty body when ready
```

You can also watch the head node logs:
```bash
tmux attach -t sglang-head
# Look for "Throughput: X tokens/s" lines — server is ready when these appear
```

## Step 3: Run Benchmarks

From the head node (sgpu4):

```bash
eval "$(/home/yizhuoliang/miniconda3/bin/conda shell.bash hook)"
conda activate sglang-fp
```

**Rate=250, 1000 prompts:**
```bash
python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model lmsys/gpt-oss-120b-bf16 \
    --dataset-name random \
    --random-input-len 256 --random-output-len 1024 \
    --random-range-ratio 0.5 \
    --num-prompts 1000 --request-rate 250 \
    2>&1 | tee logs/bench_result.log
```

**Rate=500, 2000 prompts:**
```bash
python -m sglang.bench_serving \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --model lmsys/gpt-oss-120b-bf16 \
    --dataset-name random \
    --random-input-len 256 --random-output-len 1024 \
    --random-range-ratio 0.5 \
    --num-prompts 2000 --request-rate 500 \
    2>&1 | tee logs/bench_result.log
```

## Step 4: Shutdown

Kill all server processes on all nodes:
```bash
pkill -9 -f "sglang" 2>/dev/null
pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null
ssh sgpu6 'pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null'
ssh sgpu7 'pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null'
ssh sgpu8 'pkill -9 -f "sglang" 2>/dev/null; pkill -9 -f "torch._inductor.compile_worker" 2>/dev/null'
tmux kill-session -t sglang-head 2>/dev/null
tmux kill-session -t sglang-w1 2>/dev/null
tmux kill-session -t sglang-w2 2>/dev/null
tmux kill-session -t sglang-w3 2>/dev/null
```

## Important Notes

### Memory Configuration
- `--mem-fraction-static 0.85` is the maximum safe value for TP2×PP4 on L40S
- Higher values (0.90, 0.95) cause OOM during CUDA graph capture on PP stages 2-3
- This leaves ~4.4-4.9 GB free per GPU for CUDA graphs

### Bug Fix Applied
A bug was fixed in `python/sglang/srt/layers/attention/triton_backend.py` (line 102-104):
- **Before**: `get_value_buffer(0)` — hardcoded layer 0, crashes on PP stages where `start_layer > 0`
- **After**: `get_value_buffer(model_runner.token_to_kv_pool.start_layer)` — uses correct start layer for each PP stage

This fix must be present on **all nodes**. If you reinstall sglang, rsync the fix.

### NCCL Configuration
The cluster uses RoCE networking via `ens1f1np1` interface. All NCCL environment variables are set in the launch scripts. Key settings:
- `NCCL_SOCKET_IFNAME=ens1f1np1` — RoCE network interface
- `NCCL_IB_HCA=mlx5_1` — InfiniBand HCA device
- `NCCL_IB_GID_INDEX=3` — GID index for RoCE v2

## Benchmark Results Summary

| Configuration | Profile | Rate | Output Throughput (tok/s) | Median ITL (ms) | P99 ITL (ms) | Median TTFT (ms) |
|--------------|---------|------|--------------------------|----------------|-------------|-----------------|
| TP8 | general | 250 | 1,246 | 387 | 2,317 | — |
| TP8 | general | 500 | 1,292 | 533 | 2,463 | — |
| TP2×PP4 | general | 250 | 2,871 | 286 | 456 | 87 |
| TP2×PP4 | general | 500 | 4,290 | 333 | 736 | 73 |
| TP2×PP4 | math | 250 | 3,179 | 265 | 418 | 42 |
| TP2×PP4 | math | 500 | 4,500 | 318 | 723 | 60 |
| DisagMOE | — | 250 | 4,574 | 210 | 247 | — |
| DisagMOE | — | 500 | 5,222 | 226 | 267 | — |

**Key findings:**
- TP2×PP4 provides **2.3-3.3× throughput improvement** over TP8
- Math profile performs slightly better than general profile (~5-10% better ITL)
- DisagMOE still leads by ~20-30% throughput and ~30% better latency vs best SGLang PP config

## Files Reference

| File | Purpose |
|------|---------|
| `launch_head_pp.sh` | Head node launch script (TP2×PP4) |
| `launch_worker_pp.sh` | Worker node launch script (TP2×PP4) |
| `launch_head.sh` | Head node launch script (TP8, legacy) |
| `launch_worker.sh` | Worker node launch script (TP8, legacy) |
| `plot_disagmoe.py` | Generate comparison plot |
| `sglang_vs_disagmoe.png` | Comparison plot image |
| `logs/` | Benchmark and server logs |
