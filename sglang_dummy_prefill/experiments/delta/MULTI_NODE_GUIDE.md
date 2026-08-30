# Multi-Node SGLang on Delta

## Prerequisites

1. A SLURM allocation with N nodes on the `gpuA100x4` partition (4 GPUs per node)
2. Conda environment `sglang` with SGLang installed (see `DELTA_SETUP.md`)
3. `netifaces` package: `pip install netifaces`
4. Model config + tokenizer downloaded locally (weights not needed for `--load-format dummy`)

### Downloading Model Config

Compute nodes have internet access. SSH into any allocated node and run:

```bash
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate sglang

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'lmsys/gpt-oss-120b-bf16',
    local_dir='\$PROJECT/models/gpt-oss-120b-bf16',
    ignore_patterns=['*.safetensors', '*.bin', '*.pt', '*.gguf', '*.h5'],
)
"
```

This downloads only config, tokenizer, and metadata (~MBs). The `$PROJECT` directory is accessible from all nodes.

## Network Configuration

Delta uses **HPE Slingshot** interconnect (not InfiniBand). The high-speed interface is `hsn0`.

Required environment variables (already set in the launch scripts):

```bash
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export SGLANG_LOCAL_IP_NIC=hsn0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Each node's `hsn0` IP can be found with:

```bash
ip -4 addr show hsn0 | grep -oP 'inet \K[0-9.]+'
```

## Architecture

Multi-node SGLang uses a head + workers pattern:

- **Head** (rank 0): Runs the API server on port 30000 and coordinates distributed inference
- **Workers** (rank 1..N-1): Join the distributed group and process their shard of the model

All nodes run the same `sglang.launch_server` command with different `--node-rank` values.
The head node's `hsn0` IP and a chosen port form the `--dist-init-addr` for all nodes to rendezvous.

## Scripts

### `launch_head_ep.sh`

Launches rank 0 (head node). Serves the HTTP API on `0.0.0.0:30000`.

```
Usage: ./launch_head_ep.sh <dist_init_addr> <log_file> [mem_frac]

  dist_init_addr   Head node hsn0 IP + port, e.g. 172.28.86.2:25000
  log_file         Path for server log (relative to ~/sglang/)
  mem_frac         GPU memory fraction for KV cache (default: 0.80)
```

### `launch_worker_ep.sh`

Launches a worker node (rank 1, 2, ...).

```
Usage: ./launch_worker_ep.sh <node_rank> <dist_init_addr> <log_file> [mem_frac]

  node_rank        Integer rank (1, 2, 3, ...)
  dist_init_addr   Same value as head node
  log_file         Path for server log
  mem_frac         GPU memory fraction (default: 0.80)
```

### `run_ep16.sh`

Orchestration script that automates the full launch from the login node.
It discovers nodes from SLURM, resolves the head IP, launches all processes via SSH + tmux, and waits for the health check.

```
Usage: ./run_ep16.sh

  Reads node list from your active SLURM allocation automatically.
  Override with: HEAD=<node> WORKERS="<node2> <node3> <node4>" ./run_ep16.sh
```

## Launching (Automated)

```bash
# Ensure you have a SLURM allocation running
squeue -u $USER

# Launch (auto-discovers nodes from SLURM)
cd ~/sglang/experiments/delta
./run_ep16.sh
```

The script will:
1. Discover allocated nodes via `squeue` + `scontrol show hostnames`
2. Kill any existing SGLang processes on those nodes
3. Resolve the head node's `hsn0` IP for `--dist-init-addr`
4. Launch head in tmux session `sglang-head`
5. Launch workers in tmux sessions `sglang-w1`, `sglang-w2`, `sglang-w3`
6. Poll `http://<head>:30000/health` every 10s (up to 30 min timeout)
7. Print the endpoint URL and example commands on success

## Launching (Manual)

If you prefer to launch manually or need to customize:

```bash
# 1. Find your nodes
scontrol show hostnames $(squeue -u $USER -h -o "%N" | head -1)

# 2. Get head node's hsn0 IP
ssh <head_node> "ip -4 addr show hsn0 | grep -oP 'inet \K[0-9.]+'"

# 3. Launch head (in a tmux session or background)
ssh <head_node> "bash ~/sglang/experiments/delta/launch_head_ep.sh \
    <head_ip>:25000 experiments/my-exp/server_head.log 0.80"

# 4. Launch each worker
ssh <worker1> "bash ~/sglang/experiments/delta/launch_worker_ep.sh \
    1 <head_ip>:25000 experiments/my-exp/server_w1.log 0.80"
ssh <worker2> "bash ~/sglang/experiments/delta/launch_worker_ep.sh \
    2 <head_ip>:25000 experiments/my-exp/server_w2.log 0.80"
ssh <worker3> "bash ~/sglang/experiments/delta/launch_worker_ep.sh \
    3 <head_ip>:25000 experiments/my-exp/server_w3.log 0.80"
```

## Verification

```bash
# Health check
curl http://<head_node>:30000/health

# List models
curl http://<head_node>:30000/v1/models

# Quick generation test
curl http://<head_node>:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model_path>", "prompt": "Hello", "max_tokens": 32}'
```

## Monitoring

```bash
# Attach to head node output
tmux attach -t sglang-head

# Attach to worker N
tmux attach -t sglang-wN

# Tail logs
tail -f ~/sglang/experiments/<exp_dir>/server_head.log
```

## Stopping

```bash
# Kill all SGLang processes across nodes
for node in <node1> <node2> <node3> <node4>; do
    ssh $node 'pkill -9 -f "sglang.launch_server"; pkill -9 -f "sglang.srt"' 2>/dev/null
done

# Clean up tmux sessions
for s in sglang-head sglang-w1 sglang-w2 sglang-w3; do
    tmux kill-session -t $s 2>/dev/null
done
```

## Key Server Arguments

| Argument | Description |
|---|---|
| `--tp-size N` | Tensor parallelism = total GPU count across all nodes |
| `--dp-size N` | Data parallelism (typically same as tp-size for EP) |
| `--ep-size N` | Expert parallelism (typically same as tp-size) |
| `--nnodes N` | Number of nodes |
| `--node-rank R` | This node's rank (0 = head) |
| `--dist-init-addr IP:PORT` | Head node hsn0 IP + rendezvous port |
| `--load-format dummy` | Use random weights (no real model weights needed) |
| `--enable-dp-attention` | Enable data-parallel attention |
| `--enable-dp-lm-head` | Enable data-parallel LM head |
| `--moe-a2a-backend mooncake-nccl` | MoE all-to-all via NCCL |
| `--disable-custom-all-reduce` | Required on Delta — custom all-reduce fails during CUDA graph capture |
| `--mem-fraction-static F` | Fraction of GPU memory for KV cache (0.0–1.0) |
| `--dist-timeout S` | Distributed init timeout in seconds |

## Known Issues

- **`--disable-custom-all-reduce` is required** on Delta. Without it, CUDA graph capture fails on the DP/EP topology.
- **`netifaces` must be installed** — `SGLANG_LOCAL_IP_NIC` needs it to resolve the hsn0 IP within SGLang.
- **MoE triton kernel config missing** — benign warning, falls back to default config.
- **Startup time** — expect 3–10 minutes for multi-node init (distributed rendezvous + TVM compilation + CUDA graph capture).
