# Advanced MoE Logging

SGLang supports advanced logging for MoE (Mixture of Experts) layers, enabling fine-grained analysis of MoE step batch sizes, execution times, and per-iteration global batch sizes.

## Overview

When `--enable-advanced-logging` is passed to the server, the system collects three metrics:

1. **MoE Step Batch Sizes** — The number of token-expert pairs routed to this rank's local experts in each MoE step. In EP mode, this reflects the actual per-rank workload (differs across ranks due to expert routing imbalance). Sampled at regular intervals.
2. **MoE Step Execution Times** — Wall-clock GPU time for each sampled MoE step (up-projection GEMM + activation + down-projection GEMM + combine), measured via async CUDA events (no GPU sync during serving).
3. **Per-Iteration Global Batch Size** — The number of tokens processed in each forward pass (one pass through all layers), recorded every iteration.

## Zero-Overhead Design

When `--enable-advanced-logging` is **not** set:
- The system performs a single `None` check per MoE call (~20ns) — effectively zero overhead.
- No CUDA events are created, no data is collected, no files are written.

When enabled:
- **Counter-based sampling**: Only every N-th MoE call is logged (default N=100). This avoids the overhead of `random.random()` on every call.
- **Async CUDA events**: Event pairs are recorded on the GPU stream without synchronization. `elapsed_time()` is resolved only at dump time (server shutdown).
- **In-memory buffers**: All data is stored in Python lists until dump.

## Usage

### Server Launch

Add `--enable-advanced-logging` to your server launch command:

```bash
python -m sglang.launch_server \
    --model-path /path/to/model \
    --enable-advanced-logging \
    ... other args ...
```

For the GPT-OSS 120B EP8 + DP-attention configuration:

```bash
# In launch_head_ep_nccl.sh, add --enable-advanced-logging
python -m sglang.launch_server \
    --model-path /shared_ssd/models/gpt-oss-120b \
    --tp-size 8 \
    --trust-remote-code \
    --enable-fake-prefill \
    --profile-driven-gate-path ./gating_profiles/gating_gptoss120b_200.parquet \
    --moe-runner-backend triton \
    --enable-dp-attention --dp-size 8 \
    --mem-fraction-static 0.80 \
    --enable-advanced-logging \
    ... other args ...
```

### Data Output

Log data is written to the directory specified by `--advanced-logging-output-dir` (default: `./advanced_logs/`):

```
advanced_logs/
├── advanced_log_rank0.json
├── advanced_log_rank1.json
├── ...
└── advanced_log_rank7.json
```

### Dumping Logs (IMPORTANT)

SGLang's process architecture uses `SIGKILL` for shutdown, which **bypasses** `atexit` handlers. To reliably collect logs from **all** ranks, you must trigger an explicit dump before killing the server:

```bash
# 1. Run your benchmark
python -m sglang.bench_serving ...

# 2. Trigger dump on all nodes (sends SIGUSR1 to scheduler processes)
./scripts/dump_advanced_logs.sh

# 3. Now kill the server
for node in localhost sgpu6 sgpu7 sgpu8; do
  if [ "$node" = "localhost" ]; then
    pkill -9 -f 'python.*sglang' 2>/dev/null || true
  else
    ssh $node "pkill -9 -f 'python.*sglang' 2>/dev/null || true"
  fi
done
```

The dump script sends `SIGUSR1` to all `sglang::scheduler` processes across all nodes. Each scheduler's signal handler writes a snapshot of its collected data. The process continues running after the dump — it is safe to call multiple times.

Without this step, only a random subset of ranks will produce log files (those whose processes happened to exit gracefully before `SIGKILL` arrived).

Each JSON file contains:

```json
{
  "metadata": {
    "rank": 0,
    "tp_size": 8,
    "dp_size": 8,
    "sample_interval": 100,
    "total_moe_calls": 1234567,
    "total_iterations": 15432,
    "start_time": 1741641000.0,
    "end_time": 1741641060.0
  },
  "moe_batch_sizes": [
    {"timestamp": 1741641001.0, "num_tokens": 128},
    ...
  ],
  "moe_step_times_ms": [
    {"timestamp": 1741641001.0, "num_tokens": 128, "time_ms": 2.45},
    ...
  ],
  "iteration_batch_sizes": [
    {
      "iteration": 1,
      "num_tokens": 128,
      "batch_size": 25,
      "forward_mode": "ForwardMode.DECODE",
      "global_num_tokens": [16, 16, 16, 16, 16, 16, 16, 16],
      "timestamp": 1741641001.0
    },
    ...
  ]
}
```

### Plotting

Use the provided plotting script to generate CDF and timeline plots:

```bash
# Plot all ranks
python scripts/plot_advanced_logs.py ./advanced_logs/

# Plot specific rank
python scripts/plot_advanced_logs.py ./advanced_logs/ --rank 0

# Custom output directory
python scripts/plot_advanced_logs.py ./advanced_logs/ --output-dir ./plots/

# Add prefix to output filenames
python scripts/plot_advanced_logs.py ./advanced_logs/ --prefix ep8_dp_gptoss
```

Generated plots:
- `moe_step_batch_size_cdf.png` — CDF of MoE step batch sizes with percentile annotations
- `moe_step_time_cdf.png` — CDF of MoE step execution times
- `iteration_batch_timeline.png` — Timeline of global batch sizes with rolling average

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable-advanced-logging` | `False` | Toggle advanced logging on/off |
| `--advanced-logging-output-dir` | `./advanced_logs/` | Directory for JSON output per rank |
| `sample_interval` | `100` | Log every N-th MoE call (hardcoded, modify in `advanced_logging.py`) |

## Metrics Details

### MoE Step Batch Size
- **What**: `(topk_ids >= 0).sum()` — the number of token-expert pairs routed to this rank's local experts. After EP dispatch remaps expert IDs, local experts have IDs >= 0 and non-local experts are -1.
- **Where**: Instrumented in `python/sglang/srt/layers/moe/moe_runner/runner.py`
- **Sampling**: Every 100th MoE call by default.
- **Note**: This metric differs across EP ranks due to expert routing imbalance, making it useful for load balance analysis. It does NOT include block-alignment padding or overflow bucket tokens.

### MoE Step Execution Time
- **What**: GPU wall time for the full MoE step (w1/w3 gate-up GEMM + activation + w2 down GEMM + weighted combine).
- **How**: Async CUDA events recorded before and after `fused_experts()`. `elapsed_time()` resolved at dump time — no GPU sync during serving.
- **Overhead**: ~3μs per sampled call (event creation + recording). At 1% sampling of ~16k MoE calls/s, this is <0.05 ms/s.

### Per-Iteration Global Batch Size
- **What**: `forward_batch.num_token_non_padded_cpu` — tokens actually forwarded (excluding padding) per model forward pass.
- **Where**: Instrumented in `python/sglang/srt/managers/tp_worker.py` after `ForwardBatch.init_new()`.
- **Sampling**: Every iteration (no sampling — this is already per-pass, not per-layer).
- **Extra**: Also records `global_num_tokens` (per-DP-rank token counts) when DP-attention is enabled.

## Architecture

```
server_args.py          ── enable_advanced_logging: bool
    │                      advanced_logging_output_dir: str
    │
model_runner.py         ── init_advanced_logger() on startup
    │
advanced_logging.py     ── AdvancedMoELogger singleton
    │                      ├── on_moe_step_begin/end()
    │                      ├── on_iteration()
    │                      ├── dump() on atexit / SIGTERM
    │                      └── dump_snapshot() on SIGUSR1
    │
fused_moe.py            ── fused_experts() calls on_moe_step_begin/end
tp_worker.py            ── forward_batch_generation() calls on_iteration
dump_advanced_logs.sh   ── sends SIGUSR1 to all scheduler processes
```
