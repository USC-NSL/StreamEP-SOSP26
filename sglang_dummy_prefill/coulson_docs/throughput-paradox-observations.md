# MoE EP Throughput Paradox — Observations & Reproduction

## What We Observed

In EP (Expert Parallelism) serving of a 128-expert MoE model, **more skewed expert routing produces higher output throughput**, opposite to the conventional expectation that imbalance hurts performance.

We tested 4 gating profiles with different routing skew levels. Results are consistent across two experiment batches with different concurrency levels.

### sgl-001/002 (2000 prompts, 500 req/s, input=128, output=512)

| Profile  | Output tok/s | Routing Skew |
|----------|-------------|--------------|
| gptoss   | 5,771       | low          |
| gsm8k    | 5,860       | low-medium   |
| chinese  | 6,099       | high         |
| legal    | **6,221**   | highest      |

### sgl-004 (4000 prompts, 1000 req/s, input=32, output=64)

| Profile  | Output tok/s | Routing Skew |
|----------|-------------|--------------|
| gptoss   | 7,748       | low          |
| gsm8k    | 7,886       | low-medium   |
| chinese  | 7,848       | high         |
| legal    | **8,130**   | highest      |

In both batches, `legal` (most skewed) is the fastest. The effect is 5-8% throughput difference between the most balanced and most skewed profiles.

### What "skewed" means concretely

- **gptoss**: Tokens spread roughly evenly across 128 experts. Each of 16 local experts on each rank gets ~equal share.
- **legal**: Tokens heavily concentrated on a subset of experts. Some local experts get 3-5× more tokens than others. Cross-rank max/min token ratio ~1.25×.
- Profiles were generated from real inference traces (200 samples each) on different datasets.

---

## How to Reproduce

### Prerequisites

- Fork/branch of `sglang-fake-prefill` (this repo)
- A multi-node GPU cluster with ≥8 GPUs (our setup: 4 nodes × 2 L40S, connected via RDMA/IB)
- Model: `lmsys/gpt-oss-120b-bf16` (loaded with `--load-format dummy` — no real weights needed)
- `sglang-fp` conda env with SGLang installed from this repo

### Cluster-Specific Configuration

You MUST adapt these before running:

1. **NCCL env vars** in `experiments/scripts/launch_head_ep_record.sh` and `launch_worker_ep_record.sh`:
   ```bash
   export NCCL_SOCKET_IFNAME=ens1f1np1    # your RDMA interface
   export NCCL_IB_HCA=mlx5_1              # your IB HCA
   export GLOO_SOCKET_IFNAME=ens1f1np1     # your gloo interface
   export NCCL_IB_GID_INDEX=3              # your IB GID index
   ```

2. **`--dist-init-addr`** in both scripts: set to head node IP and a free port.

3. **`--nnodes`** and **`--tp-size`**: must equal total GPU count. `--dp-size` same as `--tp-size` for dp-attention EP.

4. **`--mem-fraction-static`**: start with 0.80, reduce to 0.75 if OOM.

5. **SSH**: workers are launched via `ssh <hostname>`. Set up passwordless SSH between all nodes.

6. **NFS or rsync**: all nodes must see the same repo checkout. Either use shared filesystem or rsync after every code change.

### Step-by-step: Run One Experiment

From the head node, with the repo at `/home/yizhuoliang/sglang-fake-prefill`:

```bash
# 1. Kill any existing sglang processes on ALL nodes
pkill -9 -f "sglang.launch_server" 2>/dev/null
pkill -9 -f "sglang.srt" 2>/dev/null
ssh node2 'pkill -9 -f "sglang.launch_server"; pkill -9 -f "sglang.srt"' 2>/dev/null
# ... repeat for all worker nodes
sleep 5

# 2. Create output directories
PROFILE="gating_profiles/gating_gptoss120b_200.parquet"
EXP_DIR="experiments/sgl-XXX/exp1_dp_gptoss"
mkdir -p "$EXP_DIR/recorder_raw"

# 3. Launch head (node 0)
tmux new-session -d -s sglang-head \
  "bash experiments/scripts/launch_head_ep_record.sh '$PROFILE' '$EXP_DIR/server.log' '$EXP_DIR/recorder_raw' 0.80"

sleep 3

# 4. Launch workers (nodes 1, 2, 3, ...)
tmux new-session -d -s sglang-w1 \
  "ssh node2 'bash /path/to/launch_worker_ep_record.sh 1 \"$PROFILE\" \"$EXP_DIR/server_w1.log\" \"$EXP_DIR/recorder_raw\" 0.80'"
# ... repeat for each worker node with incrementing node-rank

# 5. Wait for health (poll until server responds)
while ! curl -sf http://localhost:30000/health > /dev/null 2>&1; do sleep 5; done

# 6. Start the recorder
curl -X POST http://localhost:30000/start_expert_distribution_record

# 7. Run benchmark
python -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --model lmsys/gpt-oss-120b-bf16 \
  --dataset-name random \
  --random-input-len 32 --random-output-len 64 --random-range-ratio 0.5 \
  --num-prompts 4000 --request-rate 1000 \
  --seed 1 --warmup-requests 1 \
  2>&1 | tee "$EXP_DIR/bench.log"

# 8. Stop & dump recorder
curl -X POST http://localhost:30000/stop_expert_distribution_record
curl -X POST http://localhost:30000/dump_expert_distribution_record
sleep 3

# 9. Kill everything
pkill -9 -f "sglang.launch_server" 2>/dev/null
pkill -9 -f "sglang.srt" 2>/dev/null
```

Repeat for each gating profile:
- `gating_gptoss120b_200.parquet` (low skew)
- `gating_math_gsm8k_200.parquet` (low-medium skew)
- `gating_legal_court_opinions_200.parquet` (high skew)
- `gating_chinese_zhihu_200.parquet` (high skew)

Or use `experiments/sgl-004/run_all.sh` as a template (automates all 4).

### Benchmark Parameters

Two parameter sets have been tested; both show the same paradox:

| Setting | input_len | output_len | num_prompts | request_rate |
|---------|-----------|------------|-------------|--------------|
| sgl-001/002 | 128 | 512 | 2000 | 500 |
| sgl-004 | 32 | 64 | 4000 | 1000 |

`--enable-fake-prefill` skips actual prefill computation, so only output (decode) throughput is meaningful.

### Plotting Results

After collecting `.pt` files from the recorder:

```bash
# Per-experiment plots (7 plots each)
python experiments/plot_moe_kernel_balance.py \
  --pt experiments/sgl-XXX/exp1_dp_gptoss/recorder_raw/*.pt \
  --out-dir experiments/plots/sgl-XXX/exp1_dp_gptoss \
  --peak-pct 0.9

# Cross-profile comparison (6 plots)
python experiments/plot_moe_recorder_compare.py \
  --labels gptoss gsm8k legal chinese \
  --pt-files \
    experiments/sgl-XXX/exp1_dp_gptoss/recorder_raw/*.pt \
    experiments/sgl-XXX/exp2_dp_gsm8k/recorder_raw/*.pt \
    experiments/sgl-XXX/exp3_dp_legal/recorder_raw/*.pt \
    experiments/sgl-XXX/exp4_dp_chinese/recorder_raw/*.pt \
  --out-dir experiments/plots/sgl-XXX/comparison \
  --peak-pct 0.9
```

`--peak-pct 0.9` filters to only the time range where global batch size ≥ 90% of its peak, removing ramp-up and drain phases.

---

## Key Server Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--moe-a2a-backend` | `mooncake-nccl` | NCCL-based EP (all-reduce, no token movement) |
| `--enable-dp-attention --dp-size 8` | — | Each GPU does 1/8 attention, all GPUs do full MoE |
| `--moe-runner-backend` | `triton` | Triton fused_moe kernel (required on L40S) |
| `--profile-driven-gate-path` | parquet path | Replays recorded gating decisions instead of real model gating |
| `--enable-fake-prefill` | — | Skips prefill compute; only decode throughput is real |
| `--load-format dummy` | — | No real weights loaded; model structure only |
| `--expert-distribution-recorder-mode` | `stat` | Records per-step per-rank per-expert token counts |

## Gating Profiles

Parquet files in `gating_profiles/`. Each contains 200 real inference samples' gating decisions (top-k expert indices per token per layer). The `--profile-driven-gate-path` flag makes the server replay these decisions deterministically instead of running the actual gating network, isolating the effect of routing distribution on throughput.

## What to Look For

1. Compare output throughput (tok/s) across the 4 profiles — does the ordering hold on your hardware?
2. The recorder `.pt` files contain per-step, per-rank, per-expert token counts. Plot the max/min ratio across ranks versus throughput.
3. On GPUs with different memory bandwidth (e.g., A100 HBM vs L40S GDDR6), the effect magnitude may change.
