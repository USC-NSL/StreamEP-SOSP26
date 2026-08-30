# Running GPT-OSS with Expert Parallelism, Profile-Driven Gating, and Fake-Prefill

Hardware: 4×A100-SXM4-80GB (SM80)
Model: `openai/gpt-oss-20b` (32 experts, top-4, hidden_size=2880, mxfp4)

---

## Prerequisites

### Environment

```bash
eval "$(conda shell.bash hook)"
conda activate /scratch1/yizhuoli/conda-envs/sglang-fp
module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda
export HF_HOME=/scratch1/yizhuoli/hf_cache
export TMPDIR=/scratch1/yizhuoli/tmp
```

### Gating Profile

Profile-driven gating requires a pre-profiled parquet file. For GPT-OSS 20B:

```
gating_profiles/gating_gptoss_sharegptv3_200.parquet
```

This profile contains 201 requests, 36 layers, ~104k tokens/layer, 128 profiled experts
(projected 4:1 to match the system's 32 experts).

Required parquet columns: `rid`, `token_index`, `layer`, `expert_logical_k0..k3`.

---

## Launch Configurations

All commands are run from the repo root (`sglang-fake-prefill/`).

### 1. EP Only (baseline)

Starts the server with DeepEP expert parallelism, no fake-prefill, no profile gating.
Output is coherent (real weights, real routing).

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8
```

Notes:
- `--disable-cuda-graph` is not needed; `deepep_mode=normal` auto-disables CUDA graphs.
- Startup takes ~5–7 minutes (weight loading + warmup).

### 2. EP + Fake-Prefill

Enables fake-prefill (decode-only mode, prefill is skipped). Output will be garbled
since decode starts from uninitialized KV cache.

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8 \
  --enable-fake-prefill
```

### 3. EP + Profile-Driven Gating

Uses pre-profiled expert routing decisions instead of the model's gate network.
Requires `--disable-radix-cache` and `--chunked-prefill-size -1`.

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8 \
  --profile-driven-gate-path ./gating_profiles/gating_gptoss_sharegptv3_200.parquet \
  --disable-radix-cache \
  --chunked-prefill-size -1
```

### 4. EP + Profile-Driven Gating + Fake-Prefill

The full combination: expert parallelism with profiled routing and fake-prefill.

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8 \
  --enable-fake-prefill \
  --profile-driven-gate-path ./gating_profiles/gating_gptoss_sharegptv3_200.parquet \
  --disable-radix-cache \
  --chunked-prefill-size -1
```

### 5. EP + Profile-Driven Gating + Fake-Prefill + Dummy Weights

Skips downloading/loading real weights (uses random bf16 + zeroed mxfp4 uint8).
Fastest startup; useful for testing the serving pipeline without model access.

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8 \
  --enable-fake-prefill \
  --profile-driven-gate-path ./gating_profiles/gating_gptoss_sharegptv3_200.parquet \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --load-format dummy
```

---

## Pipeline Parallelism (PP) — Multi-Node Alternative to EP

DeepEP only supports single-node (intranode NVLink) on A100. For multi-node setups,
pipeline parallelism is an alternative that splits layers across nodes instead of
splitting experts.

### PP=2, TP=2 (tested, works)

Each PP stage runs 12 of the 24 layers. Each stage uses 2-way tensor parallelism.
Simulates a 2-node × 2-GPU configuration.

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 2 --pp-size 2 \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.8 \
  --enable-fake-prefill \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --load-format dummy
```

Notes:
- No `--moe-a2a-backend deepep` — PP does not use EP; each rank holds all 32 experts
  for its layers.
- PP auto-disables overlap scheduling.
- Custom layer partitioning: set `SGLANG_PP_LAYER_PARTITION=10,14` (or any split
  summing to 24) for uneven memory balancing.

### PP=4, TP=1 (tested, fails)

Crashes with `IndexError: list index out of range` in KV cache memory pool
(`memory_pool.py: v_buffer[layer_id - self.start_layer]`). The attention backend
initializes with `layer_id=0` but non-first PP stages have `start_layer > 0`.
This appears to be an upstream sglang bug, not GPT-OSS specific.

### PP constraints

- PP is incompatible with: overlap scheduling (auto-disabled), speculative decoding,
  mixed chunked prefill.
- `tp_size * pp_size` must equal total GPU count.
- PP does **not** combine with EP in any useful way on A100 (DeepEP forces
  `ep_size = tp_size`, so PP+EP would just reduce EP size).

---

## Weight Dequantization Behavior

GPT-OSS uses mxfp4 quantized expert weights (uint8 packed). On A100 with EP, the
weights are **automatically dequantized to bf16 at model load time** — no code
changes or special flags are needed.

### How it works

In `Mxfp4FusedMoEMethod.process_weights_after_loading()` (in `mxfp4.py`), when the
MoE runner backend is `auto` (the default), both `use_flashinfer` and
`use_triton_kernels` are `False`. This triggers the `else` branch which calls
`upcast_from_mxfp()` to convert mxfp4 uint8 weights to bf16, then deletes the
scale tensors.

After load-time dequant:
- Expert weights are stored as bf16 tensors (no packed uint8).
- At inference time, the Triton MoE kernel runs pure bf16 GEMM — **no per-token
  dequantization overhead**.
- The MoE kernel config shows `E=8,N=2880` (full bf16 intermediate size), not
  `N=1440` (which would indicate packed uint8).

This applies to all launch configurations above (with or without `--load-format dummy`).

---

## Sending Requests

```bash
curl -s http://localhost:30005/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Expected output quality:
- **EP only**: Coherent responses (real weights + real routing).
- **EP + fake-prefill**: Garbled (uninitialized KV cache).
- **EP + profile gating**: Garbled (routing decisions from profile may not match input).
- **EP + profile gating + fake-prefill**: Garbled (both effects combined).
- **+ dummy weights**: Garbled (random/zero weights).

---

## Required Flags Explained

| Flag | Why |
|------|-----|
| `--tp 4 --ep-size 4` | EP requires ep-size == tp-size on single-node |
| `--moe-a2a-backend deepep` | Use DeepEP for all-to-all communication |
| `--deepep-mode normal` | High-throughput NVLink mode (auto-disables CUDA graphs) |
| `--mem-fraction-static 0.8` | Reserve 80% GPU memory for KV cache |
| `--disable-radix-cache` | Required with profile-driven gating (avoids cache interference) |
| `--chunked-prefill-size -1` | Disables chunked prefill (required with profile-driven gating) |
| `--enable-fake-prefill` | Skip real prefill; start decode from empty KV cache |
| `--profile-driven-gate-path` | Path to parquet file with pre-profiled expert routing |
| `--load-format dummy` | Skip real weight loading; use random/zero weights |

---

## NCCL-EP: Multi-Node Expert Parallelism without Mooncake/DeepEP (L40S / A100)

The `mooncake-nccl` backend enables expert parallelism across nodes using standard
NCCL all-reduce — no Mooncake C++ runtime or DeepEP RDMA required. Each GPU holds a
shard of experts, processes only its local experts for all tokens, and NCCL all-reduce
combines the partial results.

### How it works

- `--moe-a2a-backend mooncake-nccl` sets `ep_size = tp_size` (like `mooncake` / `deepep`)
- Uses `FusedMoE` with `StandardDispatcher` (not `DeepEPMoE`)
- `StandardDispatcher` remaps `topk_ids` to local expert indices (-1 for non-local)
- The triton fused_moe kernel processes only local experts, outputs zero for non-local
- `tensor_model_parallel_all_reduce` sums partial results across all GPUs
- No all-to-all communication — just one all-reduce per MoE layer

### GPT-OSS 120B on sgpu4/6/7/8 (4 nodes × 2 L40S = EP8 + DP-Attention)

Launch scripts: `launch_head_ep_nccl.sh` and `launch_worker_ep_nccl.sh`

**DP-Attention is now enabled by default** in the launch scripts. With
`--enable-dp-attention --dp-size 8`, each GPU handles attention for only 1/8 of the
requests, while all GPUs still participate in the full MoE computation via NCCL
all-reduce. This dramatically reduces per-GPU activation memory pressure.

```bash
# From sgpu4:
tmux new-session -d -s sglang-head \
  "bash /home/yizhuoliang/sglang-fake-prefill/launch_head_ep_nccl.sh \
   ./gating_profiles/gating_gptoss120b_200.parquet server_ep_nccl_head.log"

tmux new-session -d -s sglang-w1 \
  "ssh sgpu6 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_ep_nccl.sh 1 \
   ./gating_profiles/gating_gptoss120b_200.parquet server_ep_nccl_w1.log'"

tmux new-session -d -s sglang-w2 \
  "ssh sgpu7 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_ep_nccl.sh 2 \
   ./gating_profiles/gating_gptoss120b_200.parquet server_ep_nccl_w2.log'"

tmux new-session -d -s sglang-w3 \
  "ssh sgpu8 'bash /home/yizhuoliang/sglang-fake-prefill/launch_worker_ep_nccl.sh 3 \
   ./gating_profiles/gating_gptoss120b_200.parquet server_ep_nccl_w3.log'"
```

Key launch parameters:
```
--tp-size 8
--moe-a2a-backend mooncake-nccl
--enable-dp-attention --dp-size 8
--moe-runner-backend triton
--mem-fraction-static 0.80  # optional 3rd arg to head script, 4th to worker script
--nnodes 4 --node-rank <0|1|2|3> --dist-init-addr 10.0.0.1:25000
```

No `--deepep-mode` needed (not applicable for this backend).

**Important:** Use `--mem-fraction-static 0.80` (not 0.85). See "Activation memory
OOM" in Troubleshooting below.

### DP-Attention + Profile-Driven Router Fix

When DP-attention is enabled, `prepare_mlp()` calls `dp_gather_partial()` which
grows `hidden_states` from DP-local size (e.g. 160 tokens) to global size (e.g. 1280
tokens). However, `forward_batch.positions` and `forward_batch.req_pool_indices`
remain DP-local. The profile-driven router in `gpt_oss.py:forward_normal()` needs
global-sized metadata to index the gating profile correctly.

**Fix** (in `gpt_oss.py`): When `is_dp_attention_enabled()` and the position tensor
size doesn't match `num_tokens` (global), we all-reduce-gather `positions` and
`req_pool_indices` from all DP ranks using `memcpy_triton` + `tensor_model_parallel_all_reduce`
(via float32 cast for compatibility).

### Benchmark results (random in=128/out=512, `--mem-fraction-static 0.80`, DP-attention)

With `--enable-fake-prefill`, only output (decode) throughput is meaningful — input
tokens are not actually processed.

**EP8 + DP-Attention (gptoss profile):**

| Metric | 1000 prompts, r=250 | 2000 prompts, r=500 |
|--------|---------------------|---------------------|
| Output throughput | 5143.72 tok/s | 5809.36 tok/s |
| Median ITL | 158.11 ms | 208.32 ms |
| Mean TTFT | 137.45 ms | 11623.11 ms |
| Successful requests | 1000/1000 | 2000/2000 |

**EP8 + DP-Attention (gsm8k profile):**

| Metric | 1000 prompts, r=250 | 2000 prompts, r=500 |
|--------|---------------------|---------------------|
| Output throughput | 5254.99 tok/s | 5898.99 tok/s |
| Median ITL | 155.33 ms | 208.43 ms |
| Mean TTFT | 137.62 ms | 11481.04 ms |
| Successful requests | 1000/1000 | 2000/2000 |

Key improvement vs non-DP-attention: All experiments run at `--mem-fraction-static 0.80`
with no OOM. Previously, 2k/r500 experiments required 0.75 (gptoss) or 0.70 (gsm8k)
without DP-attention.

See `gptoss120b-ep8-vs-pp4tp2-halved-seqlen.md` for full comparison with PP4×TP2.

### When to use mooncake-nccl vs mooncake vs deepep

| Backend | Transport | GPU requirement | Best for |
|---------|-----------|-----------------|----------|
| `deepep` | NVSHMEM RDMA | NVLink (single-node) | A100/H100 intra-node EP |
| `mooncake` | IBGDA RDMA | GPUDirect RDMA (H100) | H100 multi-node EP with deep_gemm |
| `mooncake-nccl` | NCCL all-reduce | Any GPU (RoCE/IB) | L40S/A100 multi-node EP without RDMA |

### Limitations

- Every GPU receives all tokens and processes only its local experts. For very high
  token counts, the NCCL all-reduce payload (`hidden_size × num_tokens × 2 bytes`)
  may become a bottleneck.
- No overlap between communication and computation (the all-reduce is synchronous
  after expert GEMMs).
- For decode workloads (small batches, 1 token/request), these limitations are
  negligible.

---

## Troubleshooting

### DeepEP assertion: `is_token_in_rank.size(0) == x.size(0)`

This was caused by `forward_batch.positions` being padded beyond `hidden_states.shape[0]`
during warmup. Fixed by aligning batch metadata to `num_tokens` in `gpt_oss.py`
`forward_deepep()` and `forward_normal()`.

### `Only use 20 SMs for DeepEP communication`

Performance warning. Can be tuned via `--deepep-config` with a JSON config file
specifying `num_sms`. Safe to ignore for functional testing.

### `Using default MoE kernel config`

Missing Triton autotuning config for the specific expert/hidden-size combo.
Can generate with the benchmarking script referenced in the warning. Safe to ignore.

### Activation memory OOM at high request rates

At high request rates (e.g. 500 r/s), the scheduler admits many concurrent decode
requests. The KV cache pool is pre-allocated in `--mem-fraction-static`, but
**activation memory** (logits tensor, hidden states) is allocated dynamically from
the remaining VRAM. The logits tensor is `batch_size × vocab_size × 4 bytes` (float32)
— for GPT-OSS 120B (vocab_size=201,088), each concurrent request costs ~1.15 MiB of
logits memory alone.

With `--mem-fraction-static 0.85` on L40S (44.4 GiB), only ~6.7 GiB is left for
activations. At ~1,100+ concurrent requests, the logits allocation exceeds this
headroom and the scheduler OOMs fatally.

**Fix:** Use `--mem-fraction-static 0.80`. This frees an extra ~2.2 GiB for
activations, allowing the scheduler's existing `check_decode_mem` + `retract_decode`
mechanism to naturally bound concurrency via KV cache pressure. The scheduler queues
excess requests in its `waiting_queue` and processes them as slots free up — no hard
`--max-running-requests` cap needed.

### NCCL timeout during startup

Increase timeout with `--dist-timeout 600` (default 300s). Common when loading
large models over slow storage.

---

## Advanced MoE Logging

Add `--enable-advanced-logging` to the server launch to collect fine-grained MoE metrics:

1. **GroupedGEMM batch sizes** — CDF of num_tokens per MoE step (sampled every 100th call)
2. **GroupedGEMM execution times** — CDF via async CUDA events (no GPU sync overhead)
3. **Per-iteration global batch size** — Timeline of tokens per forward pass

### Usage

```bash
# Add to launch scripts (launch_head_ep_nccl.sh, etc.):
--enable-advanced-logging
```

Data is dumped to `./advanced_logs/advanced_log_rank{N}.json` on server shutdown.

### Plotting

```bash
python scripts/plot_advanced_logs.py ./advanced_logs/
python scripts/plot_advanced_logs.py ./advanced_logs/ --rank 0 --output-dir ./plots/
```

### Overhead

- **Disabled**: Single `None` check per MoE call (~20ns) — effectively zero.
- **Enabled**: Counter-based sampling + async CUDA events. At typical decode rates (~16k MoE calls/s), overhead is <0.1 ms/s.

See `docs/advanced_logging.md` for full documentation.

---

## MoE Kernel Balance Recorder

The kernel balance recorder captures per-step, per-rank MoE metrics during serving.
Unlike Advanced MoE Logging (which samples), this recorder captures **every decode
step** with zero gaps, producing dense tensors suitable for cross-system comparison.

### What it records

| Metric | Shape | Description |
|--------|-------|-------------|
| `moe_times` | `[decode_steps, world_size]` | Forward pass time per step per rank (ms), measured via CUDA events placed **outside** the torch.compile / CUDA-graph boundary |
| `local_token_counts` | `[decode_steps, num_layers, world_size]` | Number of local token-expert pairs processed per MoE layer per rank per step |
| `batch_sizes` | `[decode_steps, world_size]` | Batch size (number of tokens) per rank per step |
| `timestamps` | `[decode_steps, world_size]` | Wall-clock `time.time()` at the start of each forward step, per rank (float64 epoch seconds) |

All metrics are allgathered across ranks at dump time, so every `.pt` file contains
the full picture from all ranks.

### Enabling the recorder

Add these flags to the server launch command:

```bash
--expert-distribution-recorder-mode stat
```

And set the output directory via environment variable:

```bash
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/path/to/output/dir
```

The launch scripts `launch_head_ep_nccl_record.sh` and `launch_worker_ep_nccl_record.sh`
already include both. They accept `RECORD_DIR` as the 3rd argument:

```bash
# Head (sgpu4):
bash launch_head_ep_nccl_record.sh <gating_profile> <log_file> <record_dir> [mem_frac]

# Workers (sgpu6/7/8):
bash launch_worker_ep_nccl_record.sh <node_rank> <gating_profile> <log_file> <record_dir> [mem_frac]
```

### HTTP workflow

Recording is controlled at runtime via HTTP endpoints on the head node (port 30000):

```bash
# 1. Start recording (resets all buffers)
curl -X POST http://127.0.0.1:30000/start_expert_distribution_record

# 2. Run your benchmark
python -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
  --model lmsys/gpt-oss-120b-bf16 --dataset-name random \
  --random-input-len 128 --random-output-len 512 --random-range-ratio 0.5 \
  --num-prompts 2000 --request-rate 500 --seed 1 --warmup-requests 1

# 3. Stop recording
curl -X POST http://127.0.0.1:30000/stop_expert_distribution_record

# 4. Dump to .pt files (triggers allgather + save)
curl -X POST http://127.0.0.1:30000/dump_expert_distribution_record
```

This produces two files in `RECORD_DIR`:
- `expert_distribution_recorder_<timestamp>.pt` — raw expert distribution stats
- `moe_kernel_balance_<timestamp>.pt` — the kernel balance data (timing, tokens, timestamps)

### Plotting

**Per-experiment plots** (7 plots per experiment):

```bash
python experiments/plot_moe_kernel_balance.py \
  "experiments/sgl-001/exp1_dp_gptoss/recorder_raw/moe_kernel_balance_*.pt" \
  -o experiments/plots/sgl-001/exp1_dp_gptoss/
```

Generates:
- `step_time_timeline.png` — Forward time per rank over time
- `avg_time_per_rank.png` — Average forward time bar chart per rank
- `time_imbalance_timeline.png` — Max/min time ratio over time
- `local_tokens_avg_heatmap.png` — Avg local tokens (layer × rank)
- `local_tokens_rank_imbalance.png` — Token count CV per layer
- `local_tokens_step_rank_heatmap.png` — Token count heatmap (time × rank)
- `cumulative_tokens_timeline.png` — Cumulative local tokens per rank + divergence %

**Cross-experiment comparison** (6 plots):

```bash
python experiments/plot_moe_recorder_compare.py \
  --experiments \
    "DP+gptoss:experiments/sgl-001/exp1_dp_gptoss/recorder_raw" \
    "DP+gsm8k:experiments/sgl-001/exp2_dp_gsm8k/recorder_raw" \
  --output-dir experiments/plots/sgl-001/
```

Generates CDF comparisons (batch size, execution time, local tokens), timeline grids,
and cumulative token comparison across experiments.

When timestamps are present in the data, all timeline plots use **Time (s)** on the
x-axis. Old data without timestamps falls back to step index.

### Implementation details

- **Timing**: CUDA events are recorded in `model_runner.forward()`, outside the
  torch.compile / CUDA-graph boundary. This ensures every step gets a valid timing
  measurement regardless of execution mode (eager, compiled, or graph replay).
- **Token counts**: Written inside `FusedMoE.forward()` using pure tensor ops to a
  pre-allocated GPU buffer (`_local_tokens_gpu_buffer`). These tensor ops are
  CUDA-graph safe. After each forward step, `capture_step()` does a D2H copy.
- **Timestamps**: `time.time()` called at `record_start()` in Python (model runner),
  giving wall-clock time per rank. Cross-rank drift is typically <5ms with NTP sync.
- **Warmup**: The plot scripts skip the first 10 decode steps by default (`-w 10`).
  Override with `-w 0` to include all steps.
