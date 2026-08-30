# sgl-022: Recorder Overhead Measurement

## Goal

Quantify the throughput impact of the MoE kernel balance recorder by running each profile twice: once with the recorder enabled (for phase breakdown + expert distribution plots), once without (for true production throughput). The recorder's `capture_step()` calls `torch.cuda.synchronize()` every decode step, which breaks the overlap scheduler's CPU-GPU pipelining and creates GPU idle bubbles.

## Configuration (all 8 runs share these)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `lmsys/gpt-oss-120b-bf16` | |
| Load format | `dummy` | Random weights, skip download |
| Parallelism | EP16: TP=16, DP=16, DP-attention | `--tp-size 16 --dp-size 16 --enable-dp-attention` |
| A2A backend | `mooncake-nccl` | Standard EP with NCCL all-reduce, no Mooncake C++ runtime |
| MoE runner | `triton` | Only backend available on L40S |
| Fake prefill | Enabled | `--enable-fake-prefill` — skips real prefill, decode-only benchmarking |
| Profile-driven gating | Enabled | `--profile-driven-gate-path <profile>.parquet` with `--disable-radix-cache --chunked-prefill-size -1` |
| Memory fraction | 0.70 | `--mem-fraction-static 0.70` |
| Num nodes | 8 | `--nnodes 8` |
| Dist init | `10.0.0.1:25000` | Head node RoCE IP |
| Dist timeout | 1800s | |
| CUDA expandable segments | Enabled | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

### Benchmark client parameters

| Parameter | Value |
|-----------|-------|
| Dataset | `random` |
| Num prompts | 8000 |
| Request rate | 2000 req/s |
| Input length | 128 tokens |
| Output length | 512 tokens |
| Random range ratio | 0.5 (actual lengths uniformly sampled in `[len×0.5, len]`) |
| Seed | 1 |
| Warmup requests | 1 |

### Gating profiles (4 profiles)

| Profile name | Parquet file |
|-------------|-------------|
| ShareGPT | `gating_gptoss120b_sharegpt_200.parquet` |
| GSM8K | `gating_math_gsm8k_200.parquet` |
| Legal | `gating_legal_court_opinions_200.parquet` |
| Chinese | `gating_chinese_zhihu_200.parquet` |

## Runs (8 total)

| Run | Profile  | Recorder | Deliverables |
|-----|----------|----------|-------------|
| 1   | ShareGPT | ON       | Throughput + ITL + phase breakdown plots + expert distribution plots and moe bs/exec-time heatmaps |
| 2   | ShareGPT | OFF      | Throughput + ITL only |
| 3   | GSM8K    | ON       | Throughput + ITL + phase breakdown plots + expert distribution plots and moe bs/exec-time heatmaps |
| 4   | GSM8K    | OFF      | Throughput + ITL only |
| 5   | Legal    | ON       | Throughput + ITL + phase breakdown plots + expert distribution plots and moe bs/exec-time heatmaps |
| 6   | Legal    | OFF      | Throughput + ITL only |
| 7   | Chinese  | ON       | Throughput + ITL + phase breakdown plots + expert distribution plots and moe bs/exec-time heatmaps |
| 8   | Chinese  | OFF      | Throughput + ITL only |

### Recorder ON runs (runs 1, 3, 5, 7)

- Launch server with `--expert-distribution-recorder-mode stat`
- After server is healthy, call `POST /start_expert_distribution_record`
- Run benchmark
- Call `POST /stop_expert_distribution_record` then `POST /dump_expert_distribution_record`
- Collect `.pt` files from all worker nodes
- Generate all plots: phase breakdown (with "other" category from `fwd_times`), per-rank breakdown, expert distribution heatmaps, token count CDFs/timelines, MoE compute time CDFs/timelines

### Recorder OFF runs (runs 2, 4, 6, 8)

- Launch server WITHOUT `--expert-distribution-recorder-mode`
- The recorder is a no-op — `capture_step()` returns immediately, overlap scheduler pipelines CPU/GPU work normally
- Run benchmark (same parameters)
- Record only: output throughput (tok/s), median ITL (ms), mean ITL (ms), P99 ITL (ms)

## Metrics to compare

For each profile, compare recorder-ON vs recorder-OFF:

- **Output throughput** (tok/s)
- **Median ITL** (ms)
- **Mean ITL** (ms)
- **P99 ITL** (ms)

If the throughput difference between ON and OFF is small, it means there are other sync points or bottlenecks beyond the recorder that we haven't identified.

## Optional PP*TP comparison

For comparing performance differences, we can also enable intra-node TP and cross-node PP, where number of pp stages is equal to number of nodes, and tp-size is equal to number of gpus on each node.
We run exactly the same benchmark client parameters as EP cases.
For PP*TP runs, we also test the 4 dataset profiles, but we don't need to enable recorder, we only need their basic performance metrics.
