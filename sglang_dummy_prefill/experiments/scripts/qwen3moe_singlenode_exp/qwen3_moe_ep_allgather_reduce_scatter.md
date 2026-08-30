# Qwen3 MoE with Expert Parallelism: AllGather + ReduceScatter

## Environment

| | |
|---|---|
| Model | Qwen3 MoE (e.g. Qwen3-30B-A3B, Qwen3-235B-A22B) |
| GPUs | N× GPU |
| SGLang | 0.5.5+ |
| Parallelism | TP=N, DP=N, EP=N |

## Bug Fix (required before running)

`SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL` is referenced in `expert_distribution.py`
but missing from `environ.py`, causing a crash on the first decode step. Add it:

```python
# python/sglang/srt/environ.py — inside class Envs, EPLB section
SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)  # disabled if <= 0
```

Also remove `--enable-expert-distribution-metrics` from the launch command if present.

## Step 1 — Launch the Server

```bash
export NUM_GPUS=<N>
export SGLANG_ATTN_MAX_BS=<MAX_BS_PER_RANK>
export SGLANG_MAX_RUNNING_REQUESTS=$(( SGLANG_ATTN_MAX_BS * NUM_GPUS ))
export SGLANG_TORCH_PROFILER_DIR=<PROFILE_OUTPUT_DIR>

python -m sglang.launch_server \
    --model <MODEL_PATH> \
    --trust-remote-code \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --enable-fake-prefill \ 
    --mem-fraction-static 0.8 \
    --tp-size $NUM_GPUS --dp-size $NUM_GPUS --ep-size $NUM_GPUS \
    --enable-dp-attention \
    --enable-dp-lm-head \
    --max-running-requests $SGLANG_MAX_RUNNING_REQUESTS \
    --disable-cuda-graph \
    --log-level-http warning --log-level warning
```

Key flags:

| Flag | Purpose |
|---|---|
| `--tp-size N --dp-size N --ep-size N` | All N GPUs act as TP, DP, and EP simultaneously |
| `--enable-dp-attention` | Each rank attends only its own token subset; enables AllGather/ReduceScatter around MoE |
| `--enable-dp-lm-head` | Parallelize the LM head across DP ranks |
| `--disable-cuda-graph` | Required when CUDA graphs are not compiled |
| `--max-running-requests` | `SGLANG_ATTN_MAX_BS × DP` — sets peak decode batch size |
| `--disable-radix-cache` | Avoids prefix cache interference in benchmarks |

Wait for: `The server is fired up and ready to roll!`

## Step 2 — Benchmark (no profiling)

```bash
python -m sglang.bench_serving \
    --num-prompts 1000 \
    --sharegpt-output-len 500
```

## Step 3 — Capture a Profile (100 decode steps at peak batch)

```bash
python -m sglang.bench_serving \
    --num-prompts 1000 \
    --sharegpt-output-len 500 \
    --profile \
    --profile-start-min-batch-size <MIN_BATCH_SIZE> \
    --profile-num-steps 100
```

Trace files are written to `SGLANG_TORCH_PROFILER_DIR`, one `.trace.json.gz` per GPU rank:

```
<timestamp>-TP-0-DP-0-EP-0.trace.json.gz
<timestamp>-TP-1-DP-1-EP-1.trace.json.gz
...
```

## Teardown

```bash
pkill -f "sglang.launch_server"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```
