# CarcAI Node Rules for Agents

Rules and constraints for running SGLang on the CarcAI HPC node. **Read this before doing anything.**

---

## 1. Filesystem Layout

| Path | Type | Capacity | Use for |
|------|------|----------|---------|
| `/home1/yizhuoli/` | NFS home | **Very limited quota** | Code repos, small configs only |
| `/scratch1/yizhuoli/` | BeeGFS scratch | ~1.3 PB shared | Everything heavy: conda envs, model weights, HF cache, logs, benchmark results, temp files |

### Hard Rules

- **NEVER** write large files (models, datasets, caches, logs, conda envs) under `~/` or `/home1/`. You will blow the quota.
- **NEVER** use `/tmp` for anything large. It's a small local tmpfs. Set `TMPDIR=/scratch1/yizhuoli/tmp` instead.
- **ALWAYS** use `/scratch1/yizhuoli/` for:
  - Conda environments → `/scratch1/yizhuoli/conda-envs/`
  - HuggingFace cache → `/scratch1/yizhuoli/hf_cache`
  - Pip cache → `/scratch1/yizhuoli/pip_cache`
  - Torch hub cache → `/scratch1/yizhuoli/torch_cache`
  - Benchmark results → `/scratch1/yizhuoli/bench_results/`
  - Any temp/intermediate data → `/scratch1/yizhuoli/tmp/`

### What Lives Where Currently

```
/home1/yizhuoli/
├── sglang-fake-prefill/    # Main repo (code only, ~reasonable size)
│   ├── gating_profiles/    # Small parquet files (~MBs), OK in repo
│   ├── coulson_docs/       # Docs and scripts
│   └── investigate_profiles/ # Analysis scripts and plots
└── miniconda3/             # Conda base installation

/scratch1/yizhuoli/
├── conda-envs/sglang-fp/  # The SGLang conda environment (~GBs)
├── hf_cache/               # HuggingFace model downloads (~100s of GBs)
├── bench_results/          # Benchmark output JSONLs and logs
├── moe_instrument_logs/    # MoE instrumentation JSONL logs
├── tmp/                    # TMPDIR target
├── pip_cache/
└── torch_cache/
```

---

## 2. Hardware

- **GPUs**: 4× NVIDIA A100-SXM4-80GB (SM80, NVLink interconnect)
- **OS**: Rocky Linux 8.10
- **No sudo access** — cannot `apt install`, `yum install`, or modify system packages

---

## 3. Environment Setup

**Copy-paste this block at the start of any script or shell session:**

```bash
eval "$(conda shell.bash hook)"
conda activate /scratch1/yizhuoli/conda-envs/sglang-fp
module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda
export HF_HOME=/scratch1/yizhuoli/hf_cache
export TMPDIR=/scratch1/yizhuoli/tmp
export PYTHONPATH=/home1/yizhuoli/sglang-fake-prefill/python:$PYTHONPATH
```

### What Each Line Does

| Line | Why |
|------|-----|
| `conda activate .../sglang-fp` | Python 3.12 env with SGLang + all deps (PyTorch, Triton, DeepEP, etc.) |
| `module load cuda/12.6.3` | CUDA toolkit (nvcc, libraries). Required for Triton kernels and DeepEP |
| `module load ucx/1.16.0` | UCX transport for NCCL (GPU-to-GPU comms over NVLink) |
| `module load gdrcopy/2.5.1-cuda` | GPUDirect RDMA copy (used by DeepEP for efficient all-to-all) |
| `HF_HOME=...` | Redirect HuggingFace downloads to scratch (default goes to `~/.cache/`) |
| `TMPDIR=...` | Redirect temp files to scratch (default `/tmp` is too small) |
| `PYTHONPATH=...` | Use the local SGLang source tree (editable install from repo) |

### If You Need New Python Packages

```bash
pip install --cache-dir /scratch1/yizhuoli/pip_cache <package>
```

Do **not** create new conda envs unless specifically asked. Use the existing `sglang-fp` env.

---

## 4. Running SGLang Server

### Typical EP4 Launch (GPT-OSS with DeepEP)

```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-120b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --mem-fraction-static 0.85 \
  --enable-fake-prefill \
  --profile-driven-gate-path ./gating_profiles/<profile>.parquet \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --load-format dummy
```

### Key Parameters

| Parameter | Notes |
|-----------|-------|
| `--mem-fraction-static 0.85` | For EP4. DeepEP needs ~10-15% GPU memory for communication buffers. Using 0.95 OOMs. |
| `--mem-fraction-static 0.85` | For TP4 as well (safe default). Can try 0.92 if more KV cache needed. |
| `--load-format dummy` | Skip real weight download. Uses random/zero weights. Good for pipeline testing. |
| `--enable-fake-prefill` | Skip real prefill (decode-only benchmarking). Output will be garbled. |
| `--deepep-mode normal` | High-throughput NVLink mode. Auto-disables CUDA graphs. |
| `--port 30005` | Our standard port. Check nothing else is using it first. |

### Server Startup Time

- With `--load-format dummy`: ~2-3 minutes
- With real weights (120b): ~8-12 minutes (downloading + loading + warmup)

### Killing the Server

SGLang spawns multiple processes (scheduler, detokenizer, workers). Kill them all:

```bash
pkill -9 -f "sglang.launch_server"
pkill -9 -f "sglang::scheduler"
pkill -9 -f "sglang::detokenizer"
sleep 5
```

Always kill and wait before starting a new server — leftover processes hold GPU memory.

---

## 5. Running Benchmarks

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 --port 30005 \
  --model openai/gpt-oss-120b \
  --dataset-name random \
  --random-input-len 256 --random-output-len 1024 \
  --random-range-ratio 0.5 \
  --num-prompts 2000 \
  --request-rate 500 \
  --output-file /scratch1/yizhuoli/bench_results/result_<name>.jsonl
```

Always write output files to `/scratch1/`, not `~/`.

---

## 6. Available Gating Profiles

```
gating_profiles/gating_gptoss_sharegptv3_200.parquet  # General conversation routing
gating_profiles/gating_math_gsm8k_200.parquet         # Math-focused routing
```

Used with `--profile-driven-gate-path`. Requires `--disable-radix-cache --chunked-prefill-size -1`.

---

## 7. Common Pitfalls

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Writing to `~/.cache/huggingface/` | Fills home quota | Set `HF_HOME=/scratch1/yizhuoli/hf_cache` |
| Using `/tmp` for large files | Fills tmpfs, kills node | Set `TMPDIR=/scratch1/yizhuoli/tmp` |
| `--mem-fraction-static 0.95` with EP4 | OOM from DeepEP buffer allocation | Use 0.85 |
| Forgetting `module load` | NCCL/CUDA failures, missing libcuda | Always load all three modules |
| Not killing old server before new one | Port conflict or GPU memory exhaustion | `pkill -9 -f sglang` + sleep |
| Creating conda envs under `~/` | Fills home quota fast | Always use `/scratch1/yizhuoli/conda-envs/` |
| `pip install` without `--cache-dir` | Pip cache fills home quota | Use `--cache-dir /scratch1/yizhuoli/pip_cache` |
| Running `conda create` without `-p` | Creates env in `~/miniconda3/envs/` | Use `conda create -p /scratch1/yizhuoli/conda-envs/<name>` |

---

## 8. Quick Reference

```bash
# Full setup one-liner
eval "$(conda shell.bash hook)" && conda activate /scratch1/yizhuoli/conda-envs/sglang-fp && module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda && export HF_HOME=/scratch1/yizhuoli/hf_cache TMPDIR=/scratch1/yizhuoli/tmp PYTHONPATH=/home1/yizhuoli/sglang-fake-prefill/python:$PYTHONPATH

# Check GPU status
nvidia-smi

# Check if server is running
curl -s http://localhost:30005/health

# Kill all SGLang processes
pkill -9 -f sglang; sleep 5

# Repo root
cd /home1/yizhuoli/sglang-fake-prefill
```
