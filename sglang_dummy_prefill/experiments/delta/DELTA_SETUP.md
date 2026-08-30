# sglang fake_prefill_coul — Delta 4×A100 Setup

## Hardware

- Node: `gpua080` on NCSA Delta (job `16932438`, partition `gpuA100x4`)
- 4× NVIDIA A100-SXM4-40GB, CUDA driver 12.8
- Home dir shared across login + compute nodes

## Environment Setup

```bash
# 1. SSH to the GPU node
ssh delta
ssh gpua080

# 2. Clone the branch
cd ~
git clone -b fake_prefill_coul https://github.com/shawlleyw/sglang.git

# 3. Create conda env
~/miniconda3/bin/conda create -n sglang python=3.11 -y

# 4. Activate and install PyTorch with CUDA 12.6
source ~/miniconda3/bin/activate sglang
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 5. Install sglang (editable)
cd ~/sglang/python
pip install -e .
```

## Model Preparation

The full `lmsys/gpt-oss-120b-bf16` (36 layers, 128 experts) doesn't fit on 4×40GB.
We download only config + tokenizer and cut layers to 1/4 (9 layers) since we use `--load-format dummy`.

```bash
python ~/sglang/scripts/prepare_model.py
```

This creates `~/gpt-oss-120b-bf16-mini/` with a modified `config.json` (9 layers instead of 36).

## Launch Server

```bash
~/sglang/scripts/launch_server.sh ~/sglang/logs/test.log
```

The script runs:
```
python -m sglang.launch_server \
    --model-path ~/gpt-oss-120b-bf16-mini \
    --load-format dummy \
    --tp-size 4 --dp-size 4 --ep-size 4 \
    --enable-dp-attention --enable-dp-lm-head \
    --moe-a2a-backend mooncake-nccl \
    --nnodes 1 \
    --enable-fake-prefill \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --mem-fraction-static 0.80 \
    --max-running-requests 256 \
    --cuda-graph-max-bs 16 \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --moe-runner-backend triton \
    --dist-timeout 1800 \
    --log-level-http warning --log-level warning
```

Startup takes ~3 minutes (TVM/inductor compilation + CUDA graph capture).
Look for `The server is fired up and ready to roll!` in the log.

## Known Issues

- **Custom all-reduce fails during CUDA graph capture** on this DP/EP topology.
  Fix: `--disable-custom-all-reduce` (falls back to NCCL, keeps CUDA graphs).
- **MoE triton kernel config missing** — benign warning, uses default config.
  Can be tuned via `benchmark/kernels/fused_moe_triton` if needed.
