# sglang_dummy_prefill

A fork of SGLang with **fake prefill** (decode from a dummy KV-cache) — the
baseline system for the StreamInfer artifact. It benchmarks MoE decoding without
running real prefill or loading real weights, so it is directly comparable to
StreamInfer's dummy-weight decoding.

> This is SGLang **0.5.5** and requires **torch 2.8** (cu128). That conflicts
> with the StreamInfer env (torch 2.6 / cu124), so install this in its **own**
> conda env. The two systems are always run one at a time, never together.
>
> (The original upstream SGLang README is kept as
> [`README_upstream.md`](README_upstream.md).)

## Dependencies

| Dependency | Version | Notes |
|---|---|---|
| CUDA runtime | 12.x (cu128) | provided by the torch 2.8 wheel |
| Python | 3.12 | |
| torch / torchvision / torchaudio | 2.8.0 | pulled in by the editable install |
| flashinfer_python / flashinfer_cubin | 0.5.0 | pulled in automatically |
| sgl-kernel | 0.3.16.post5 | pulled in automatically |
| transformers | 4.57.1 | pulled in automatically |

## Install

### 1. Dedicated conda env

```bash
conda create -n sglang python=3.12 -y
conda activate sglang
pip install --upgrade pip
```

### 2. Editable install (pulls torch 2.8 + kernels)

```bash
cd sglang_dummy_prefill      # this repo
pip install -e .
```

This downloads torch 2.8, `sgl-kernel`, and `flashinfer` (several GB) and installs
`sglang` in editable mode.

### 3. Verify

```bash
python -c "import sglang; print('sglang', sglang.__version__)"
sglang --help >/dev/null && echo "CLI OK"
```

## Running gpt-oss (dummy weights, fake prefill, reduced layers)

The gpt-oss benchmark runs on **dummy weights**: only `config.json` + the
tokenizer are fetched from `lmsys/gpt-oss-120b-bf16` (no 120B checkpoint). Enable
fake prefill and cut the layer count to **1/4** (36 → 9) at launch — the same
reduction the StreamInfer side uses for the 1/4-size cluster:

```bash
python -m sglang.launch_server \
    --model-path lmsys/gpt-oss-120b-bf16 \
    --load-format dummy \
    --enable-fake-prefill \
    --num-hidden-layers-override 9 \
    --trust-remote-code \
    --host <HEAD_IP> --port 30000 \
    # + parallelism flags for your cluster (e.g. --tp-size / --dp-size / --ep-size / --pp-size)
```

Key flags:

| Flag | Effect |
|---|---|
| `--load-format dummy` | random in-memory weights; no checkpoint download |
| `--enable-fake-prefill` | decode from a dummy KV-cache (skips real prefill) |
| `--num-hidden-layers-override 9` | use 9 of the 36 layers (fits the smaller cluster) |

Benchmark the server with `python -m sglang.bench_serving` (`--dataset-name npy
--dataset-path <lengths.npy> --request-rate <rps> --num-prompts <n>`). See
[`experiments/sphere/eval/`](experiments/sphere/eval/) for the reference
multi-node launch + benchmark scripts (server profiles and EP/PP/TP sizes).
