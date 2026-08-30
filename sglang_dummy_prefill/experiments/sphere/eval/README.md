# SGLang Multi-Baseline Evaluation — Sphere

## sphere cluster Sphere-16 Cluster Setup & Launch Notes
Cluster: 8 nodes × 2 L40S GPUs = 16 GPUs total
Head node: sgpu0 (10.0.0.1)
Workers:   sgpu2 (10.0.0.2), sgpu3 (10.0.0.3), sgpu4 (10.0.0.4),
           sgpu6 (10.0.0.5), sgpu7 (10.0.0.6), sgpu8 (10.0.0.7),
           sgpu9 (10.0.0.8)
Network:   RoCE via ens1f1np1 (mlx5_1 HCA)
Conda env: disag12 (Python 3.12)
Note: there is NO NFS on sphere cluster, every node is fully bare-metal. You can ssh into every worker.

## File layout

```
eval/
  gptoss_eval.sh      # gpt-oss-120b: experiment matrix + orchestration loop
  glm4air_eval.sh     # GLM-4.5-Air:  experiment matrix + orchestration loop
  config.sh           # shared cluster / network / benchmark config
  config_gptoss.sh    # gpt-oss-120b model config (sources config.sh)
  config_glm4air.sh   # GLM-4.5-Air model config  (sources config.sh)
  evallib/
    cluster.sh        # discover_nodes(), kill_all() — SSH+tmux cluster management
    server.sh         # launch_server(), wait_for_server(), kill_server(), is_oom()
    benchmark.sh      # run_benchmark() — sglang.bench_serving wrapper
  README.md
```

`evallib/` scripts only define functions; they are sourced, not executed.
To change any single concern, edit only that one file.

### Config hierarchy

```
config.sh                  ← shared: cluster, network, benchmark, runtime
  ├── config_gptoss.sh     ← model: MODEL_PATH, GATE_PROFILES, CUSTOMIZED_ARGS
  └── config_glm4air.sh    ← model: MODEL_PATH, GATE_PROFILES, CUSTOMIZED_ARGS
```

Each eval script sources its model config (which internally sources the
shared config), then sources the function libraries.

---

## Models

| Eval script | Model config | Model | Params | Layers | Experts | HuggingFace path |
|---|---|---|---|---|---|---|
| `gptoss_eval.sh` | `config_gptoss.sh` | gpt-oss-120b-bf16 | 120B | 36 | 128 | `lmsys/gpt-oss-120b-bf16` |
| `glm4air_eval.sh` | `config_glm4air.sh` | GLM-4.5-Air | 106B (12B active) | 46 | 128 | `zai-org/GLM-4.5-Air` |

Both models use 128 routed experts and are evaluated with `--load-format dummy`.

---

## CUSTOMIZED_ARGS

Each model config defines a `CUSTOMIZED_ARGS` variable (string) that is
appended verbatim to every `sglang.launch_server` command. Use this to pass
model-specific flags without modifying the shared server profiles.

Examples:

```bash
# In config_glm4air.sh — disable shared-expert fusion for FP8 runs:
CUSTOMIZED_ARGS="--disable-shared-experts-fusion"

# In config_gptoss.sh — no extra args needed:
CUSTOMIZED_ARGS=""
```

---

## What each eval script does

Evaluates **3 server profiles × 4 gate profiles = 12 experiments**.
For each experiment, up to `MAX_RETRIES=3` times:

1. **`kill_server`** — kills any existing sglang processes + tmux sessions
   across all nodes. Required between runs to release GPUs.
2. **`launch_server`** — builds the full `sglang.launch_server` command based
   on the current `SERVER_PROFILE` (ep16, ep16_limited, or pp8tp2) with
   `--enable-fake-prefill` and `--profile-driven-gate-path`, plus any
   `CUSTOMIZED_ARGS`. Launches head (rank 0) + 7 workers via SSH+tmux.
   Saves commands to `server_cmd.sh`.
3. **`wait_for_server`** — polls `http://<head>:30000/health` every 10s,
   up to `SERVER_READY_TIMEOUT=1800s` (multi-node init can be slow).
   - If timeout, calls **`is_oom`** on the server logs. On OOM,
     `MEM_FRAC` is decreased by `MEM_FRAC_STEP=0.02` before the next attempt.
4. **`run_benchmark`** — runs `sglang.bench_serving` with fixed parameters
   matching the AsyncMoE evaluation; saves output to `bench_result.json`.
5. On success, copies the result to `result.json` and moves to the next experiment.

Final cleanup: `kill_server`.

---

## Server profiles

| Profile | Parallelism | Key flags |
|---|---|---|
| `ep16` | tp=16, dp=16, ep=16 (8 nodes) | `--enable-dp-attention --enable-dp-lm-head --moe-a2a-backend mooncake-nccl` |
| `ep16_limited` | tp=16, dp=16, ep=16 (8 nodes) | Same as ep16 + `--max-running-requests 256` |
| `ep8` | tp=8, dp=8, ep=8 (4 nodes) | Same flags as ep16 (DP-attention, DP-lm-head) but half the cluster |
| `pp8tp2` | tp=2, pp=8 (8 nodes) | `--disable-custom-all-reduce` (no EP/DP-attention) |

All profiles share: `--enable-fake-prefill`, `--profile-driven-gate-path`,
`--disable-radix-cache`, `--chunked-prefill-size -1`,
`--moe-runner-backend triton`, plus any `CUSTOMIZED_ARGS` from the model config.

---

## Shared config (edit `config.sh`)

| Parameter | Value |
|---|---|
| Cluster | 8 nodes × 2 L40S = 16 GPUs |
| Network | InfiniBand `ens1f1np1`, HCA `mlx5_1`, NCCL+Gloo |
| Initial memory fraction | 0.80 |
| OOM step | −0.02 per retry |
| Benchmark dataset | `.npy` files (aligned with AsyncMoE), auto-resolved per workload |
| Benchmark rate / prompts | 2000 rps × 10k reqs |
| Benchmark streaming | `BENCH_DISABLE_STREAM=0` (set to `1` to pass `--disable-stream`) |
| Server ready timeout | 1800s (30 min) |
| Benchmark timeout | 1200s (20 min) |
| Conda env | `sglang-fp` |

### Benchmark dataset

The benchmark always uses `.npy` files for sequence lengths (aligned with AsyncMoE).
The `.npy` path is auto-resolved per workload from `BENCH_DATASET_PATHS` in `config.sh`.

| Config variable | Default | Description |
|---|---|---|
| `BENCH_DATASET_PATHS` | `sharegpt_lengths.npy:sharegpt`, `gsm8k_lengths.npy:gsm8k` | `path:label_substring` mapping |
| `BENCH_NPY_CONTEXT_LEN` | 2048 | Filter: skip samples where input+output > this |

---

## Benchmark dataset: npy mode (default, aligned with AsyncMoE)

By default, `BENCH_DATASET=npy` is used. This loads input/output token lengths from `.npy`
files (shape `[N, 2]`, column 0 = input len, column 1 = output len) — the same format used
by AsyncMoE's `DatasetGenerator`. Since the server runs with `--enable-fake-prefill`, actual
token content is irrelevant; only the lengths matter.

The eval scripts **automatically resolve** which `.npy` file to use based on the gate profile
label: labels containing `sharegpt` use `datasets/sharegpt_lengths.npy`, labels containing
`gsm8k` use `datasets/gsm8k_lengths.npy`. This mapping is defined in `config.sh` via
`BENCH_DATASET_PATHS`.

To add a new dataset, place the `.npy` file in `datasets/` and add a corresponding entry to
`BENCH_DATASET_PATHS` in `config.sh`.

To add a new workload with different sequence lengths, place a `.npy` file in `datasets/`
and add an entry to `BENCH_DATASET_PATHS` in `config.sh`.

---

## Experiment matrix

Each eval script runs `SERVER_PROFILES × GATE_PROFILES` (4 × 4 = 16 experiments).

**4 server profiles**: `ep16`, `ep16_limited`, `pp8tp2`, `ep8`

**4 workloads** (`{sharegpt, gsm8k} × {regular, balanced}`):

### gpt-oss-120b gate profiles (`config_gptoss.sh`)

| Workload | Gate profile |
|---|---|
| `sharegpt_regular` | `gating_gptoss120b_sharegpt_200.parquet` |
| `sharegpt_balanced` | `balanced_gptoss120b_sharegpt_200.parquet` |
| `gsm8k_regular` | `gating_math_gsm8k_200.parquet` |
| `gsm8k_balanced` | `balanced_math_gsm8k_200.parquet` |

### GLM-4.5-Air gate profiles (`config_glm4air.sh`)

| Workload | Gate profile |
|---|---|
| `sharegpt_regular` | `gating_glm4air_sharegpt_200.parquet` |
| `sharegpt_balanced` | `balanced_glm4air_sharegpt_200.parquet` |
| `gsm8k_regular` | `gating_glm45air_gsm8k_200.parquet` |
| `gsm8k_balanced` | `balanced_glm45air_gsm8k_200.parquet` |

Regular profiles are captured from real inference traces.
Balanced profiles are pre-generated and placed in `gating_profiles/balanced_output/`.

---

## How to run

```bash
# 1. Ensure all 8 nodes are reachable via SSH
#    Default: head=local, workers=sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9

# 2. Run from repo root — gpt-oss-120b (all 12 experiments)
cd ~/sglang
bash experiments/sphere/eval/gptoss_eval.sh /path/to/gptoss_results \
    |& tee experiments/gptoss_eval.log

# 3. Run GLM-4.5-Air (all 12 experiments)
bash experiments/sphere/eval/glm4air_eval.sh /path/to/glm4air_results \
    |& tee experiments/glm4air_eval.log

# Or override node discovery:
HEAD=sgpu0 WORKERS="sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9" \
bash experiments/sphere/eval/gptoss_eval.sh /path/to/results

# Override npy context length filter:
BENCH_NPY_CONTEXT_LEN=4096 bash experiments/sphere/eval/gptoss_eval.sh /path/to/results

# Disable streaming in sglang.bench_serving:
bash experiments/sphere/eval/gptoss_eval.sh /path/to/results --disable-stream

# Or set the shared benchmark toggle directly:
BENCH_DISABLE_STREAM=1 bash experiments/sphere/eval/gptoss_eval.sh /path/to/results
```

### Running a single experiment

Use `--list` to see available experiments and `--only` to select which to run:

```bash
# List all 16 experiments (prints index + name, then exits)
bash experiments/sphere/eval/gptoss_eval.sh /path/to/results --list

# Run by index (1-based)
bash experiments/sphere/eval/gptoss_eval.sh /path/to/results --only 1

# Run multiple by index
bash experiments/sphere/eval/gptoss_eval.sh /path/to/results --only 1,5,9

# Run all experiments for a server profile
bash experiments/sphere/eval/gptoss_eval.sh /path/to/results --only pp8tp2

# Run all experiments for a workload across all server profiles
bash experiments/sphere/eval/gptoss_eval.sh /path/to/results --only sharegpt_regular

# Run one exact experiment (server profile + workload)
bash experiments/sphere/eval/gptoss_eval.sh /path/to/results --only ep16-sharegpt_regular

# Run a single experiment without streaming
bash experiments/sphere/eval/gptoss_eval.sh /path/to/results --only ep16-sharegpt_regular --disable-stream
```

The `--only` filter accepts comma-separated values. Each value is matched as
a 1-based index (if numeric) or as a substring of the run name (e.g.
`sglang_ep16-sharegpt_regular`). Omitting `--only` runs all 16 experiments.

`--disable-stream` propagates to `sglang.bench_serving --disable-stream`, so the
benchmark records aggregate request latency while the server-side detokenizer log
remains the source of ITL and in-flight metrics.

**Note:** substring `ep16` matches both `ep16` and `ep16_limited` run names.
Use `_ep16-` to match only the base ep16 profile.

---

## Output layout

Run directories are named `<system>_<server_profile>-<dataset_label>` under `RESULTS_DIR`.

```
<RESULTS_DIR>/
  sglang_ep16-sharegpt_regular/
    server_cmd.sh                        # exact server launch commands (replayable)
    logs/
      server_head.log                    # head node stdout/stderr
      server_w1.log                      # worker 1 log
      server_w2.log … server_w7.log      # workers 2–7
    bench_cmd.sh                         # exact benchmark command (replayable)
    bench_result.json                    # raw benchmark output
    bench_result.log                     # benchmark stdout
    result.json                          # copy of the successful result
  sglang_ep16-sharegpt_balanced/         ...
  sglang_ep16-gsm8k_regular/            ...
  sglang_ep16-gsm8k_balanced/           ...
  ...  (16 directories total)
```

Only the final successful attempt's logs are kept; previous attempts are
cleaned up before each retry.

To plot global batch size (in-flight/waiting) timelines: `python experiments/sphere/eval/plot_sglang_inflight_timeline.py <RESULTS_DIR> [profile_filter]`

---

## Adding a new model

1. Create `config_<model>.sh` — source `config.sh`, set `MODEL_PATH`,
   `MODEL_NAME`, `LOAD_FORMAT`, `GATE_PROFILES`, and `CUSTOMIZED_ARGS`.
2. Copy any eval script (e.g. `gptoss_eval.sh`), change the source line
   to `config_<model>.sh`, and update the header comment.
3. Generate gate profiles for the new model and place them in `gating_profiles/`.
