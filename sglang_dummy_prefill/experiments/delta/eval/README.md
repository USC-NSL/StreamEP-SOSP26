# SGLang Multi-Baseline Evaluation — NCSA Delta

## File layout

```
eval/
  gptoss_eval.sh      # gpt-oss-120b: experiment matrix + orchestration loop
  config.sh           # shared cluster / network / benchmark config
  config_gptoss.sh    # gpt-oss-120b model config (sources config.sh)
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
  └── config_gptoss.sh     ← model: MODEL_PATH, GATE_PROFILES, CUSTOMIZED_ARGS
```

Each eval script sources its model config (which internally sources the
shared config), then sources the function libraries.

---

## Models

| Eval script | Model config | Model | Params | Layers | Experts | HuggingFace path |
|---|---|---|---|---|---|---|
| `gptoss_eval.sh` | `config_gptoss.sh` | gpt-oss-120b-bf16 | 120B | 36 | 128 | `lmsys/gpt-oss-120b-bf16` |

The model uses 128 routed experts and is evaluated with `--load-format dummy`.

---

## CUSTOMIZED_ARGS

Each model config defines a `CUSTOMIZED_ARGS` variable (string) that is
appended verbatim to every `sglang.launch_server` command for EP profiles.
Use this to pass model-specific flags without modifying the shared server profiles.

Example:

```bash
# In config_gptoss.sh — mooncake-nccl a2a backend:
CUSTOMIZED_ARGS="--moe-a2a-backend mooncake-nccl"
```

---

## What each eval script does

Evaluates **4 server profiles × 4 gate profiles = 16 experiments**.
For each experiment, up to `MAX_RETRIES=3` times:

1. **`kill_server`** — kills any existing sglang processes + tmux sessions
   across all nodes. Required between runs to release GPUs.
2. **`launch_server`** — builds the full `sglang.launch_server` command based
   on the current `SERVER_PROFILE` (ep16, ep16_limited, pp4tp4, or ep8) with
   `--enable-fake-prefill` and `--profile-driven-gate-path`, plus any
   `CUSTOMIZED_ARGS`. Launches head (rank 0) + workers via SSH+tmux.
   Saves commands to `server_cmd.sh`.
3. **`wait_for_server`** — polls `http://<head>:30000/health` every 10s,
   up to `SERVER_READY_TIMEOUT=1800s` (multi-node init on Delta can be slow).
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
| `ep16` | tp=16, dp=16, ep=16 (4 nodes) | `--enable-dp-attention --enable-dp-lm-head --disable-custom-all-reduce` + `CUSTOMIZED_ARGS` |
| `ep16_limited` | tp=16, dp=16, ep=16 (4 nodes) | Same as ep16 + `--max-running-requests 256` |
| `ep8` | tp=8, dp=8, ep=8 (2 nodes) | Same flags as ep16 but 2 nodes only |
| `pp4tp4` | tp=4, pp=4 (4 nodes) | `--disable-custom-all-reduce` (no EP/DP-attention) |

All profiles share: `--enable-fake-prefill`, `--profile-driven-gate-path`,
`--disable-radix-cache`, `--chunked-prefill-size -1`,
`--moe-runner-backend triton`, `--disable-custom-all-reduce`.

EP profiles additionally append `CUSTOMIZED_ARGS` from the model config
(e.g. `--moe-a2a-backend mooncake-nccl` for gpt-oss).

**`--disable-custom-all-reduce` is required on ALL profiles on Delta.** Without it,
CUDA graph capture hangs indefinitely on DP/EP topologies (falls back to Gloo and deadlocks).

---

## Shared config (edit `config.sh`)

| Parameter | Value |
|---|---|
| Cluster | 4 nodes × 4 A100-SXM4-40GB = 16 GPUs |
| Network | HPE Slingshot `hsn0`, NCCL+Gloo |
| Initial memory fraction | 0.65 (ep16/pp4tp4), 0.85 (ep8) — see Delta notes |
| OOM step | −0.02 per retry |
| Benchmark dataset | `sharegpt` (default), `random`, `gsm8k` |
| Benchmark rate / prompts | 2000 rps × 10k reqs |
| Server ready timeout | 300s (5 min) |
| Benchmark timeout | ep16: 600s, ep16_limited: 1200s, ep8: 900s, pp4tp4: 1500s |
| Conda env | `sglang` |

### Benchmark datasets

Each experiment auto-selects its benchmark dataset from the gate profile label
(`sharegpt*` → `sharegpt`, `gsm8k*` → `gsm8k`). Override for all experiments
in a run by setting `BENCH_DATASET` explicitly. All dataset parameters are
overridable via environment:

| Dataset | Config variables (env override) | Defaults |
|---|---|---|
| `sharegpt` | `BENCH_SHAREGPT_CONTEXT_LEN`, `BENCH_SHAREGPT_OUTPUT_LEN` (optional) | context_len=2048, natural output length |
| `gsm8k` | `BENCH_GSM8K_CONTEXT_LEN`, `BENCH_GSM8K_OUTPUT_LEN` (optional) | context_len=2048, natural answer length |
| `random` | `BENCH_RANDOM_INPUT_LEN`, `BENCH_RANDOM_OUTPUT_LEN`, `BENCH_RANDOM_RANGE_RATIO` | 512 in/out, 0.5 range ratio |

---

## Experiment matrix

Each eval script runs `SERVER_PROFILES × GATE_PROFILES` (4 × 4 = 16 experiments).

**4 server profiles**: `ep16`, `ep16_limited`, `pp4tp4`, `ep8`

**4 workloads** (`{sharegpt, gsm8k} × {regular, balanced}`):

### gpt-oss-120b gate profiles (`config_gptoss.sh`)

| Workload | Gate profile |
|---|---|
| `sharegpt_regular` | `gating_gptoss120b_sharegpt_200.parquet` |
| `sharegpt_balanced` | `balanced_gptoss120b_sharegpt_200.parquet` |
| `gsm8k_regular` | `gating_math_gsm8k_200.parquet` |
| `gsm8k_balanced` | `balanced_math_gsm8k_200.parquet` |

Regular profiles are captured from real inference traces.
Balanced profiles live in `gating_profiles/gptosss_balanced_output/` (note triple-s).

---

## How to run

```bash
# 1. Ensure you have a SLURM allocation (4 nodes × 4 A100)
squeue -u $USER

# 2. Run from login node (uses SSH+tmux to reach compute nodes)
cd ~/sglang
bash experiments/delta/eval/gptoss_eval.sh /path/to/gptoss_results \
    |& tee experiments/gptoss_eval.log

# Or override node discovery:
HEAD=gpua002 WORKERS="gpua007 gpua047 gpua076" \
    bash experiments/delta/eval/gptoss_eval.sh /path/to/results

# Use a different benchmark dataset:
BENCH_DATASET=gsm8k bash experiments/delta/eval/gptoss_eval.sh /path/to/results
BENCH_DATASET=random bash experiments/delta/eval/gptoss_eval.sh /path/to/results

# Override context length for sharegpt or gsm8k:
BENCH_SHAREGPT_CONTEXT_LEN=4096 bash experiments/delta/eval/gptoss_eval.sh /path/to/results
BENCH_DATASET=gsm8k BENCH_GSM8K_CONTEXT_LEN=4096 \
    bash experiments/delta/eval/gptoss_eval.sh /path/to/results
```

### Running a single experiment

Use `--list` to see available experiments and `--only` to select which to run:

```bash
# List all 16 experiments (prints index + name, then exits)
bash experiments/delta/eval/gptoss_eval.sh /path/to/results --list

# Run by index (1-based)
bash experiments/delta/eval/gptoss_eval.sh /path/to/results --only 1

# Run multiple by index
bash experiments/delta/eval/gptoss_eval.sh /path/to/results --only 1,5,9

# Run all experiments for a server profile
bash experiments/delta/eval/gptoss_eval.sh /path/to/results --only pp4tp4

# Run all experiments for a workload across all server profiles
bash experiments/delta/eval/gptoss_eval.sh /path/to/results --only sharegpt_regular

# Run one exact experiment (server profile + workload)
bash experiments/delta/eval/gptoss_eval.sh /path/to/results --only ep16-sharegpt_regular
```

The `--only` filter accepts comma-separated values. Each value is matched as
a 1-based index (if numeric) or as a substring of the run name (e.g.
`sglang_ep16-sharegpt_regular`). Omitting `--only` runs all 16 experiments.

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
      server_w2.log                      # worker 2 log
      server_w3.log                      # worker 3 log
    bench_cmd.sh                         # exact benchmark command (replayable)
    bench_result.json                    # raw benchmark output
    bench_result.log                     # benchmark stdout
    result.json                          # copy of the successful result
  sglang_ep16-sharegpt_balanced/         ...
  sglang_ep16-gsm8k_regular/            ...
  sglang_ep16-gsm8k_balanced/           ...
  ...  (16 directories total)
```

Failed attempt artifacts are archived to `attempt<N>/` subdirectories before
each retry, preserving logs for post-mortem debugging.

---

## Monitoring

Always check **both** the main eval log and the server head log — critical server
errors (OOM, NCCL failures) only appear in `server_head.log` and are not surfaced
in the main `[main]`/`[server]` output.

```bash
# Main eval progress
tail -f ~/unified-eval-mar28-1/sglang-gptoss.log | grep '\[main\]\|\[server\]'

# Server health (most important — check this on every poll)
tail -20 <RESULTS_DIR>/<experiment>/logs/server_head.log

# Worker logs (check if a worker dropped out)
tail -5 <RESULTS_DIR>/<experiment>/logs/server_w1.log
tail -5 <RESULTS_DIR>/<experiment>/logs/server_w2.log
tail -5 <RESULTS_DIR>/<experiment>/logs/server_w3.log

# GPU utilization — if any node shows 0%, a worker crashed
for node in gpua003 gpua017 gpua072 gpua080; do
    printf "$node: "
    ssh $node 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr "\n" " "'
    echo
done
```

Healthy server log shows:
- `Capturing batches (bs=N avail_mem=X GB): N%|...` — CUDA graph capture in progress
- `The server is fired up and ready to roll!` — ready
- `Throughput: N tokens/s, In-flight requests: N` — benchmark running

Unhealthy signs:
- `RuntimeError: Not enough memory. Please try to increase --mem-fraction-static` → OOM, increase MEM_FRAC
- `CUDA graph capture` frozen >5 min with all GPUs at 100% → deadlock (check `--disable-custom-all-reduce`)
- Any node at 0% GPU after server ready → worker crashed, check that node's log

---

## Delta-specific notes

### `--disable-custom-all-reduce` is required on all profiles

Without it, CUDA graph capture on DP/EP topologies hangs indefinitely on Delta —
falls back to Gloo and deadlocks. All four server profiles now include this flag.

### MEM_FRAC is profile-dependent

| Profile | Required MEM_FRAC | Reason |
|---|---|---|
| `ep16`, `ep16_limited`, `pp4tp4` | 0.65 (confirmed working) | 16-GPU sharding leaves adequate headroom |
| `ep8` | ≥ 0.85 | 8-GPU sharding → ~30 GB weights/GPU for 120B model; 0.75 is not enough |

Setting MEM_FRAC too high on ep16 (≥ 0.70 with `--moe-a2a-backend mooncake-nccl`)
causes CUDA graph capture to deadlock. `CUDA_LAUNCH_BLOCKING=1` makes this worse
(full deadlock) because mooncake-nccl relies on async communication during capture.

### Use the HuggingFace model path

Set `MODEL_PATH="lmsys/gpt-oss-120b-bf16"` (HF model ID) rather than a local
directory. Local copies may be stale or have modified configs. Works correctly
with `--load-format dummy` which downloads only config + tokenizer (~MBs).

### Killing a stuck eval

`kill $PID` only kills the nohup wrapper — sglang children survive. Always clean up:

```bash
# Kill eval script
kill $(cat <pid_file>)

# Kill all sglang processes on all nodes
for node in gpua003 gpua017 gpua072 gpua080; do
    ssh $node 'pkill -9 -f "sglang" 2>/dev/null' || true
done

# Kill tmux sessions
for s in sglang-head sglang-w1 sglang-w2 sglang-w3; do
    tmux kill-session -t "$s" 2>/dev/null || true
done
```

---

## Adding a new model

1. Create `config_<model>.sh` — source `config.sh`, set `MODEL_PATH`,
   `MODEL_NAME`, `LOAD_FORMAT`, `GATE_PROFILES`, and `CUSTOMIZED_ARGS`.
2. Copy any eval script (e.g. `gptoss_eval.sh`), change the source line
   to `config_<model>.sh`, and update the header comment.
3. Generate gate profiles for the new model and place them in `gating_profiles/`.
