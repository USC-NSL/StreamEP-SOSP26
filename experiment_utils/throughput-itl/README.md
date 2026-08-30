# Throughput vs. ITL — StreamInfer vs. sglang EP

Measures how **token throughput** and **inter-token latency (ITL)** trade off as
the request rate rises, for **both** systems on the same gpt-oss MoE workload, so
their latency–throughput curves can be compared:

- **StreamInfer** — the system under evaluation (barrier-free distributed MoE).
- **sglang EP** (`sglang_dummy_prefill`) — the baseline: expert-parallel sglang
  with fake prefill (matches the paper's `ep16` profile).

Both run gpt-oss-120b cut to **18 layers** (half of the model's 36) on
**dummy weights** (no checkpoints) on the **4-node, 8-GPU** testbed, sharegpt request
lengths, at rates **25 / 50 / 100 / 150 / 200 rps**. The output is a single
**latency-vs-throughput** plot.

---

## Quick reproduction on the provided testbed (SPHERE)

On SPHERE everything is **already prepared** — 4 nodes (**sgpu6/7/8/9**,
8× L40S), both conda envs (`streaminfer`, `sglang`) and `gdrdrv` installed on every
node, and [`config.sh`](config.sh) already points at the cluster (`N_NODE=4`,
`WORLD_SIZE=8`, `NUM_LAYERS=18`,
`WORKER_HOSTS="sgpu7 sgpu8 sgpu9"`, RoCE device `mlx5_1`).
**You only launch the two runs — one at a time — and read the plot.** Run everything
from the artifact root on the head (**sgpu6**).

### 1 · sglang baseline — head only (it SSH-launches its own workers)

```bash
bash experiment_utils/throughput-itl/run_head_sglang.sh
```

> **One-time** (sglang's benchmark client opens many
> connections and exhausts the head's ephemeral ports): widen the range on head, sgpu6 —
> ```bash
> sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535" net.ipv4.tcp_tw_reuse=1
> ```

### 2 · StreamInfer — head only (it SSH-launches its own workers)

```bash
# on the head node — sgpu6 (workers are SSH-launched and stopped automatically)
cd ~/StreamEP-Artifact
bash experiment_utils/throughput-itl/run_head_streaminfer.sh
```

The head starts Ray, SSH-launches `run_worker_streaminfer.sh` on every worker,
waits until all 8 GPUs have joined, sweeps the rates, records results, parses,
and stops the workers.

### 3 · Read the result

Both head runs re-parse at the end, so once **both** systems have run you have:

- **`results/comparison.png`** — inter-token latency (ms) vs. output throughput
  (tok/s), one labeled point per rate, both systems.
- `results/comparison.csv` and `results/{streaminfer,sglang}/…/result.json` — raw numbers,
  including sglang's request success rate (`SG ok` — completed/prompted; under overload
  sglang sheds a few % of requests with connection resets, reported rather than hidden).

You should see **StreamInfer** climb smoothly rightward, while
**sglang** peaks and then **hooks backward** — throughput
regresses and latency spikes — as its per-step DP-attention barrier collapses
under overload.

> Run the two systems **one at a time** — each uses all 8 GPUs. The comparison plot is
> drawn by the parse step under each run's own conda env, which needs `matplotlib`
> (both envs have it on SPHERE). If a run prints `plot skipped: No module named …`, or
> to (re)draw the plot any time after both runs, run the parse under any env that has
> matplotlib:

---

## Outputs

```
results/
├── streaminfer/sharegpt-<rate>rps/result.json   # /run_once response (embeds metrics)
├── sglang/sharegpt-<rate>rps/result.json         # sglang.bench_serving output JSON
├── comparison.csv                                # one row per rate, both systems
└── comparison.png                                # latency vs throughput, both systems
```

---

## Detailed configuration & general (any-cluster) reproduction

These scripts are tuned for **SPHERE**. On other hardware/networks treat them as a
**reference**, not a turnkey tool — the parallelism degree, layer count and NIC
names will differ, and the DP-attention path is fragile enough that you may hit one
of the issues below. Every value is environment-overridable (or edit `config.sh`).

### Knobs

Shared knobs are in [`config.sh`](config.sh); system-specific ones are at the top
of each `run_head_<sys>.sh`.

| Knob | Meaning |
|---|---|
| `N_NODE`, `N_GPU_PER_NODE` | → `WORLD_SIZE` = total GPUs = the EP/DP degree |
| `WORKER_HOSTS` | worker hostnames → ranks 1..N (head = rank 0) |
| `HEAD_IP`, `HOST_IFNAME` | head's cluster IP + NIC for NCCL sockets / rendezvous |
| `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX` | RoCE/IB device(s) + GID index. A **comma-list** lets nodes that name the NIC differently each pick their own (SPHERE: all nodes use `mlx5_1`) |
| `NUM_LAYERS` | gpt-oss number of layers (original model is 36; the paper uses 16 GPUs to run the full model — reduce layers proportionally on a scaled-down cluster, here 18 on 8 GPUs) |
| `RATES`, `BENCH_TIME` | request rates + seconds of load per rate |
| `MEM_FRAC` / `MEM_FRAC_HIGH` / `HIGH_RATE_THRESHOLD` (sglang) | KV-cache mem fraction; the runner uses the lower `MEM_FRAC_HIGH` above the threshold so activation memory fits at high rates |
| `GLOO_IFNAME` (sglang) | dedicated NIC for sglang's DP-attention GLOO barrier |

### What each run does

| | StreamInfer | sglang EP |
|---|---|---|
| env | `streaminfer` (torch 2.6) | `sglang` (torch 2.8) |
| launch | Ray head + `benchmark/server.py` (one server, all rates) | `sglang.launch_server` per node — head SSH-launches workers; fresh server + retries per rate |
| parallelism | `--dp-size W --ep-size W` colocate | `--tp/dp/ep-size W` + DP-attention + `mooncake-nccl` a2a |
| layers / prefill | `--num-layers N` | `--num-hidden-layers-override N` + `--enable-fake-prefill` |
| weights | dummy | `--load-format dummy` |
| benchmark | `POST /run_once` (rate, time) | `sglang.bench_serving` (`--request-rate`, `--num-prompts`) |

**sglang uses expert parallelism** to match the paper: `--tp-size W --dp-size W --ep-size W --enable-dp-attention
--enable-dp-lm-head --moe-a2a-backend mooncake-nccl` (`mooncake-nccl` = standard NCCL
EP dispatch; **no Mooncake package needed**), uncapped (no `--max-running-requests`).
The `sglang_dummy_prefill` fork carries three fixes required to run EP under load:
the GLOO barrier is moved off the datapath NIC (`GLOO_IFNAME`), and two DP-attention
batch-size off-by-one clamps (`overlap_utils.py`, `forward_batch_info.py`).

### Common issues on other clusters

- **NCCL can't find the IB/RoCE device** — set `NCCL_IB_HCA` to your device name(s)
  (`ibstat -l`); use the right `NCCL_IB_GID_INDEX` for RoCE v2. If nodes name the NIC
  differently, pass a comma-list.
- **sglang DP-attention drops connections / hangs under load** — its per-decode-step
  GLOO all-gather barrier is starved when it shares the NCCL datapath NIC. Point
  `GLOO_IFNAME` at a second NIC; on a single-NIC cluster set `GLOO_IFNAME=$HOST_IFNAME`
  and expect less headroom.
- **sglang OOM at high rate** — lower `MEM_FRAC_HIGH` (smaller KV pool → more
  activation headroom → the decode batch fits).
- **sglang bench `Cannot assign requested address`** — the client exhausted ephemeral
  ports; widen `net.ipv4.ip_local_port_range` (see the sysctl above).
- **StreamInfer** requires `gdrdrv` loaded and `disagmoe_c` built in the env on **every**
  node — see the install pointers in the root [`README.md`](../../README.md).

---


| Aspect | Original (sphere-16) | Here |
|---|---|---|
| Scale | 8 nodes × 2 L40S = 16 GPUs, 36 layers | 4 nodes × 2 L40S = 8 GPUs, 18 layers (same per-GPU work) |
| Gate profiles | balanced profiles | balanced sharegpt profile (both systems) |
| Datasets | sharegpt + gsm8k | sharegpt only |
| Rates | 5–6 per dataset | 25, 50, 100, 150, 200 (top rate raised from the 16-GPU sweep's 150 — sglang's 8-rank barrier collapses later) |
| Systems | StreamInfer + several sglang profiles | StreamInfer + sglang EP |
