# sgl-040 Report Summary

Generated: 2026-03-20 04:30:08 UTC

## Scope
- Cluster: sphere-16 (8 nodes x 2 L40S = 16 GPUs)
- Model: `lmsys/gpt-oss-120b-bf16` (`--load-format dummy`)
- Workload: random dataset, 10,000 requests, 2000 rps
- Length distribution: input uniform [256, 512], output uniform [256, 512]
  - via `--random-input-len 512 --random-output-len 512 --random-range-ratio 0.5`
- Fake prefill: enabled
- Recorder: enabled (`--expert-distribution-recorder-mode stat`)
- Memory fraction: `--mem-fraction-static 0.80`
- Profiles:
  - `gating_legal_court_opinions_200.parquet`
  - `balanced_legal_court_opinions_200.parquet`

## Configs Run
1. `pp8tp2` + legal
2. `pp8tp2` + balanced_legal
3. `pp8ep2` + legal
4. `pp8ep2` + balanced_legal

## Results
| Run | Successful Requests | Duration (s) | Request Throughput (req/s) | Output Throughput (tok/s) | Median ITL (ms) |
|---|---:|---:|---:|---:|---:|
| `pp8tp2_legal` | 10000 | 791.33 | 12.64 | 4849.30 | 410.97 |
| `pp8tp2_balanced_legal` | 10000 | 775.52 | 12.89 | 4948.20 | 402.50 |
| `pp8ep2_legal` | 10000 | 843.23 | 11.86 | 4550.85 | 429.08 |
| `pp8ep2_balanced_legal` | 10000 | 899.32 | 11.12 | 4267.01 | 480.65 |

## Recorder Artifacts
Each run has recorder output in `recorder_raw/`.
- `pp8tp2_legal`: 2 `.pt` files
- `pp8tp2_balanced_legal`: 2 `.pt` files
- `pp8ep2_legal`: 2 `.pt` files
- `pp8ep2_balanced_legal`: 2 `.pt` files

## Quick Takeaways
- `pp8tp2` outperformed `pp8ep2` on both legal profiles in this run set.
- `balanced_legal` was slightly better than `legal` for `pp8tp2`, but worse for `pp8ep2`.
- `pp8ep2_balanced_legal` was the slowest of the four runs.
