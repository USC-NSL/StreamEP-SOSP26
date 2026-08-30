# sgl-043 Report Summary

## Scope
- EP16 balanced-legal mem-frac sweep on sphere-16.
- Workload: 10k requests, 2000 rps, random input/output [256,512], fake-prefill, recorder on.
- Mem-frac values tested: `0.80`, `0.77`, `0.70`, `0.60`, `0.57`, `0.55`, `0.50`.

## Results
| Run | Status | Duration(s) | Req/s | Output tok/s | Steady-State tok/s (>=90% peak global concurrency) | Bench Concurrency | Median ITL(ms) | Peak Running (recorder) | Peak VRAM Alloc (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mem_frac=0.8` | failed | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `mem_frac=0.77` | ok | 436.10 | 22.93 | 8799.47 | 9281.31 | 6978.87 | 457.83 | 6479.00 | 38.31 |
| `mem_frac=0.7` | ok | 403.55 | 24.78 | 9509.24 | 10055.95 | 6729.52 | 430.92 | 5796.00 | 34.95 |
| `mem_frac=0.6` | ok | 419.82 | 23.82 | 9140.55 | 9717.52 | 6113.49 | 355.77 | 4133.00 | 30.07 |
| `mem_frac=0.57` | ok | 470.64 | 21.25 | 8153.58 | 9005.44 | 5809.53 | 347.59 | 3553.00 | 28.59 |
| `mem_frac=0.55` | ok | 472.41 | 21.17 | 8123.08 | 8822.28 | 5751.10 | 316.68 | 3179.00 | 27.62 |
| `mem_frac=0.5` | ok | 476.79 | 20.97 | 8048.36 | 8600.54 | 5512.76 | 232.11 | 2224.00 | 24.90 |

## Key Takeaways
- `mem_frac=0.80` failed during decode with CUDA OOM in logits all-gather and is excluded from the plotted curves.
- `mem_frac=0.70` still gives the best end-to-end output throughput and the best steady-state throughput among successful runs.
- `mem_frac=0.77` increases peak global running requests and VRAM allocation versus `0.70`, but still underperforms `0.70` in throughput.
- `mem_frac=0.57` slightly outperforms `0.55` and sits between `0.60` and `0.55` in both throughput and recorder peak concurrency.
- `per_step_time_breakdown.png` shows average decode-step time split into Attention / MoE / AllGather / AllReduce / Other from `moe_kernel_balance` recorder data.

## Plots
- `experiments/sgl-043/plots/global_running_requests_timeline.png`
- `experiments/sgl-043/plots/summary_overview.png`
- `experiments/sgl-043/plots/vram_timeline.png`
- `experiments/sgl-043/plots/per_step_time_breakdown.png`
- `experiments/sgl-043/plots/summary_metrics.csv`
