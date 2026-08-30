# sgl-044 Report Summary

## Scope
- EP16 legal + balanced-legal on sphere-16.
- Workload: 10k requests, 2000 rps, random input/output [512,1024], fake-prefill, recorder on.
- Mem-frac values tested: `0.77`, `0.55` for both profiles.

## Results
| Run | Duration(s) | Req/s | Output tok/s | Steady-State tok/s | Bench Concurrency | Median ITL(ms) |
|---|---:|---:|---:|---:|---:|---:|
| `balanced 0.77` | 940.34 | 10.63 | 8185.98 | 8805.88 | 6023.93 | 363.86 |
| `balanced 0.55` | 1177.88 | 8.49 | 6535.13 | 6993.82 | 5345.36 | 207.33 |
| `legal 0.77` | 982.58 | 10.18 | 7834.09 | 8364.99 | 6055.29 | 382.36 |
| `legal 0.55` | 1190.42 | 8.40 | 6466.28 | 6878.83 | 5375.75 | 211.04 |

## Plots
- `experiments/sgl-044/plots/global_running_requests_timeline.png`
- `experiments/sgl-044/plots/summary_overview.png`
- `experiments/sgl-044/plots/vram_timeline.png`
- `experiments/sgl-044/plots/per_step_time_breakdown.png`
- `experiments/sgl-044/plots/summary_metrics.csv`
