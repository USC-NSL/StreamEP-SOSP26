# sgl-046 Report Summary

## Scope
- EP16 balanced-legal comparison on sphere-16.
- Workload: 10k requests, 2000 rps, random input/output [256,512], fake-prefill, recorder on.
- Compares original `sgl-043` `mem_frac=0.77` run against a new `sgl-046` run with `--max-running-requests 3240`.

## Results
| Run | Duration(s) | Req/s | Output tok/s | Steady-State tok/s | Median ITL(ms) |
|---|---:|---:|---:|---:|---:|
| `baseline 0.77` | 436.10 | 22.93 | 8799.47 | 9281.31 | 457.83 |
| `cap3240 0.77` | 447.68 | 22.34 | 8571.77 | 9165.35 | 341.94 |

## Delta vs baseline
- Output throughput: `-227.70 tok/s` (`-2.59%`)
- Duration: `+11.58s`
- Median ITL: `-115.89 ms`
- Peak running requests (recorder): `-3247.00`

## Key Takeaways
- Capping `max-running-requests` to `3240` reduced effective global concurrency and reduced throughput relative to the original `sgl-043` `0.77` run.
- This directly measures the cost of roughly halving the scheduler's allowed running-request budget under the same workload and mem-frac setting.

## Plots
- `experiments/sgl-046/plots/global_running_requests_timeline.png`
- `experiments/sgl-046/plots/summary_overview.png`
- `experiments/sgl-046/plots/vram_timeline.png`
- `experiments/sgl-046/plots/per_step_time_breakdown.png`
- `experiments/sgl-046/plots/summary_metrics.csv`
