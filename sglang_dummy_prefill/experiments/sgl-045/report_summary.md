# sgl-045 Report Summary

## Scope
- EP16 balanced-legal on sphere-16 with two-node TCP interference on `sgpu0<->sgpu2` using the updated calibrated interference generator.
- Workload: 10k requests, 2000 rps, random input/output [256,512], fake-prefill, recorder on.
- Interference traces: `aws_hpc_metal`, `azure_hpc_200g`.
- Mem-frac values tested: `0.77`, `0.55`.
- Includes `sgl-043` balanced-legal no-interference baselines for comparison.

## Results
| Run | Duration(s) | Req/s | Output tok/s | Steady-State tok/s | Median ITL(ms) |
|---|---:|---:|---:|---:|---:|
| `baseline 0.77` | 436.10 | 22.93 | 8799.47 | 9281.31 | 457.83 |
| `aws 0.77` | 478.29 | 20.91 | 8023.11 | 8017.31 | 531.64 |
| `azure 0.77` | 473.44 | 21.12 | 8105.30 | 8173.11 | 517.39 |
| `baseline 0.55` | 472.41 | 21.17 | 8123.08 | 8822.28 | 316.68 |
| `aws 0.55` | 507.49 | 19.70 | 7561.49 | 8934.77 | 317.65 |
| `azure 0.55` | 508.46 | 19.67 | 7547.16 | 8930.61 | 317.63 |

## Key Takeaways
- With the updated calibrated interference generator, both AWS and Azure traces now clearly reduce throughput versus the `sgl-043` no-interference baselines at both mem-frac values.
- At `0.77`, throughput drops from `8799.47 tok/s` (baseline) to `8023.11` (AWS, `-8.82%`) and `8105.30` (Azure, `-7.89%`).
- At `0.55`, throughput drops from `8123.08 tok/s` (baseline) to `7561.49` (AWS, `-6.91%`) and `7547.16` (Azure, `-7.09%`).
- AWS and Azure produce very similar degradation. AWS is slightly worse at `0.77`; Azure is slightly worse at `0.55`, but the difference is small.
- Compared with the earlier sgl-045 run, the updated generator now produces the intended measurable interference effect.
- Bandwidth profile CSV/PNG outputs from the updated interference generator were captured inside each run directory and should be inspected alongside the SGLang plots.

## Plots
- `experiments/sgl-045/plots/global_running_requests_timeline.png`
- `experiments/sgl-045/plots/summary_overview.png`
- `experiments/sgl-045/plots/vram_timeline.png`
- `experiments/sgl-045/plots/per_step_time_breakdown.png`
- `experiments/sgl-045/plots/summary_metrics.csv`
