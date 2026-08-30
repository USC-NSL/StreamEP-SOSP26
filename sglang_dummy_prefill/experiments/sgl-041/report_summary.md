# sgl-041 Report Summary

Generated: 2026-03-20 06:25:07 UTC

## Scope
- Same setup as `sgl-040` (PP8xTP2 + PP8xEP2 on legal/balanced_legal, 10k req, 2000 rps, random in/out [256,512], fake-prefill, recorder on, mem-frac 0.80).
- Change in `sgl-041`: enabled `SGLANG_PP_TOKEN_ONLY_MAX_RUNNING=true` on head and workers.
- Code path changed: `python/sglang/srt/managers/tp_worker.py` allows PP max-running to follow token-derived policy without clamping by `req_to_token_pool.size` in this experimental mode.

## sgl-041 Results
| Run | Success | Duration(s) | Req/s | Output tok/s | Median ITL(ms) | Bench Concurrency | Peak Running (recorder) | Peak VRAM Alloc (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pp8tp2_legal` | 10000 | 786.73 | 12.71 | 4877.65 | 404.69 | 5934.30 | 429.00 | 22.11 |
| `pp8tp2_balanced_legal` | 10000 | 786.58 | 12.71 | 4878.63 | 405.61 | 5935.56 | 429.00 | 22.11 |
| `pp8ep2_legal` | 10000 | 843.93 | 11.85 | 4547.09 | 429.86 | 5841.53 | 420.00 | 22.10 |
| `pp8ep2_balanced_legal` | 10000 | 899.25 | 11.12 | 4267.35 | 481.69 | 5828.60 | 424.00 | 22.12 |

## Delta vs sgl-040 (sgl-041 - sgl-040)
| Run | Δ Duration(s) | Δ Req/s | Δ Output tok/s | Δ Median ITL(ms) | Δ Bench Concurrency | Δ Peak Running (recorder) | Δ Peak VRAM Alloc (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pp8tp2_legal` | -4.60 | +0.07 | +28.35 | -6.28 | -27.90 | +0.00 | -0.00 |
| `pp8tp2_balanced_legal` | +11.06 | -0.18 | -69.57 | +3.11 | +5.75 | +0.00 | -0.01 |
| `pp8ep2_legal` | +0.70 | -0.01 | -3.76 | +0.78 | -21.46 | -8.00 | -0.02 |
| `pp8ep2_balanced_legal` | -0.07 | +0.00 | +0.34 | +1.04 | +10.13 | -4.00 | -0.00 |

## Interpretation
- Mean output throughput across 4 runs: `sgl-040=4653.84 tok/s`, `sgl-041=4642.68 tok/s` (`-11.16` absolute, `-0.24%` relative).
- Mean recorder peak running requests: `sgl-040=428.50`, `sgl-041=425.50` (`-3.00`).
- Mean recorder peak VRAM allocated: `sgl-040=22.12 GB`, `sgl-041=22.11 GB` (`-0.01 GB`).
- In this run, enabling token-only max-running did not produce a material concurrency/VRAM gain; performance remained very close to sgl-040.

## Artifacts
- `pp8tp2_legal`: `experiments/sgl-041/pp8tp2_legal/bench.log`, `experiments/sgl-041/pp8tp2_legal/recorder_raw/`
- `pp8tp2_balanced_legal`: `experiments/sgl-041/pp8tp2_balanced_legal/bench.log`, `experiments/sgl-041/pp8tp2_balanced_legal/recorder_raw/`
- `pp8ep2_legal`: `experiments/sgl-041/pp8ep2_legal/bench.log`, `experiments/sgl-041/pp8ep2_legal/recorder_raw/`
- `pp8ep2_balanced_legal`: `experiments/sgl-041/pp8ep2_balanced_legal/bench.log`, `experiments/sgl-041/pp8ep2_balanced_legal/recorder_raw/`
