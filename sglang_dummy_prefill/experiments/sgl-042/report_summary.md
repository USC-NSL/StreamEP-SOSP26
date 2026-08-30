# sgl-042 Report Summary

Generated: 2026-03-20 07:05:39 UTC

## Scope
- Single-case run to test higher request-pool cap.
- Config: PP8xTP2, legal profile, 10k requests, 2000 rps, random input/output [256,512], fake-prefill, recorder on, mem-frac 0.80.
- Code change under test: `python/sglang/srt/model_executor/model_runner.py` request-pool estimate cap increased from `4096` to `8192`.

## sgl-042 Result
- Successful requests: `10000`
- Benchmark duration: `781.76s`
- Request throughput: `12.79 req/s`
- Output throughput: `4908.69 tok/s`
- Bench concurrency: `5934.30`
- Recorder peak running reqs: `429.00`
- Recorder peak VRAM allocated: `24.27 GB`

## Comparison (PP8xTP2 legal)
| Case | Duration(s) | Output tok/s | Bench Concurrency | Peak Running (recorder) | Peak VRAM Alloc (GB) |
|---|---:|---:|---:|---:|---:|
| `sgl-040_pp8tp2_legal` | 791.33 | 4849.30 | 5962.20 | 429.00 | 22.11 |
| `sgl-041_pp8tp2_legal_token_only` | 786.73 | 4877.65 | 5934.30 | 429.00 | 22.11 |
| `sgl-042_pp8tp2_legal_reqcap8192` | 781.76 | 4908.69 | 5934.30 | 429.00 | 24.27 |

## Delta vs sgl-040 baseline
- Duration: `-9.57s`
- Output throughput: `+59.39 tok/s` (`+1.22%`)
- Bench concurrency: `-27.90`
- Recorder peak running reqs: `+0.00`
- Recorder peak VRAM allocated: `+2.16 GB`

## Interpretation
- Increasing request-pool cap to 8192 did not materially raise running concurrency in this case.
- This indicates another scheduler/admission limiter (e.g., PP microbatch cap and/or token allocator constraints) is still dominant for this workload.

## Artifacts
- Bench log: `experiments/sgl-042/pp8tp2_legal_reqcap8192/bench.log`
- Recorder raw: `experiments/sgl-042/pp8tp2_legal_reqcap8192/recorder_raw/`
