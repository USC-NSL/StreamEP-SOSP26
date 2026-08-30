# Why PP Is Underutilizing KV/VRAM in sgl-040

## Question
Why do `PP8xTP2` / `PP8xEP2` show much lower effective running batch / throughput than EP-style runs, even when total model replication is still 1x?

## Observed in sgl-040
From `experiments/sgl-040/*/bench.log`:

- `pp8tp2_legal`: 791.33s, 4849.30 tok/s, median ITL 410.97 ms
- `pp8tp2_balanced_legal`: 775.52s, 4948.20 tok/s, median ITL 402.50 ms
- `pp8ep2_legal`: 843.23s, 4550.85 tok/s, median ITL 429.08 ms
- `pp8ep2_balanced_legal`: 899.32s, 4267.01 tok/s, median ITL 480.65 ms

So `PP8xEP2` was slower than `PP8xTP2` here, and both are far below EP16 throughput from earlier runs.

## What The Current Code Explicitly Does

### 1) PP forces non-overlap scheduling
- `python/sglang/srt/server_args.py:1488` sets `disable_overlap_schedule=True` when `pp_size > 1`
- `python/sglang/srt/server_args.py:3714` enforces PP incompatibility with overlap/speculative/mixed-chunk

This removes one major throughput optimization path for decode.

### 2) Runtime token capacity is reduced to the weakest PP stage
- `python/sglang/srt/model_executor/model_runner.py:1571` profiles `max_total_num_tokens`
- `python/sglang/srt/model_executor/model_runner.py:1633` applies PP min-reduction across ranks

So PP does not use per-stage headroom independently; it clamps to the minimum stage.

### 3) `max_running_requests` is bounded by both token capacity and request-pool size
- `python/sglang/srt/managers/tp_worker.py:281`
- Formula: `min(max_total_num_tokens // 2, req_to_token_pool.size)` (default case)
- `model_runner.py:1575-1584` also caps estimated request pool in `[2048, 4096]`

This means PP can become request-slot limited before token/KV limited on short-to-medium request mixes.

### 4) PP default admission budget is divided by pipeline size
- `python/sglang/srt/managers/scheduler.py:368`
- If not set manually: `pp_max_micro_batch_size = max_running_requests // pp_size`

For PP8, that is an 8x split of the running-request budget per microbatch scheduler lane.

### 5) Admission path enforces that per-microbatch cap
- `python/sglang/srt/managers/scheduler.py:1711`
- `get_num_allocatable_reqs = pp_max_micro_batch_size - running_bs`
- PP additionally caps by `req_to_token_pool.available_size()`
- Used as hard gate in prefill admission (`scheduler.py:1738`, `scheduler.py:1790`)

## Why They Likely Designed It This Way

## Explicitly stated in code/comments/docs
1. **Correctness and memory-safety across PP microbatches**
   - `python/sglang/srt/managers/scheduler.py:1736` says PP chunked requests can span microbatches and strict microbatch handling can cause memory leaks.
   - This strongly indicates conservative admission policy is partly a safety mechanism, not purely throughput-oriented.

2. **PP architecture is still focused on long-context / TTFT use-cases**
   - Upstream PP doc is explicitly framed as long-context PP (`docs/advanced_features/pipeline_parallelism.md`).
   - Upstream PP roadmap (`sglang` issue #11857) tracks high-throughput decode as an ongoing optimization area.

3. **PP dynamic chunking is upstream-documented as a bubble-mitigation mechanism**
   - Upstream PP docs say fixed chunking causes bubbles, especially for larger PP sizes, and describe dynamic chunking to reduce this (`docs/advanced_features/pipeline_parallelism.md`).

Reference links (upstream):
- https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pipeline_parallelism.md
- https://github.com/sgl-project/sglang/issues/11857

## Inferred (architecture-grounded, not explicitly stated)
1. **Pipeline bubble risk drives conservative microbatch admission**
   - With PP8, small or imbalanced microbatches cause idle pipeline stages.
   - A conservative cap can reduce scheduling pathologies and state explosions, but can underfill KV relative to EP.

2. **Non-overlap PP + microbatch slicing trades throughput for scheduler simplicity/stability**
   - Since PP disables overlap scheduling, higher admitted global batch does not necessarily scale linearly in decode throughput.
   - Maintainers likely prioritize robust PP semantics first, then optimize throughput incrementally.

3. **Single weakest-stage token limit + per-microbatch cap compounds underutilization**
   - Min-across-stages token cap and `// pp_size` admission slicing can jointly leave available VRAM/KV headroom unused.

4. **PP correctness constraints likely dominate admission policy priority**
   - The explicit chunked-request memory-leak comment in scheduler suggests robustness constraints are prioritized over aggressively filling KV at all times.

5. **The default `//2` running-request headroom and `//pp_size` microbatch split compound**
   - `max_running_requests` defaulting to `max_total_num_tokens // 2` and then `pp_max_micro_batch_size = max_running_requests // pp_size` can create a conservative two-stage cap before requests even hit true KV limits.

## Why This Looks Worse Than Intuition
Intuition says: same total weights replication => similar total KV budget => similar global running batch.

But current PP policy does not optimize directly for "max global admitted requests":
- it optimizes per-microbatch admissibility,
- with conservative defaults tied to `pp_size`,
- and with PP-specific correctness constraints (chunked req spanning microbatches).

So practical admitted/running batch can be much lower than raw KV capacity suggests.

## Practical Validation Knobs (No core code changes)
To validate the hypotheses above:

1. Override `pp_max_micro_batch_size` upward (within allowed range) and measure throughput/ITL/regressions.
2. Sweep `max_running_requests` with fixed `pp_size=8` and track running-batch timeline from recorder.
3. Compare `pp8tp2` with and without `moe-a2a-backend mooncake-nccl` (EP2 inside TP group) under same mem-frac.
4. Profile stage-level token capacity and confirm which stage sets the PP min-cap.
5. Run shorter decode-only sweeps with varied `mem-fraction-static` to test if admission, not memory, is the dominant limiter.
6. Sweep `chunked-prefill-size` (including `-1`) to test sensitivity to PP chunked-request cross-microbatch bookkeeping.

## Quick-Fix Feasibility (Fake-Prefill PP)

The following are the most practical short-term options to raise PP effective global concurrency and throughput.

| Option | Type | Feasibility | Expected Impact | Main Risk |
|---|---|---|---|---|
| Increase `--pp-max-micro-batch-size` toward `max_running_requests // pp_size` ceiling | Launch-only | High | Medium-High | OOM / scheduler instability if pushed too far |
| Increase `--max-running-requests` (then re-tune `pp_max_micro_batch_size`) | Launch-only | High | Medium | Can hit request-pool/token-pool limits first; possible regressions in tail latency |
| Increase `--mem-fraction-static` for PP stages | Launch-only | Medium | Low-Medium | CUDA graph capture OOM on weakest stage |
| Keep fake-prefill but reduce chunking complexity (`--chunked-prefill-size -1`) when possible | Launch-only | Medium | Low-Medium | May reduce robustness for mixed/chunked cases |
| Increase `--max-queued-requests` when burst admission is queue-limited | Launch-only | Medium | Low | Higher queueing delay, bigger host memory footprint |
| Patch scheduler default from `max_running_requests // pp_size` to a larger policy (or explicit override) | Small code patch | Medium | High | Correctness risk around chunked cross-microbatch bookkeeping |
| Patch fake-prefill PP admission to use global running cap (across all microbatches) instead of per-microbatch cap | Small code patch | Medium | High | Could destabilize microbatch balance and memory spikes |

### Recommended quick sequence
1. Keep code unchanged first; tune launch args in this order: `pp_max_micro_batch_size` -> `max_running_requests` -> `mem-fraction-static`.
2. For each step, gate by: no OOM, no stuck scheduler, and improved output token throughput.
3. Only if tuning saturates early, test a tiny scheduler policy patch in a separate branch.

### Why this is feasible for fake-prefill
- Fake-prefill removes real prefill compute pressure, so decode-path scheduling/admission limits dominate behavior.
- That makes PP admission knobs (`pp_max_micro_batch_size`, `max_running_requests`) the highest-leverage fast controls before any major code refactor.

### Concrete patch points (if tuning is insufficient)
- `python/sglang/srt/managers/scheduler.py:1711` (`get_num_allocatable_reqs`): add fake-prefill-specific global cap logic using total running across `running_mbs` instead of one-microbatch `running_bs`.
- `python/sglang/srt/managers/scheduler.py:368` (default PP microbatch cap): relax initialization policy for fake-prefill runs.
- `python/sglang/srt/managers/scheduler.py:2486` (runtime cap validation): if needed, widen permissible runtime update range for `pp_max_micro_batch_size` in fake-prefill-only experiments.

### Upstream-backed fast knobs (and branch availability)
- Upstream evidence strongly supports making `pp_max_micro_batch_size` explicit (instead of inheriting `max_running_requests // pp_size`) as a first-pass PP tuning control.
- Upstream docs/issues also indicate `max_running_requests`, chunking strategy, and CUDA-graph sizing as practical near-term levers.
- Upstream async PP depth (`pp_async_batch_depth`) has reported throughput gains in newer branches, but this fork currently does not expose that flag (`pp_async_batch_depth` not found under `python/sglang/srt`).
- Therefore, on this branch, highest-leverage quick fixes remain the scheduler admission knobs and memory/request capacity knobs listed above.

## Bottom Line
`PP8xTP2/PP8xEP2` low throughput in this codebase is consistent with current scheduler design: conservative per-microbatch admission, PP non-overlap mode, and weakest-stage token-cap clamping. The behavior is explainable by current implementation choices, and upstream docs/issues suggest PP throughput optimization is still evolving.
