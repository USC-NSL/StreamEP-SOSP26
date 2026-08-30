# Supporting DeepEP on A100s for SGLang (Fake-Prefill Fork)

## Background

We have two SGLang forks:

| Fork | Location | What it has | What it lacks |
|------|----------|-------------|---------------|
| **sglang-fake-prefill** | `/home1/yizhuoli/sglang-fake-prefill` | Fake-prefill hacks (profile-driven MoE routing, decoding-only baseline), upstream DeepEP code paths | A100-specific DeepEP adaptations |
| **sglang-a100-deepep** | `/home1/yizhuoli/sglang-a100-deepep` | A100-adapted DeepEP support (Triton runner glue, bf16 paths) | Fake-prefill functionality |

**Goal**: Port the A100 DeepEP adaptations into the fake-prefill fork, then run expert parallelism on a 4×A100 node.

---

## What is DeepEP and Why Does A100 Need Special Treatment?

[DeepEP](https://github.com/deepseek-ai/DeepEP) is DeepSeek's CUDA library for MoE expert-parallel all-to-all communication ("dispatch" and "combine"). It has two kernel families:

| Kernel Mode | Purpose | GPU Requirement |
|-------------|---------|-----------------|
| **Normal** | High-throughput prefill/batch dispatch | SM80+ (A100 ✅) |
| **Low-latency** | Fast decode with CUDA graph support | SM90 only (H100/H800), uses TMA/mbarrier |

### A100 (SM80) vs H100 (SM90) — What Matters for DeepEP

| Feature | A100 (SM80) | H100 (SM90) | DeepEP Impact |
|---------|-------------|-------------|---------------|
| TMA (Tensor Memory Accelerator) | ❌ | ✅ | Internode + low-latency kernels need TMA |
| `cp.async.bulk` PTX | ❌ | ✅ | Used in intranode fast paths |
| NVLink bandwidth | 600 GB/s | 900 GB/s | Lower throughput but functional |
| NVSHMEM/IBGDA | Supported | Supported | Required only for internode (not us) |

### DeepEP's Official A100 Stance

DeepEP's README roadmap marks **"A100 support (intranode only)"** as done. Building with `DISABLE_SM90_FEATURES=1` compiles only SM80-compatible kernels and disables:
- Internode kernels (`internode.cu`, `internode_ll.cu`)
- Low-latency mode
- FP8 launch methods
- TMA-based paths

**For our 4×A100 single-node setup, intranode-only is exactly what we need.** We don't need internode communication.

---

## What the sglang-a100-deepep Fork Changed

The colleague's fork made these key adaptations (commits on branch `debug_a100_deepep`):

### 1. Triton MoE Runner ↔ DeepEP Normal-Mode Glue (`4ae15fcb7`)

**Problem**: DeepEP's "normal dispatch" output format doesn't match what the default MoE runners expect. On H100, `deep_gemm` (FP8 grouped GEMM) handles this. On A100, `deep_gemm` may not be available (it targets SM90 features).

**Solution**: Added `deepep_normal` format adapters to the Triton MoE runner:
- `@register_pre_permute("deepep_normal", "triton")` — converts DeepEP normal dispatch output into Triton-compatible layout
- `@register_post_permute("triton", "deepep_normal")` — converts Triton output back for DeepEP combine
- When `moe_a2a_backend=deepep` and deep_gemm is unavailable, sets `no_combine = True` in the runner config (combine is handled by the DeepEP dispatcher instead)

**Files**: `layers/moe/moe_runner/triton.py`, `layers/moe/fused_moe_triton/layer.py`

### 2. BF16 Support for DeepGEMM Runner (`be3430a10`, `ef4f76aac`)

**Problem**: DeepGEMM was FP8-only. Normal-mode dispatch on A100 uses BF16 tensors.

**Solution**: Added bf16 grouped GEMM execution paths:
- New bf16 kernel types + warmup executors in the deep_gemm compile utils
- bf16 grouped GEMM entrypoints (`grouped_gemm_nt_bf16bf16bf16_*`)
- Runner splits execution into FP8 vs BF16 paths based on tensor dtype
- Unquantized MoE routes to DeepGEMM backend when EP is active

**Files**: `layers/deep_gemm_wrapper/compile_utils.py`, `layers/deep_gemm_wrapper/entrypoint.py`, `layers/moe/moe_runner/deep_gemm.py`, `layers/quantization/unquant.py`

### 3. Empty-Input Tolerance (`af7212a2e`)

**Problem**: After DeepEP dispatch, some ranks may receive zero tokens for certain experts, causing crashes in Triton kernels.

**Solution**: Added guards for empty-input cases in the Triton runner and EP MoE kernels.

**Files**: `layers/moe/moe_runner/triton.py`, `layers/moe/ep_moe/kernels.py`, `layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`

### 4. W4A8 Hotfix (`42889acbd`)

**Solution**: Fixed condition for quantize/route decisions in DeepEP normal dispatch — blocks `deep_gemm` when the runner backend is cutlass.

**Files**: `layers/moe/token_dispatcher/deepep.py`

---

## Feasibility Assessment

### Verdict: ✅ Highly Feasible

| Factor | Assessment |
|--------|------------|
| **Hardware fit** | 4×A100 = single node. DeepEP officially supports A100 intranode. We don't need internode/low-latency. |
| **Reference implementation exists** | The colleague's fork already proves it works. This is a porting task, not invention. |
| **Code overlap** | Both forks share the same SGLang base. The fake-prefill fork already has DeepEP code paths from upstream — we just need the A100-specific adaptations. |
| **Fake-prefill compatibility** | Fake-prefill operates at the scheduling/routing layer (`scheduler.py`, `profile_driven_router.py`). DeepEP operates at the dispatch/communication layer (`token_dispatcher/deepep.py`). These are largely orthogonal. |
| **DeepEP build on A100** | Straightforward: `DISABLE_SM90_FEATURES=1 pip install .` — no NVSHMEM needed for intranode. |

### Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Fork divergence — the two forks may be based on different SGLang versions | Medium | Diff carefully; port semantics, not blind patches |
| deep_gemm availability on A100 | Low | The Triton runner fallback path was built exactly for this |
| CUDA/driver compatibility | Low | Node has CUDA 12.6.3 via module load; DeepEP needs CUDA 11.0+ for SM80 |
| Fake-prefill + EP interaction edge cases | Medium | Profile-driven routing may need adjustment for EP topology |

Human notes:
You can use "module spider cuda" to look for other available cuda versions,
not just 12.6.3

---

## Execution Plan

### Phase 0: Environment Setup
1. Create conda env `sglang-fp` (Python 3.12) under `/scratch1/yizhuoli/conda-envs/`
2. `module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda`
3. Install PyTorch (CUDA 12.x compatible)
4. Build & install DeepEP from source with `DISABLE_SM90_FEATURES=1` (no NVSHMEM needed)
5. Install `sglang-fake-prefill` in dev mode

### Phase 1: Port A100 DeepEP Adaptations
Compare and port changes from `sglang-a100-deepep` into `sglang-fake-prefill`, file by file:

| Priority | File(s) | What to port |
|----------|---------|--------------|
| P0 | `layers/moe/moe_runner/triton.py` | `deepep_normal` ↔ `triton` pre/post-permute adapters |
| P0 | `layers/moe/fused_moe_triton/layer.py` | `deep_gemm` gating logic, `no_combine` hack for DeepEP+Triton |
| P0 | `layers/moe/ep_moe/kernels.py` | Empty-input guards |
| P1 | `layers/deep_gemm_wrapper/compile_utils.py` | BF16 kernel types (if deep_gemm is usable on A100) |
| P1 | `layers/deep_gemm_wrapper/entrypoint.py` | BF16 grouped GEMM entrypoints |
| P1 | `layers/moe/moe_runner/deep_gemm.py` | BF16 execution paths |
| P1 | `layers/quantization/unquant.py` | DeepGEMM routing for unquantized MoE |
| P2 | `layers/moe/token_dispatcher/deepep.py` | W4A8 hotfix condition |
| P2 | `server_args.py` | Verify CLI args match (likely already present) |

### Phase 2: Smoke Test
1. Launch SGLang with a small MoE model on 4×A100 with `--moe-a2a-backend deepep --deepep-mode normal`
2. Verify basic inference works (single request)
3. Verify fake-prefill mode still works (`--enable-fake-prefill`)
4. Verify both combined: fake-prefill + DeepEP EP

### Phase 3: Validation
1. Run DeepEP tuning script to generate config for A100 NVLink topology
2. Test with production-size MoE model
3. Compare outputs against non-EP baseline for correctness
4. Benchmark throughput

---

## Key Files Reference

### Fake-Prefill Layer (already in this fork)
- `python/sglang/srt/server_args.py` — `--enable-fake-prefill` flag
- `python/sglang/srt/managers/scheduler.py` — fake-prefill scheduling logic
- `python/sglang/srt/layers/moe/profile_driven_router.py` — pre-profiled routing override

### DeepEP Integration Layer (needs A100 adaptations)
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` — dispatch/combine implementation
- `python/sglang/srt/layers/moe/utils.py` — `MoeA2ABackend`, `DeepEPMode` enums
- `python/sglang/srt/layers/moe/ep_moe/layer.py` — `DeepEPMoE` layer
- `python/sglang/srt/model_executor/cuda_graph_runner.py` — DeepEP CUDA graph adapter

### A100-Specific Adaptations (to be ported)
- `python/sglang/srt/layers/moe/moe_runner/triton.py` — Triton ↔ DeepEP format bridge
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` — deep_gemm fallback logic
- `python/sglang/srt/layers/deep_gemm_wrapper/` — BF16 kernel support
