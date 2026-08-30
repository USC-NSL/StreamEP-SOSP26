# DeepEP + A100 + Fake-Prefill: Change Log

Base commit: `5aa18370f` (profile-driven routing with cudagraph compatibility support)

Goal: Run SGLang with DeepEP expert parallelism on 4×A100-SXM4-80GB (SM80), combined with the fake-prefill feature already in this fork.

Tested with: `openai/gpt-oss-20b` (32 experts, top-4, hidden_size=2880, mxfp4 quantization, 24 layers).

Launch command:
```bash
python -m sglang.launch_server \
  --model openai/gpt-oss-20b \
  --tp 4 --ep-size 4 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --host 0.0.0.0 --port 30005 \
  --disable-cuda-graph --mem-fraction-static 0.8 \
  --enable-fake-prefill  # optional
```

---

## 1. DeepEP Build Fix (external, not in this repo)

**File**: `/scratch1/yizhuoli/DeepEP/csrc/deep_ep.cpp` (lines 1825-1842)

**Problem**: Building DeepEP with `DISABLE_SM90_FEATURES=1` for A100 fails because three `low_latency_*_mask_buffer` functions reference `internode_ll::*` symbols that are compiled out on SM80.

**Fix**: Wrapped the three functions with `#ifndef DISABLE_NVSHMEM` guards to exclude them when SM90 features are disabled.

**Build command**:
```bash
DISABLE_SM90_FEATURES=1 pip install .
```

---

## 2. SGLang Changes (8 files)

### 2a. `python/sglang/srt/layers/moe/moe_runner/triton.py`

**What**: Added ~170 lines — two adapter functions that bridge DeepEP's dispatch format to the Triton MoE runner.

**Why**: DeepEP dispatch returns `DeepEPNormalDispatchOutput` (tokens grouped by source rank, with per-expert counts). The Triton fused-MoE kernel expects tokens sorted by expert with `sorted_token_ids`/`expert_ids`. These adapters translate between the two formats.

**Functions added**:

- `pre_permute_deepep_normal_to_triton()` — registered as `@register_pre_permute("deepep_normal", "triton")`:
  1. Dequantizes FP8 hidden states if `hidden_states_scale` is present (group_size=128).
  2. Calls `ep_scatter` to rearrange tokens into expert-contiguous layout.
  3. Creates synthetic top-1 routing (`new_topk_ids = m_indices`, `new_topk_weights = ones`) since each scattered token maps to exactly one expert.
  4. Runs `moe_align_block_size` to prepare Triton kernel config.
  5. Stashes `output_index`, `topk_ids`, `topk_weights`, `hidden_states_shape` in `running_state` for the post-permute.

- `post_permute_triton_to_deepep_normal()` — registered as `@register_post_permute("triton", "deepep_normal")`:
  1. Calls `ep_gather` to reverse the scatter — weighted-sum expert outputs back into original token order using `output_index`, `topk_ids`, `topk_weights`.
  2. Returns `DeepEPNormalCombineInput`.

**Also changed**:

- `M == 0` early return in `TritonRunnerCore.run()` — DeepEP can dispatch zero tokens to some ranks. Without this, Triton kernels crash on empty inputs.
- Removed `assert not inplace` in the `no_combine` branch — was overly restrictive.

---

### 2b. `python/sglang/srt/layers/moe/fused_moe_triton/layer.py`

**What**: Two changes in `FusedMoE.__init__()`.

1. **`use_deep_gemm` gating** — changed from `use_deep_gemm=(self.moe_ep_size > 1)` to `use_deep_gemm=(self.moe_ep_size > 1 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM)`. On A100 where deep_gemm is not available, this prevents `UnquantizedFusedMoEMethod` from trying to use deep_gemm kernels.

2. **`no_combine = True` hack** — when using DeepEP without deep_gemm (i.e., A100 with Triton runner), sets `self.moe_runner_config.no_combine = True`. This tells the Triton runner to skip its internal token combination step, because DeepEP's combine phase handles cross-GPU aggregation separately via `ep_gather` in the post-permute adapter.

---

### 2c. `python/sglang/srt/layers/moe/ep_moe/kernels.py`

**What**: Three changes to the `ep_scatter` and `ep_gather` Triton kernels.

1. **Dynamic `BLOCK_D` selection** (both `ep_scatter` and `ep_gather`) — the original code hardcoded `BLOCK_D = 128`, but GPT-OSS has `hidden_size=2880` and `2880 % 128 ≠ 0`. Changed to pick the largest power-of-2 that divides `hidden_size`:
   ```python
   if hidden_size % 1024 == 0:
       BLOCK_D = 1024
   elif hidden_size % 128 == 0:
       BLOCK_D = 128
   elif hidden_size % 64 == 0:
       BLOCK_D = 64
   else:
       BLOCK_D = 32
   ```

2. **Empty-input guards** — added `if grid == 0: return` in `ep_scatter` and `if num_tokens == 0: return` in `ep_gather` to handle DeepEP dispatching zero tokens to some ranks.

3. **Better assert message** in `ep_gather` — includes tensor shape, hidden_size, and BLOCK_D in the assertion error.

---

### 2d. `python/sglang/srt/layers/moe/token_dispatcher/deepep.py`

**What**: Three changes in the DeepEP token dispatcher.

1. **RDMA bytes forced to 0** — `num_rdma_bytes = 0` instead of computing from config. A100 only supports intranode NVLink communication (no RDMA/InfiniBand for DeepEP). Allocating RDMA buffers wastes memory and can cause errors.

2. **`expert_alignment=128` always** — changed from `128 if ENABLE_JIT_DEEPGEMM else 1`. The Triton runner's `ep_scatter` kernel requires `BLOCK_E=128`-aligned expert token counts regardless of whether deep_gemm is used. Without alignment=128, the scatter kernel hits `assert m_indices.shape[0] % BLOCK_E == 0`.

3. **Triton runner re-enabled** — the non-deep_gemm/non-aiter/non-npu path for `deepep_normal` had `raise NotImplementedError()`. Changed to `output = hidden_states` (pass-through), which lets the registered Triton pre/post-permute adapters handle the computation.

---

### 2e. `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`

**What**: Two empty-input guards.

1. `if sorted_token_ids.numel() == 0: return` at the top of `invoke_fused_moe_kernel()`.
2. `if token_num == 0: return` at the top of `moe_sum_reduce_triton()`.

**Why**: DeepEP can dispatch zero tokens to some ranks. Without these guards, the Triton kernels launch with zero-size grids and crash.

---

### 2f. `python/sglang/srt/layers/moe/ep_moe/layer.py`

**What**: Three changes in `DeepEPMoE`.

1. **Constructor params** — added `gemm1_alpha`, `gemm1_clamp_limit`, and `**kwargs` to `DeepEPMoE.__init__()`. The GPT-OSS model passes these mxfp4-specific parameters, but the original `DeepEPMoE` didn't accept them, causing a `TypeError`.

2. **`deprecate_flag` logic** — added an early branch: `elif not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM: self.deprecate_flag = True`. On A100, `ENABLE_JIT_DEEPGEMM` is False because deep_gemm requires SM90+ (Hopper TMA instructions). With `deprecate_flag=True`, `DeepEPMoE.forward()` delegates to the generic `FusedMoE.forward()` pipeline, which uses the Triton runner with our deepep_normal adapters. Without this, mxfp4 quant would enter deep_gemm-specific code paths and crash.

3. **`run_moe_core` fallback** — changed the `deepep_normal` non-w4afp8 path from `assert False, "forward_deepgemm_contiguous is deprecated"` to `return super().run_moe_core(dispatch_output)`. This allows the generic FusedMoE pipeline (with Triton runner) to handle DeepEP dispatch output for non-w4afp8 quantization methods.

---

### 2g. `python/sglang/srt/layers/quantization/mxfp4.py`

**What**: Moved `topk_output = dispatch_output.topk_output` inside the `if self.use_flashinfer:` branch in `Mxfp4MoEMethod.apply()`.

**Why**: `DeepEPNormalDispatchOutput` does not have a `topk_output` attribute (it has `topk_ids` and `topk_weights` separately). The `topk_output` extraction was unconditionally executed before the flashinfer/triton branch, but only the flashinfer path uses it. The non-flashinfer path calls `self.runner.run(dispatch_output, quant_info)`, which passes the dispatch output directly to the runner (where the deepep_normal pre/post-permute adapters handle format conversion). Moving the extraction inside the flashinfer branch avoids the `AttributeError` on A100.

---

### 2h. `python/sglang/srt/models/gpt_oss.py`

**What**: Two changes in `GptOssSparseMoeBlock`.

1. **`forward_deepep()` method** — added a new method that handles the DeepEP code path. It's a simplified version of `forward_normal()` that omits the final `tensor_model_parallel_all_reduce` (because DeepEP's combine phase handles cross-GPU communication). It also supports profile-driven routing.

2. **`forward()` dispatch** — changed the DeepEP branch from `raise Exception("forward_deepep branch not implemented yet")` to `return self.forward_deepep(hidden_states, forward_batch)`.

---

## 3. Environment Setup

```bash
# Conda env
conda create -p /scratch1/yizhuoli/conda-envs/sglang-fp python=3.12 -y
conda activate /scratch1/yizhuoli/conda-envs/sglang-fp

# System modules
module load cuda/12.6.3 ucx/1.16.0 gdrcopy/2.5.1-cuda

# PyTorch
pip install torch==2.8.0+cu126 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# SGLang (dev mode from this fork)
pip install -e "python[all]"

# DeepEP (from source, A100 mode)
cd /scratch1/yizhuoli/DeepEP
DISABLE_SM90_FEATURES=1 pip install .

# Runtime env vars
export HF_HOME=/scratch1/yizhuoli/hf_cache
export TMPDIR=/scratch1/yizhuoli/tmp
```

---

## 4. Architecture Notes

### Data Flow (A100 DeepEP Normal Mode)

```
FusedMoE.forward()
  │
  ├─ dispatcher.dispatch()  →  DeepEPNormalDispatchOutput
  │    (DeepEP all-to-all: sends tokens to expert-owning ranks)
  │
  ├─ run_moe_core()
  │    ├─ pre_permute: deepep_normal → triton
  │    │    (ep_scatter: rearrange by expert, create sorted_token_ids)
  │    ├─ TritonRunnerCore.run()
  │    │    (fused expert computation: gate_up → activation → down)
  │    └─ post_permute: triton → deepep_normal
  │         (ep_gather: weighted-sum back to original token order)
  │
  └─ dispatcher.combine()
       (DeepEP all-to-all: sends results back to source ranks)
```

### Key Constraint: A100 vs H100

| Feature | H100 (SM90) | A100 (SM80) |
|---------|-------------|-------------|
| deep_gemm | ✅ | ❌ (needs TMA) |
| DeepEP internode | ✅ | ❌ (needs SM90 mbarrier) |
| DeepEP low-latency | ✅ | ❌ (needs TMA) |
| DeepEP normal (intranode) | ✅ | ✅ |
| MoE runner | deep_gemm or Triton | Triton only |
| RDMA buffers | Allocated | Forced to 0 |

### Why `no_combine = True`

In the standard (non-EP) Triton runner flow, the runner does top-k weighted combination internally. With DeepEP, this combination must happen **after** the runner (in `post_permute`), because:
- The runner sees synthetic top-1 routing (each scattered token → one expert).
- The real top-k combination happens in `ep_gather` using the original `topk_ids` and `topk_weights`.
- DeepEP's `combine()` then does the cross-rank all-to-all to return results.

Setting `no_combine = True` makes the runner output per-expert results without combining, which `ep_gather` then aggregates.

---

## 10. Profile-Driven Gating + DeepEP Batch Alignment Fix

**File**: `python/sglang/srt/models/gpt_oss.py` — `forward_deepep()` and `forward_normal()`

**Problem**: When profile-driven gating is enabled with DeepEP, the server crashes during warmup with:
```
RuntimeError: Assertion error 'is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks'
```

**Root cause**: `forward_batch.positions` has more elements than `hidden_states.shape[0]` during warmup (e.g. `positions=[8]` but `hidden_states=[2, 2880]`). The `ProfileDrivenRouter.route_gpu()` uses `positions` to index into the routing table, producing `topk_ids` with shape `[8, 4]` instead of `[2, 4]`. DeepEP's `get_dispatch_layout()` then builds `is_token_in_rank` with 8 rows, which fails the assertion against `x.size(0)=2`.

**Fix**: Align `req_pool_indices` and `positions` to `num_tokens` (from `hidden_states.shape[0]`) before calling `route_gpu()`:
- If `positions` is longer than `num_tokens`: truncate to `[:num_tokens]`
- If `req_pool_indices` has 1 entry (single request): expand to `num_tokens`
- If `req_pool_indices` is longer: truncate
- If shorter (multiple requests, fewer than tokens): cycle with modular indexing

Applied to both `forward_deepep()` and `forward_normal()` methods.

---

## 11. Load-Time mxfp4→bf16 Weight Dequantization (EP Path)

**File**: `python/sglang/srt/layers/quantization/mxfp4.py` — `Mxfp4FusedMoEMethod.process_weights_after_loading()`

**Finding**: No code change was needed. The existing code already handles load-time dequantization for the EP (generic Triton) path.

**How it works**: `process_weights_after_loading()` has three branches:

1. `if self.use_flashinfer:` — shuffles weights for TRT-LLM kernel (SM89+), returns early.
2. `if self.use_triton_kernels:` — swizzles for SM100 TRITON_KERNELS backend.
3. `else:` (line 574) — calls `upcast_from_mxfp()` from `triton_kernels.numerics_details.mxfp` to dequantize mxfp4 uint8 weights to bf16, then replaces the layer parameters with bf16 tensors and deletes the scales.

When running with EP, `moe_runner_backend` defaults to `"auto"` (`MoeRunnerBackend.AUTO`). Both `AUTO.is_triton_kernels()` and `AUTO.is_flashinfer_mxfp4()` return `False`, so the `else` branch executes — performing load-time dequantization automatically.

**Evidence**: After dequant, the Triton fused-MoE kernel reports `E=8,N=2880` (full intermediate size in bf16). Without dequant, it would show `N=1440` (packed uint8). Server starts successfully and serves requests.

**Runtime behavior**: After load-time dequant, the MoE experts run pure bf16 GEMM at inference time — no per-token dequantization overhead. The `TritonMoeQuantInfo` is created with bf16 weights and no scale tensors, so the Triton kernel treats them as standard dense bf16 matmul.

**Key detail**: `upcast_from_mxfp(tensor, scale, dtype=torch.bfloat16, axis=-1)` doubles the last dimension (unpacks 2 fp4 values per uint8 byte), so weight shapes change:
- `w13_weight`: `(num_local_experts, 2*intermediate, hidden//2)` → `(num_local_experts, 2*intermediate, hidden)`
- `w2_weight`: `(num_local_experts, hidden, intermediate//2)` → `(num_local_experts, hidden, intermediate)`
