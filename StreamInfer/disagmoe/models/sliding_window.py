"""Sliding-window decode attention."""
from typing import Tuple

import torch
import triton
import triton.language as tl
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_with_kvcache


@triton.jit
def _sw_prep_kernel(
    ctx_ptr, bt_ptr, bt_a_ptr, seq_a_ptr,
    bt_stride, n_pages,
    P: tl.constexpr, WIN: tl.constexpr, NB: tl.constexpr,
):
    """Part A's trimmed block table + cache_seqlens, one program per token."""
    row = tl.program_id(0)
    c = tl.load(ctx_ptr + row).to(tl.int32)
    start = tl.maximum(c + 1 - WIN, 0)
    b1 = (start + P - 1) // P
    j = tl.arange(0, NB)
    # clamp keeps padded entries pointing at a real page; flash reads only
    # ceil(seq_a / P) <= W/P + 1 of them
    idx = tl.minimum(b1 + j, n_pages - 1)
    tl.store(bt_a_ptr + row * NB + j, tl.load(bt_ptr + row * bt_stride + idx))
    tl.store(seq_a_ptr + row, c - b1 * P)


@triton.jit
def _sw_merge_kernel(
    q_ptr, kf_ptr, vf_ptr, ctx_ptr, bt_ptr,
    out_a_ptr, lse_a_ptr, out_ptr, lse_ptr,
    scale, bt_stride, q_stride_n, q_stride_h,
    H: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    G: tl.constexpr, KV: tl.constexpr, WIN: tl.constexpr,
):
    """Attend the boundary tokens and merge into Part A by log-sum-exp.

    One program per (token, head); K/V for the boundary are gathered from the
    paged cache inside the kernel.
    """
    row = tl.program_id(0)
    h = tl.program_id(1)
    kvh = h // G
    d = tl.arange(0, D)
    p = tl.arange(0, P)

    # q is a split() view of qkv, so its row stride is the full qkv width
    qv = tl.load(q_ptr + row * q_stride_n + h * q_stride_h + d).to(tl.float32)

    c = tl.load(ctx_ptr + row).to(tl.int32)
    start = tl.maximum(c + 1 - WIN, 0)
    off = start % P
    cnt = tl.where(off > 0, P - off, 0)          # 0 when the window starts aligned
    b0 = start // P
    page = tl.load(bt_ptr + row * bt_stride + b0).to(tl.int64)
    valid = p < cnt

    slots = page * P + tl.minimum(off + p, P - 1).to(tl.int64)
    addr = slots[:, None] * (KV * D) + kvh * D + d[None, :]
    ktile = tl.load(kf_ptr + addr, mask=valid[:, None], other=0.0).to(tl.float32)
    vtile = tl.load(vf_ptr + addr, mask=valid[:, None], other=0.0).to(tl.float32)

    s = tl.sum(qv[None, :] * ktile, axis=1) * scale
    s = tl.where(valid, s, float("-inf"))
    has = cnt > 0
    # guard the empty case: an all -inf row would make max/exp produce NaN
    mb = tl.where(has, tl.max(s, axis=0), 0.0)
    pe = tl.where(valid, tl.exp(s - mb), 0.0)
    tot = tl.sum(pe, axis=0)
    acc = tl.sum(pe[:, None] * vtile, axis=0)
    safe = tl.where(tot > 0, tot, 1.0)
    lse_b = tl.where(has, mb + tl.log(safe), float("-inf"))
    out_b = acc / safe

    lse_a = tl.load(lse_a_ptr + row * H + h).to(tl.float32)
    out_a = tl.load(out_a_ptr + row * H * D + h * D + d).to(tl.float32)
    m = tl.maximum(lse_a, lse_b)
    wa = tl.exp(lse_a - m)
    wb = tl.exp(lse_b - m)
    den = wa + wb
    tl.store(out_ptr + row * H * D + h * D + d,
             ((out_a * wa + out_b * wb) / den).to(tl.bfloat16))
    tl.store(lse_ptr + row * H + h, m + tl.log(den))


def sliding_window_decode_attention(
    q: torch.Tensor,                # [n, 1, H, D]
    k: torch.Tensor,                # [n, 1, KV, D]
    v: torch.Tensor,                # [n, 1, KV, D]
    kv_cache: torch.Tensor,         # [2, pages, P, KV, D]
    block_table: torch.Tensor,      # [n, max_pages] int32
    context_lens: torch.Tensor,     # [n] int32, PRE-append
    window: int,
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (out [n, 1, H, D], lse [n, H, 1]) for a W-token sliding window."""
    n, _, H, D = q.shape
    KV = k.shape[2]
    P = kv_cache.shape[2]
    G = H // KV
    dev = q.device
    NB = triton.next_power_of_2(window // P + 1)

    assert D == triton.next_power_of_2(D), f"head_dim {D} must be a power of 2"
    assert P == triton.next_power_of_2(P), f"page size {P} must be a power of 2"
    assert P <= window, f"page size {P} exceeds window {window}"
    assert kv_cache.dtype == torch.bfloat16, f"expected bf16 kv cache, got {kv_cache.dtype}"

    bt_a = torch.empty((n, NB), dtype=torch.int32, device=dev)
    seq_a = torch.empty((n,), dtype=torch.int32, device=dev)
    _sw_prep_kernel[(n,)](
        context_lens, block_table, bt_a, seq_a,
        block_table.stride(0), block_table.shape[1],
        P=P, WIN=window, NB=NB, num_warps=1, num_stages=1)

    out_a, lse_a = flash_attn_with_kvcache(
        q, kv_cache[0], kv_cache[1], k=k, v=v,
        cache_seqlens=seq_a, block_table=bt_a,
        softmax_scale=softmax_scale, causal=True, window_size=(-1, -1),
        return_softmax_lse=True)

    # view(), not reshape(): the merge kernel addresses these as a flat
    # [pages*P, KV, D] block, so a non-contiguous pool slice must fail here
    # rather than silently copy the whole KV cache per layer
    kf = kv_cache[0].view(-1, KV, D)
    vf = kv_cache[1].view(-1, KV, D)
    out = torch.empty_like(out_a)
    lse = torch.empty((n, H), dtype=torch.float32, device=dev)
    _sw_merge_kernel[(n, H)](
        q, kf, vf, context_lens, block_table,
        out_a, lse_a.reshape(n, H).contiguous(), out, lse,
        softmax_scale, block_table.stride(0), q.stride(0), q.stride(2),
        H=H, D=D, P=P, G=G, KV=KV, WIN=window,
        num_warps=2, num_stages=1)
    return out, lse.unsqueeze(-1)
