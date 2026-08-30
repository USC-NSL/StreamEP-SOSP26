from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from vllm.attention import AttentionMetadata
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_with_kvcache
from disagmoe.models.sliding_window import sliding_window_decode_attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.config import CacheConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from disagmoe.models.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from disagmoe.models.gate import ProfileDrivenRouter
from disagmoe.ops.memory import permute_tokens_cuda
from disagmoe.models.experts import SharedExpertMLP

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope

from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk)

import triton
import triton.language as tl
import os

# debug-only: set by engine.preprocess_batch_attn when DMOE_LAYER_DUMP is on,
# so layer dumps can attribute rows to requests
_DBG_REQ_IDS = None

@triton.jit
def compute_seg_indptr_triton_kernel(reorder_topk_ids, seg_indptr, num_toks):
    expert = tl.program_id(0)
    low = 0
    high = num_toks - 1
    target_location = -1
    while low <= high:
        mid = (low + high) // 2

        if tl.load(reorder_topk_ids + mid) > expert:
            high = mid - 1
        else:
            low = mid + 1
            target_location = mid
    tl.store(seg_indptr + expert + 1, target_location + 1)

@triton.jit
def compute_src2dst_triton_kernel(
    reorder_ids, src2dst, num_toks, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(reorder_ids + dst_id, mask=mask)
    tl.store(src2dst + src_id, dst_id, mask=mask)
    
@torch.compile
def multinomial_no_replacement(probs, num_samples):
    """
    probs: [batch_size, num_classes] or [num_classes] (will be normalized)
    num_samples: number of samples to draw (must be <= num_classes)
    Returns: [batch_size, num_samples] or [num_samples]
    """
    # Normalize and reshape to [batch_size, num_classes]
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    batch_size, num_classes = probs.shape
    assert num_samples <= num_classes, "Can't sample more than population"
    
    # Gumbel trick: logp + Uniform noise
    gumbel_noise = -torch.empty_like(probs).exponential_().log()  # ~Gumbel(0,1)
    noisy_logits = torch.log(probs) + gumbel_noise
    
    # Top-k selection (equivalent to sampling without replacement)
    _, samples = torch.topk(noisy_logits, num_samples, dim=-1)
    
    return samples

class MoEAttention(nn.Module):

    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        num_experts: int,
        top_k: int = 1,
        tp_size: int = 1,
        tp_rank: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        rms_norm_eps: float = 1e-6,
        sliding_window: Optional[int] = None,
        use_sinks: bool = False,
        attention_bias: bool = False,
        router_norm: str = "softmax_topk_renorm",
        router_bias: bool = False,
        router_mode: str = "model",
        first_layer: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        quant_config_qkv: Optional[QuantizationConfig] = None,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
        gate_profile_bytes: Optional[bytes] = None,
        num_shared_experts: int = 0,
        shared_expert_intermediate_size: Optional[int] = None,
        quant_config_shared: Optional[QuantizationConfig] = None,
        intermediate_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_experts = num_experts
        self.top_k = top_k
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        # flash-attn convention: (left, 0) attends to `left` previous tokens
        # plus the current one, so a model window of W tokens (incl. current)
        # maps to left = W - 1
        self.sliding_window = sliding_window
        self.router_norm = router_norm
        self.router_mode = router_mode
        # F1 residual stream: the executor assigns one shared stash tensor
        # [max_running_reqs + 1, hidden] to every layer after construction.
        # Layer 0 starts a fresh stream (its input is an embedding, not an
        # expert sum), so it only writes the stash; all other layers read too.
        self.residual_stash: Optional[torch.Tensor] = None
        self.stash_read = not first_layer

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # NOTE(shaoyuw): must invoke initialize_model_parallel
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            tp_size=tp_size,
            bias=attention_bias,
            quant_config=quant_config_qkv,
            prefix=f"{prefix}.qkv_proj",
            params_dtype=params_dtype,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config_qkv,
            prefix=f"{prefix}.o_proj",
            params_dtype=params_dtype,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )
        if use_sinks:
            # one learned sink logit per (local) head; loaded by the weight
            # loader in real mode, randn otherwise like every other parameter
            self.sinks = nn.Parameter(
                torch.randn(self.num_heads, dtype=params_dtype),
                requires_grad=False,
            )
        else:
            self.sinks = None

        self.gate = ReplicatedLinear(hidden_size,
                                     num_experts,
                                     bias=router_bias,
                                     params_dtype=params_dtype,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")
        
        if gate_profile_bytes is not None and len(gate_profile_bytes) > 0:
            self.profile_driven_router = ProfileDrivenRouter(gate_profile_bytes, num_experts, top_k, layer_id=layer_id)
        else:
            self.profile_driven_router = None
        
        self.pre_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        routing_trace_file_path = os.environ.get("DMOE_WEIGHTED_ROUTER_FILE")
        if routing_trace_file_path is not None and routing_trace_file_path != "":
            category = os.environ.get("DMOE_WEIGHTED_ROUTER_CATEGORY", "closed_qa")
            assert os.path.exists(routing_trace_file_path), f"Weighted router file {routing_trace_file_path} does not exist."
            import pandas as pd
            df = pd.read_csv(routing_trace_file_path)
            df = df[df['category'] == category]
            df = df[df['layer_id'] == layer_id]
            df = df.iloc[:, list(range(-8, 0))]
            
            data = torch.tensor(df.values, dtype=torch.int32).sum(dim=0)
            self.weighted_router = data / data.sum(dim=-1, keepdim=True)
        else:
            self.weighted_router = None

        # Shared experts: process ALL tokens, no routing
        self.num_shared_experts = num_shared_experts
        if num_shared_experts > 0:
            se_intermediate = shared_expert_intermediate_size if shared_expert_intermediate_size is not None else (intermediate_size if intermediate_size is not None else hidden_size)
            self.shared_experts = nn.ModuleList([
                SharedExpertMLP(
                    hidden_size=hidden_size,
                    intermediate_size=se_intermediate,
                    params_dtype=params_dtype,
                    quant_config=quant_config_shared,
                    prefix=f"{prefix}.shared_experts.{i}",
                )
                for i in range(num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def _random_routing_with_weights(self, router_logits: torch.Tensor) -> torch.Tensor:
        num_tokens = router_logits.shape[0]
        weights = self.weighted_router.expand(num_tokens, -1)
        topk_ids = multinomial_no_replacement(weights, self.top_k)
        sampled_weights = torch.gather(weights, 1, topk_ids)
        topk_weights = sampled_weights / sampled_weights.sum(dim=1, keepdim=True)
        return topk_weights, topk_ids

    def permute_by_exp_ids(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Permute the attention output by expert IDs.
        """
        _, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)

        permuted_output = permute_tokens_cuda(hidden_states, reorder_ids)
        
        return permuted_output, reorder_ids
    
    def _paged_decode_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Decode-shaped paged attention with in-kernel KV append.

        kv_cache is this layer's pool slice [2, blocks, page, kv_heads, dim].
        cache_seqlens must be the PRE-append lengths (= context_lens): the
        kernel writes the new k/v at that offset inside the pages given by
        block_table, then attends over context+1 tokens. Sliding window and
        attention sinks (exact: one extra softmax logit per head, applied as
        out * sigmoid(lse - sink)) are handled here.
        """
        n = q.shape[0]
        q = q.view(n, 1, self.num_heads, self.head_dim)
        k = k.view(n, 1, self.num_kv_heads, self.head_dim)
        v = v.view(n, 1, self.num_kv_heads, self.head_dim)
        if self.sliding_window:
            out, lse = sliding_window_decode_attention(
                q, k, v, kv_cache,
                attn_metadata.block_tables,
                attn_metadata.context_lens_tensor,
                self.sliding_window, self.scaling,
            )
        else:
            out, lse = flash_attn_with_kvcache(
                q,
                kv_cache[0],
                kv_cache[1],
                k=k,
                v=v,
                cache_seqlens=attn_metadata.context_lens_tensor,
                block_table=attn_metadata.block_tables,
                softmax_scale=self.scaling,
                causal=True,
                window_size=(-1, -1),
                return_softmax_lse=True,
            )
        out = out.view(n, self.num_heads, self.head_dim)
        if self.sinks is not None:
            rescale = torch.sigmoid(
                lse.view(n, self.num_heads, 1).float()
                - self.sinks.view(1, self.num_heads, 1).float()
            )
            out = out * rescale.to(out.dtype)
        return out.view(n, self.num_heads * self.head_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor = None,
        request_ids: Optional[torch.Tensor] = None,
        stash_slots: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # F1: hidden_states arriving from the expert side is the bare
        # Σ wₖ·FFNₖ(...); adding the residual saved by the previous layer
        # completes x_l = residual_{l-1} + expert_sum. Layer 0's input is an
        # embedding — a fresh stream — so it skips the read.
        if self.stash_read and self.residual_stash is not None and stash_slots is not None:
            hidden_states = hidden_states + self.residual_stash.index_select(0, stash_slots)

        if residual is None:
            residual = hidden_states
            hidden_states = self.pre_attention_layernorm(hidden_states)
        else:
            hidden_states, residual = self.pre_attention_layernorm(hidden_states, residual)

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self._paged_decode_attention(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        output, residual = self.post_attention_layernorm(output, residual)

        import os as _os
        _dbg = _os.environ.get("DMOE_LAYER_DUMP", "")
        _dbg_layers = {int(x) for x in _os.environ.get("DMOE_LAYER_DUMP_LAYERS", "0").split(",")}
        if _dbg and self.layer_id in _dbg_layers:
            _os.makedirs(_dbg, exist_ok=True)
            self._dbg_step = getattr(self, "_dbg_step", 0)
            torch.save({
                "req_ids": list(_DBG_REQ_IDS) if _DBG_REQ_IDS is not None else None,
                "layer_id": self.layer_id,
                "hidden_in": hidden_states.detach().cpu(),
                "positions": positions.detach().cpu(),
                "context_lens": attn_metadata.context_lens_tensor.detach().cpu(),
                "block_row0": attn_metadata.block_tables[0].detach().cpu(),
                "q": q.detach().cpu(), "k": k.detach().cpu(), "v": v.detach().cpu(),
                "attn_out": attn_output.detach().cpu(),
                "norm2_out": output.detach().cpu(),
                "residual": residual.detach().cpu(),
            }, f"{_dbg}/l{self.layer_id}_step{self._dbg_step:04d}_"
               f"pid{_os.getpid()}.pt")
            self._dbg_step += 1

        # residual = x_l + attn_out: the next layer (or the sampler, after the
        # final layer) completes the stream with this layer's expert sum
        if self.residual_stash is not None and stash_slots is not None:
            self.residual_stash.index_copy_(0, stash_slots, residual)

        # Shared experts: process all tokens and add to output
        if self.shared_experts is not None:
            shared_output = torch.zeros_like(output)
            for shared_expert in self.shared_experts:
                shared_output = shared_output + shared_expert(output)
            output = output + shared_output

        router_logits, _ = self.gate(output)

        if self.profile_driven_router is not None:
            assert request_ids is not None, "Profile-driven routing requires request_ids"
            topk_weights, topk_ids = self.profile_driven_router.route(
                request_ids=request_ids,
                token_indices=positions,
                layer_id=self.layer_id,
                top_k=self.top_k,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        elif self.weighted_router is not None:
            topk_weights, topk_ids = self._random_routing_with_weights(router_logits)
        else:
            # F8: computed router logits are the default; random routing is the
            # explicit dummy-experiment opt-in (the pre-F8 default behavior)
            if self.router_mode == "random":
                router_logits = torch.rand_like(router_logits)
            if self.router_norm == "topk_softmax":
                # gpt-oss: top-k over raw logits, softmax over the selected k
                topk_vals, topk_ids64 = torch.topk(router_logits, self.top_k, dim=-1)
                topk_weights = torch.softmax(topk_vals.float(), dim=-1)
                topk_ids = topk_ids64.to(torch.int32)
            else:
                # mixtral-style: softmax over all, top-k, renormalize
                topk_weights, topk_ids = fused_topk(hidden_states=hidden_states,
                                        gating_output=router_logits,
                                        topk=self.top_k,
                                        renormalize=True)

        return output, topk_weights, topk_ids