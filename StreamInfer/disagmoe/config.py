from math import exp
from dataclasses import dataclass
from typing import Optional, List

import vllm
import torch
import vllm.config

@dataclass
class ModelConfig:
    hidden_size: int
    num_layers: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    num_experts: int
    intermediate_size: int
    dtype: torch.dtype
    ep_size: int = 1 # default to 1
    tp_size: int = 1
    dp_size: int = 1
    rank: int = 0
    layer_ids: Optional[List[int]] = None
    top_k: int = 1
    max_seq_len: int = 4096

    # Attention-specific quantization option for QKV projection
    # e.g., "fp8" or None
    attn_qkv_quant: Optional[str] = None
    # MoE experts linear quantization option (applies to MoEExpertsSerial only)
    # e.g., "fp8" or None
    moe_linear_quant: Optional[str] = None

    # Shared expert configuration
    num_shared_experts: int = 0
    shared_expert_intermediate_size: Optional[int] = None

    # ---- model-fidelity fields (all config-driven; defaults preserve the
    # ---- pre-existing behavior for configs that don't set them) ----
    max_position: int = 4096 * 32
    rope_theta: float = 10000.0
    # vllm get_rope rope_scaling dict (e.g. YaRN); None = plain RoPE
    rope_scaling: Optional[dict] = None
    rms_norm_eps: float = 1e-6
    # per-layer "sliding_attention"/"full_attention"; None = all full
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None  # tokens attended incl. current (gpt-oss 128)
    attention_bias: bool = False          # qkv/o projection bias
    attention_sinks: bool = False         # learned per-head sink logits
    # router: "softmax_topk_renorm" (mixtral-style, existing behavior) or
    # "topk_softmax" (gpt-oss: top-k over raw logits, softmax over the k)
    router_norm: str = "softmax_topk_renorm"
    router_bias: bool = False
    # "model" = computed logits (default), "random" = rand_like override for
    # dummy-mode experiments (the pre-F8 default), "profile"/"weighted" keep
    # their existing dedicated objects
    router_mode: str = "model"
    # experts: plain SwiGLU when swiglu_limit is None; gpt-oss variant
    # ((up+1)*gate*sigmoid(alpha*gate) with clamping) when set
    swiglu_limit: Optional[float] = None
    swiglu_alpha: float = 1.702
    expert_bias: bool = False
    # tokenizer/sampler facts (consumed by M-SMP/M-TOK)
    vocab_size: Optional[int] = None
    eos_token_ids: Optional[List[int]] = None
    # M-WL: directory holding per-rank shards + attn_shared; None = dummy weights
    weights_dir: Optional[str] = None
    # M-SMP: "dummy" (length-counting sampler, rand loopback) or "model"
    # (real final-norm + lm_head + greedy + embedding loopback)
    sampler: str = "dummy"
    # M-KVL: "none" (no KV ingestion; teacher-forced warm start under the real
    # sampler) or "mooncake" (prompt KV loaded at admission; token_ids must be
    # prompt + [first generated token], init_prefill_len = len(prompt))
    kv_source: str = "none"
    mooncake_master: Optional[str] = None
    kv_model_tag: str = "gptoss120b"

    @property
    def num_experts_per_rank(self):
        return self.num_experts // self.ep_size

    def sliding_window_of(self, layer_id: int) -> Optional[int]:
        """This layer's sliding window in tokens (incl. current), or None."""
        if self.layer_types is None or self.sliding_window is None:
            return None
        if self.layer_types[layer_id] == "sliding_attention":
            return self.sliding_window
        return None
    
@dataclass
class EngineConfig:
    # Unified (colocate) scheduler configuration.
    unified_scheduler_type: str
    defrag_weight_decay: float
    defrag_lookahead_steps: int
    defrag_lookback_steps: int

    enable_cuda_graph_attn: bool = False
    enable_cuda_graph_expert: bool = False
    enable_grouped_gemm: bool = False
    less_than_sm90: bool = False
    
    max_batch_size_attn: int = 160
    max_batch_size_expert: int = 512
    max_attn_graph_bsz: int = 160
    max_pending_sends: int = 16
    
    # FIXME(hogura|20250110): temporary field, should be moved to other place
    enable_trace: bool = False

    enable_advanced_logging: bool = False
    advanced_logging_dir: str = "./advanced_logs"
    advanced_logging_sample_rate: float = 0.1
    
@dataclass
class CacheConfig(vllm.config.CacheConfig):
    
    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: float,
        cache_dtype: str,
        num_gpu_blocks_override: Optional[int] = None,
        sliding_window: Optional[int] = None,
        enable_prefix_caching: bool = False,
        cpu_offload_gb: float = 0,
    ) -> None:
        super().__init__(block_size, gpu_memory_utilization, 
                         swap_space, cache_dtype, num_gpu_blocks_override, 
                         sliding_window, enable_prefix_caching, cpu_offload_gb)

mixtral_config = ModelConfig(
    hidden_size = 4096,
    num_layers = 32,
    head_dim = 128,
    num_heads = 32,
    num_kv_heads = 8,
    num_experts = 8,
    intermediate_size = 14336,
    dtype = torch.bfloat16,
    ep_size = 8,
    top_k = 2,
)

duo_expert_mixtral = ModelConfig(
    hidden_size = 4096,
    num_layers = 32,
    head_dim = 128,
    num_heads = 32,
    num_kv_heads = 8,
    num_experts = 2,
    intermediate_size = 14336,
    dtype = torch.bfloat16,
    ep_size = 2,
)

qwen3_235b_config = ModelConfig(
    hidden_size = 4096,
    num_layers = 94,
    head_dim = 128,
    num_heads = 64,
    num_kv_heads = 4,
    num_experts = 128,
    intermediate_size = 1536,
    dtype = torch.bfloat16,
    top_k = 8,
)

qwen3_30b_config = ModelConfig(
    hidden_size = 2048,
    num_layers = 48,
    head_dim = 128,
    num_heads = 32,
    num_kv_heads = 4,
    num_experts = 128,
    intermediate_size = 768,
    dtype = torch.bfloat16,
    top_k = 8,
)

# Real gpt-oss-120b values, verified against the checkpoint's config.json
# (/mnt/nvme14t/models/gpt-oss-120b): 36 layers alternating sliding(128)/full
# starting with sliding at layer 0; YaRN rope; qkv/o bias; sinks; router bias;
# clamped (up+1)*gate*sigmoid(1.702*gate) expert activation with expert biases.
gptoss_120b_config = ModelConfig(
    hidden_size = 2880,
    num_layers = 36,
    head_dim = 64,
    num_heads = 64, # original 36
    num_kv_heads = 8, # original 8
    num_experts = 128,
    intermediate_size = 2880,
    dtype = torch.bfloat16,
    top_k = 4,
    max_position = 131072,
    rope_theta = 150000.0,
    rope_scaling = {
        "rope_type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    },
    rms_norm_eps = 1e-5,
    layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention"
                   for i in range(36)],
    sliding_window = 128,
    attention_bias = True,
    attention_sinks = True,
    router_norm = "topk_softmax",
    router_bias = True,
    swiglu_limit = 7.0,
    swiglu_alpha = 1.702,
    expert_bias = True,
    vocab_size = 201088,
    eos_token_ids = [200002, 199999, 200012],
)

glm45air_106b_config = ModelConfig(
    hidden_size = 4096,
    num_layers = 45,
    head_dim = 128,
    num_heads = 96,
    num_kv_heads = 8,
    num_experts = 128,
    intermediate_size = 1408,
    dtype = torch.bfloat16,
    top_k = 8,
)

glm45air_half_config = ModelConfig(
    hidden_size = 4096,
    num_layers = 23,
    head_dim = 128,
    num_heads = 96,
    num_kv_heads = 8,
    num_experts = 128,
    intermediate_size = 1408,
    dtype = torch.bfloat16,
    top_k = 8,
)