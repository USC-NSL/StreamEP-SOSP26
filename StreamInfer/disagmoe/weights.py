"""M-WL: centralized real-weight loading for every StreamInfer component.

ONE component owns all checkpoint weights. Modules keep their existing randn
creation ("dummy mode" = this loader never runs, i.e. model_config.weights_dir
is None); when a weights dir is configured the loader overwrites every parameter
in place after construction. No real/dummy branching lives in model code.

Expected directory layout (produced by experiment_utils/batch-decode/convert_gptoss_weights.py
stage "shard"):
  attn_shared.safetensors      HF names; qkv fused as
                               model.layers.{L}.self_attn.qkv.{weight,bias};
                               plus embedding / final norm / lm_head
  experts_rank{R}.safetensors  layers.{L}.{w13_weight,w13_bias,w2_weight,w2_bias}
                               sliced to that rank's experts
  expert_map.json              {"rank": [global expert ids]} — the loader picks
                               the shard whose id set covers this worker's
                               local experts, so any placement the converter
                               was cut for keeps working
"""

import json
import os
from typing import Dict, List, Optional

import torch
from safetensors import safe_open

from disagmoe.utils.logger import get_logger


def _log(msg: str):
    logger = get_logger()
    if logger is not None:
        logger.info(msg)
    else:
        print(msg, flush=True)


def _copy(param: torch.nn.Parameter, tensor: torch.Tensor, what: str):
    assert param is not None, f"{what}: module has no such parameter (config mismatch?)"
    assert param.shape == tensor.shape, f"{what}: shape {tuple(param.shape)} != checkpoint {tuple(tensor.shape)}"
    with torch.no_grad():
        param.copy_(tensor.to(param.dtype))


def load_attn_weights(operators: List[torch.nn.Module], layer_ids: List[int],
                      weights_dir: str) -> None:
    """Overwrite every attention-side parameter from attn_shared.safetensors.

    operators[i] serves global layer layer_ids[i]. Covers qkv (+bias), o_proj
    (+bias), sinks, both norms, and the router gate (+bias).
    """
    path = os.path.join(weights_dir, "attn_shared.safetensors")
    with safe_open(path, framework="pt", device="cpu") as f:
        for i, op in enumerate(operators):
            L = layer_ids[i]
            p = f"model.layers.{L}"
            _copy(op.qkv_proj.weight, f.get_tensor(f"{p}.self_attn.qkv.weight"), f"L{L} qkv.weight")
            if getattr(op.qkv_proj, "bias", None) is not None:
                _copy(op.qkv_proj.bias, f.get_tensor(f"{p}.self_attn.qkv.bias"), f"L{L} qkv.bias")
            _copy(op.o_proj.weight, f.get_tensor(f"{p}.self_attn.o_proj.weight"), f"L{L} o_proj.weight")
            if getattr(op.o_proj, "bias", None) is not None:
                _copy(op.o_proj.bias, f.get_tensor(f"{p}.self_attn.o_proj.bias"), f"L{L} o_proj.bias")
            if op.sinks is not None:
                _copy(op.sinks, f.get_tensor(f"{p}.self_attn.sinks"), f"L{L} sinks")
            _copy(op.pre_attention_layernorm.weight,
                  f.get_tensor(f"{p}.input_layernorm.weight"), f"L{L} input_layernorm")
            _copy(op.post_attention_layernorm.weight,
                  f.get_tensor(f"{p}.post_attention_layernorm.weight"), f"L{L} post_attention_layernorm")
            _copy(op.gate.weight, f.get_tensor(f"{p}.mlp.router.weight"), f"L{L} router.weight")
            if getattr(op.gate, "bias", None) is not None:
                _copy(op.gate.bias, f.get_tensor(f"{p}.mlp.router.bias"), f"L{L} router.bias")
    _log(f"[M-WL] loaded attention weights for layers {layer_ids} from {path}")


def _find_expert_shard(weights_dir: str, local_expert_ids: List[int]):
    """Pick the shard file whose expert set covers this worker's experts and
    return (path, {global expert id -> row within the shard})."""
    with open(os.path.join(weights_dir, "expert_map.json")) as f:
        emap = {int(k): v for k, v in json.load(f).items()}
    want = set(local_expert_ids)
    for rank, ids in sorted(emap.items()):
        if want.issubset(set(ids)):
            row_of = {eid: row for row, eid in enumerate(ids)}
            return os.path.join(weights_dir, f"experts_rank{rank}.safetensors"), row_of
    raise RuntimeError(
        f"no expert shard in {weights_dir} covers local experts {sorted(want)}; "
        f"re-cut shards with convert_gptoss_weights.py (experiment_utils/batch-decode) --stage shard "
        f"--expert-map matching this placement")


def load_expert_weights(operators: List[torch.nn.Module], layer_ids: List[int],
                        local_expert_ids: List[int], weights_dir: str) -> None:
    """Overwrite expert weights (+ biases when present) for this worker's
    local experts, in local-expert order (row j of a parameter belongs to
    local_expert_ids[j], matching the dispatcher's local ordering)."""
    path, row_of = _find_expert_shard(weights_dir, local_expert_ids)
    # explicit cpu: engines set cuda as torch default device, but the shard
    # tensors are read on cpu
    rows = torch.tensor([row_of[e] for e in local_expert_ids],
                        dtype=torch.long, device="cpu")
    with safe_open(path, framework="pt", device="cpu") as f:
        for i, op in enumerate(operators):
            L = layer_ids[i]
            for name in ("w13_weight", "w2_weight", "w13_bias", "w2_bias"):
                param = getattr(op, name, None)
                if param is None:
                    continue
                tensor = f.get_tensor(f"layers.{L}.{name}").index_select(0, rows)
                _copy(param, tensor, f"L{L} {name}")
    _log(f"[M-WL] loaded expert weights for layers {layer_ids}, "
         f"experts {local_expert_ids} from {path}")


def load_sampler_tensors(weights_dir: str, device: str = "cuda",
                         dtype: torch.dtype = torch.bfloat16) -> Dict[str, torch.Tensor]:
    """Embedding, final norm, and lm_head for the sampler stack (M-SMP)."""
    path = os.path.join(weights_dir, "attn_shared.safetensors")
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        out["embedding"] = f.get_tensor("model.embed_tokens.weight").to(device=device, dtype=dtype)
        out["final_norm"] = f.get_tensor("model.norm.weight").to(device=device, dtype=dtype)
        out["lm_head"] = f.get_tensor("lm_head.weight").to(device=device, dtype=dtype)
    return out
