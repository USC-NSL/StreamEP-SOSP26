"""Profile-driven MoE routing for fake-prefill mode.

Loads pre-profiled token top-k routing outcomes from a Parquet file and
overwrites the normal gate topk routing decisions in MoE layers.  This
ensures expert routing remains realistic even when running decode-only
with fake prefill.

Compatible with DisagMoE profile format — shares the same Parquet files.

GPU Dense Lookup Design (CUDA graph compatible):
- At init: builds flat GPU tensors from parquet data
- At route time: pure GPU tensor ops (%, gather, index) — no CPU involvement
"""

from typing import Optional, Tuple

import logging
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_profile_router: Optional["ProfileDrivenRouter"] = None


def get_global_profile_router() -> Optional["ProfileDrivenRouter"]:
    """Return the global profile router, or None if not configured."""
    return _global_profile_router


def init_global_profile_router(
    profile_path: str, num_experts: int, top_k: int
) -> "ProfileDrivenRouter":
    """Load a profile and install it as the global router."""
    global _global_profile_router
    logger.info(
        f"Loading profile-driven routing from {profile_path} "
        f"(num_experts={num_experts}, top_k={top_k})"
    )
    with open(profile_path, "rb") as f:
        profile_bytes = f.read()
    _global_profile_router = ProfileDrivenRouter(
        profile_bytes=profile_bytes,
        num_experts_expected=num_experts,
        top_k=top_k,
    )
    logger.info(
        f"Profile-driven router loaded: "
        f"{_global_profile_router.num_profiled_requests} requests, "
        f"{_global_profile_router.num_layers + 1} layers, "
        f"{_global_profile_router.total_tokens_per_layer} tokens/layer"
    )
    return _global_profile_router


def maybe_init_global_profile_router(num_experts: int, top_k: int) -> None:
    """Initialize the global profile router if configured via server args.

    Safe to call multiple times — only the first call loads the profile.
    """
    global _global_profile_router
    if _global_profile_router is not None:
        return
    from sglang.srt.server_args import get_global_server_args

    server_args = get_global_server_args()
    profile_path = getattr(server_args, "profile_driven_gate_path", None)
    if not profile_path:
        return
    init_global_profile_router(profile_path, num_experts, top_k)


# ---------------------------------------------------------------------------
# ProfileDrivenRouter
# ---------------------------------------------------------------------------


class ProfileDrivenRouter:
    """Profile-driven simulated routing module (GPU dense lookup).

    Loads pre-profiled routing outcomes from a Parquet file and builds
    GPU-resident dense lookup tensors for O(1) expert selection.

    At route time, only GPU tensor operations are used (%, gather, index),
    making this fully compatible with CUDA graph capture.

    Data layout::

        routing_data       [num_layer_slots * total_tokens_per_layer, top_k]
        tokens_per_request [num_profiled_requests]
        token_prefix_sum   [num_profiled_requests + 1]

    Indexing scheme (all pure GPU ops)::

        mapped_rid = request_ids % num_profiled_requests
        mapped_tok = token_indices % tokens_per_request[mapped_rid]
        flat_idx   = layer_id * total_tokens_per_layer
                     + token_prefix_sum[mapped_rid] + mapped_tok
        topk_ids   = routing_data[flat_idx]

    Compatible with DisagMoE profile format.
    """

    # GPU tensors (set by _load_profile_from_bytes)
    routing_data: torch.Tensor  # [num_layer_slots * total_tokens_per_layer, top_k], int32
    tokens_per_request: torch.Tensor  # [num_profiled_requests], int64
    token_prefix_sum: torch.Tensor  # [num_profiled_requests + 1], int64

    def __init__(
        self,
        profile_bytes: bytes,
        num_experts_expected: int,
        top_k: int,
    ) -> None:
        self.num_experts = int(num_experts_expected)
        self.top_k = top_k
        if len(profile_bytes) == 0:
            raise ValueError(
                "ProfileDrivenRouter requires non-empty profile bytes at init"
            )
        self._load_profile_from_bytes(profile_bytes, top_k)

    # ------------------------------------------------------------------
    # Profile loading — builds GPU dense lookup tensors
    # ------------------------------------------------------------------

    def _load_profile_from_bytes(self, data: bytes, top_k: int) -> None:
        """Parse Parquet profile and build GPU dense lookup tensors.

        Expected columns:
            rid, token_index, layer, expert_logical_k0, expert_logical_k1, …

        Token indices within each request must be 0-based contiguous
        (0, 1, 2, …, N-1).
        """
        try:
            reader = pa.BufferReader(data)
            table = pq.read_table(reader)
        except Exception as e:
            raise RuntimeError(f"Failed to parse Parquet profile bytes: {e}")

        start_time = time.perf_counter()

        # --- required columns ---
        col_set = set(table.column_names)
        for required_col in ("rid", "token_index", "layer"):
            if required_col not in col_set:
                raise ValueError(
                    f"Required column {required_col} not found in profile"
                )

        # --- expert columns ---
        expert_columns = [
            c for c in table.column_names if c.startswith("expert_logical_k")
        ]
        if not expert_columns:
            raise ValueError("No expert columns found in profile")
        if len(expert_columns) != top_k:
            raise ValueError(
                f"Number of expert columns {len(expert_columns)} in the "
                f"profile does not match system's K = {top_k}"
            )

        # --- load to numpy ---
        use_cols = ["rid", "token_index", "layer"] + expert_columns
        df = table.select(use_cols).to_pandas(types_mapper=pd.ArrowDtype)

        rid_np = df["rid"].to_numpy(dtype=np.int64, copy=True)
        tok_np = df["token_index"].to_numpy(dtype=np.int64, copy=True)
        layer_np = df["layer"].to_numpy(dtype=np.int64, copy=True)
        expert_np = df[expert_columns].to_numpy(dtype=np.int32, copy=True)

        # --- expert projection (when profile has more experts than system) ---
        unique_experts = np.unique(expert_np)
        unique_experts = unique_experts[unique_experts >= 0]
        num_unique_experts = int(unique_experts.size)
        project_group_size: Optional[int] = None

        if num_unique_experts == self.num_experts:
            pass
        elif (
            num_unique_experts > self.num_experts
            and num_unique_experts % self.num_experts == 0
        ):
            project_group_size = num_unique_experts // self.num_experts
            print(
                f"\033[33m[ProfileDrivenRouter] Profile has "
                f"{num_unique_experts} experts; system expects "
                f"{self.num_experts}. Projecting by grouping "
                f"{project_group_size} profiled experts per system "
                f"expert.\033[0m"
            )
        else:
            raise ValueError(
                f"Profile contains {num_unique_experts} unique experts, "
                f"but system expects {self.num_experts}."
            )

        if project_group_size is not None:
            mask = expert_np >= 0
            expert_np[mask] = expert_np[mask] // int(project_group_size)

        # --- build contiguous rid mapping ---
        unique_rids = np.unique(rid_np)
        self.num_profiled_requests = int(unique_rids.size)
        # Vectorized mapping via searchsorted on sorted unique_rids
        contiguous_rid = np.searchsorted(unique_rids, rid_np).astype(np.int64)

        # --- tokens per request (unique token indices per rid) ---
        tokens_per_request_np = np.zeros(
            self.num_profiled_requests, dtype=np.int64
        )
        for i in range(self.num_profiled_requests):
            req_mask = contiguous_rid == i
            tokens_per_request_np[i] = int(np.unique(tok_np[req_mask]).size)

        # --- prefix sum for O(1) offset computation ---
        token_prefix_sum_np = np.zeros(
            self.num_profiled_requests + 1, dtype=np.int64
        )
        np.cumsum(tokens_per_request_np, out=token_prefix_sum_np[1:])
        self.total_tokens_per_layer = int(token_prefix_sum_np[-1])

        # --- layer count (0-indexed) ---
        self.num_layers = int(layer_np.max())
        num_layer_slots = self.num_layers + 1

        # --- build flat routing_data on CPU ---
        table_size = num_layer_slots * self.total_tokens_per_layer
        routing_data_np = np.zeros((table_size, top_k), dtype=np.int32)

        # Vectorized flat index: layer * total_tokens + prefix_sum[rid] + tok
        flat_indices = (
            layer_np * self.total_tokens_per_layer
            + token_prefix_sum_np[contiguous_rid]
            + tok_np
        )

        # Bounds check
        if np.any(flat_indices < 0) or np.any(flat_indices >= table_size):
            raise ValueError(
                "Flat index out of bounds — token indices must be 0-based "
                "contiguous per request in the profile."
            )

        routing_data_np[flat_indices] = expert_np

        # --- move to GPU ---
        self.routing_data = torch.tensor(
            routing_data_np, dtype=torch.int32, device="cuda"
        )
        self.tokens_per_request = torch.tensor(
            tokens_per_request_np, dtype=torch.int64, device="cuda"
        )
        self.token_prefix_sum = torch.tensor(
            token_prefix_sum_np, dtype=torch.int64, device="cuda"
        )

        gpu_bytes = (
            self.routing_data.element_size() * self.routing_data.nelement()
            + self.tokens_per_request.element_size()
            * self.tokens_per_request.nelement()
            + self.token_prefix_sum.element_size()
            * self.token_prefix_sum.nelement()
        )

        end_time = time.perf_counter()
        logger.info(
            f"Profile loaded in {end_time - start_time:.3f}s: "
            f"{self.num_profiled_requests} requests, "
            f"{self.total_tokens_per_layer} tokens/layer, "
            f"{num_layer_slots} layers, "
            f"routing_data {list(self.routing_data.shape)}, "
            f"GPU memory {gpu_bytes / (1024 * 1024):.1f} MB"
        )

    # ------------------------------------------------------------------
    # GPU-only routing (CUDA graph compatible)
    # ------------------------------------------------------------------

    def route_gpu(
        self,
        request_ids: torch.Tensor,
        token_indices: torch.Tensor,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Look up pre-profiled routing using pure GPU tensor ops.

        All operations (%, index, arithmetic) are GPU-only and fully
        compatible with CUDA graph capture.

        Args:
            request_ids: per-token request identifiers, GPU tensor.
                (e.g. ``forward_batch.req_pool_indices``)
            token_indices: per-token position indices, GPU tensor.
                (e.g. ``forward_batch.positions``)
            layer_id: decoder layer index (0-based Python int).

        Returns:
            topk_weights: ``[num_tokens, top_k]`` float32
            topk_ids:     ``[num_tokens, top_k]`` int32
        """
        # Map via modular arithmetic — pure GPU ops
        mapped_rid = request_ids.to(torch.int64) % self.num_profiled_requests
        mapped_tok = (
            token_indices.to(torch.int64)
            % self.tokens_per_request[mapped_rid]
        )

        # Flat index into routing_data — pure GPU ops
        flat_idx = (
            layer_id * self.total_tokens_per_layer
            + self.token_prefix_sum[mapped_rid]
            + mapped_tok
        )

        # Gather expert ids — pure GPU indexing
        topk_ids = self.routing_data[flat_idx]  # [num_tokens, top_k] int32

        # Uniform weights — pure GPU allocation
        num_tokens = topk_ids.shape[0]
        if self.top_k == 1:
            topk_weights = torch.ones(
                (num_tokens, 1), device=topk_ids.device, dtype=torch.float32
            )
        else:
            topk_weights = torch.full(
                (num_tokens, self.top_k),
                1.0 / float(self.top_k),
                device=topk_ids.device,
                dtype=torch.float32,
            )

        return topk_weights, topk_ids


__all__ = [
    "ProfileDrivenRouter",
    "get_global_profile_router",
    "init_global_profile_router",
    "maybe_init_global_profile_router",
]
