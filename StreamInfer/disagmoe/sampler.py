"""M-SMP: sampler stack for real-token serving.

Both samplers expose one interface consumed at the engine's two seams
(`recv_new_request` admission and the final-pseudo-layer `sample_results`):

  create_request(req_id, max_output_len, token_ids)
  admit_data(req_id, init_prefill_len, token_ids) -> (data [1,H] | None, prefill_len)
      None data means "engine keeps its existing behavior" (dummy: torch.rand).
  step(completed_hidden [n,H], req_ids) ->
      (continue_ids, finish_req_ids, out_token_ids, continued_data | None)
      completed_hidden is the FINISHED residual stream (expert sum + stash row).
      None continued_data means "loop back completed_hidden[continue_ids]".

RealSampler implements the real final stage: RMSNorm -> lm_head -> greedy argmax
-> eos/max-len termination -> loopback data = embedding(next_token). Prompt
tokens beyond the first are TEACHER-FORCED through the decode loop (`--kv-source
none` oracle mode): while a request's forced queue is non-empty, the "sampled"
token is the next prompt token (not counted against max_output_len), which
builds correct KV step by step without any prefill. With a KV source (M3),
admission uses the manifest's first generated token and no forcing happens.

Everything here is eager, off-graph, on the engine's stream (plan C4).
"""

from typing import Dict, List, Optional, Tuple

import torch

from disagmoe.weights import load_sampler_tensors


class RealSampler:

    def __init__(self, weights_dir: str, eos_token_ids: List[int],
                 rms_norm_eps: float, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16,
                 kv_source: str = "none"):
        t = load_sampler_tensors(weights_dir, device=device, dtype=dtype)
        self.embedding = t["embedding"]        # [vocab, H]
        self.final_norm_w = t["final_norm"].float()
        self.lm_head = t["lm_head"]            # [vocab, H]
        self.eos_token_ids = set(eos_token_ids or [])
        self.eps = rms_norm_eps
        self.kv_source = kv_source
        self.forced: Dict[int, List[int]] = {}
        self.out_len: Dict[int, int] = {}
        self.req_max_output_len: Dict[int, int] = {}
        self.last_token: Dict[int, int] = {}

    def create_request(self, req_id: int, max_output_len: int,
                       token_ids: Optional[List[int]] = None):
        assert token_ids, "RealSampler needs the prompt token ids (M-REQ)"
        self.req_max_output_len[req_id] = max_output_len
        self.out_len[req_id] = 0
        self.forced[req_id] = list(token_ids[1:])

    def admit_data(self, req_id: int, init_prefill_len: int,
                   token_ids: Optional[List[int]]) -> Tuple[Optional[torch.Tensor], int]:
        # init_prefill_len semantics in the C++ contract: number of KV slots
        # that already hold VALID context before the current token.
        if self.kv_source == "mooncake":
            # Prompt KV for the first L (page-aligned) positions is loaded at
            # admission; token_ids = prompt + [first generated token]. The
            # store holds full 16-token pages only, so the prompt tail plus
            # the first token replay through the forced queue: each forced
            # step computes that position's KV in place, then sampling takes
            # over. Forced tokens occupy KV slots (they are part of the
            # request's output_len block budget) but are not generated
            # output, so they must not consume the generation budget.
            L = init_prefill_len
            assert len(token_ids) >= L + 1, \
                "kv_source=mooncake needs token_ids = prompt + [first token]"
            self.forced[req_id] = list(token_ids[L + 1:])
            if self.forced[req_id]:
                self.req_max_output_len[req_id] = max(
                    1, self.req_max_output_len[req_id] - len(self.forced[req_id]))
            first = token_ids[L]
            self.last_token[req_id] = first
            return self.embedding[first].view(1, -1).clone(), L
        # Teacher-forced warm start: NO pre-existing KV context — the first
        # prompt token IS the current token, entering at position 0. Returning
        # 1 here allocated a garbage context slot and shifted every RoPE
        # position by one (found via the M2 step-0 bisection). 0 is safe: the
        # finish-signal encoding of 0 is dormant in the live engine path, and
        # BlockManager::allocate(rid, 0) just pre-registers the block list.
        first = token_ids[0]
        self.last_token[req_id] = first
        return self.embedding[first].view(1, -1).clone(), 0

    def clean_request(self, req_id: int):
        for d in (self.forced, self.out_len, self.req_max_output_len, self.last_token):
            d.pop(req_id, None)

    @torch.inference_mode()
    def step(self, completed_hidden: torch.Tensor, req_ids: List[int]):
        n = completed_hidden.shape[0]
        h = completed_hidden.float()
        var = h.pow(2).mean(-1, keepdim=True)
        normed = (h * torch.rsqrt(var + self.eps) * self.final_norm_w).to(self.lm_head.dtype)
        sampled = torch.argmax(normed @ self.lm_head.t(), dim=-1).tolist()

        continue_ids, finish_req_ids, out_tokens = [], [], []
        for i, rid in enumerate(req_ids):
            forced_q = self.forced.get(rid)
            if forced_q:
                tok = forced_q.pop(0)          # teacher-forced prompt token
                is_finished = False
            else:
                tok = int(sampled[i])
                self.out_len[rid] += 1
                is_finished = (tok in self.eos_token_ids
                               or self.out_len[rid] >= self.req_max_output_len[rid])
            out_tokens.append(tok)
            self.last_token[rid] = tok
            if is_finished:
                finish_req_ids.append(rid)
                self.clean_request(rid)
            else:
                continue_ids.append(i)

        if continue_ids:
            next_tok = torch.tensor([out_tokens[i] for i in continue_ids],
                                    dtype=torch.long, device=self.embedding.device)
            continued_data = self.embedding.index_select(0, next_tok)
        else:
            continued_data = completed_hidden[:0]
        return continue_ids, finish_req_ids, out_tokens, continued_data
