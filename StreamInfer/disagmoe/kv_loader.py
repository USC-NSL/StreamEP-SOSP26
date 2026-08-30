"""Mooncake KV ingestion for decode-only serving.

Reads the KV corpus that a stock sglang HiCache prefill server persists to
Mooncake (--enable-hierarchical-cache --hicache-storage-backend mooncake
--hicache-mem-layout page_first --page-size 16):

  key   = "{page_hash_hex}_0_k" / "{page_hash_hex}_0_v"
          (the 0 is sglang's attention-TP rank; dp-attention serving has
          attention-TP 1, so rank-0 objects carry the FULL KV heads)
  hash  = per-16-token-page sha256 chain over raw digests:
          digest_p = sha256(raw_digest_{p-1} || u32le(page token ids)),
          digest_-1 = b""
  value = one contiguous slab per page per K/V, page_first (token-major):
          bf16 [16 tokens][num_layers][kv_heads][head_dim]

HiCache persists FULL pages only, so a prompt's tail (len % 16 tokens) has no
objects. The loader installs floor(P/16) pages; the driver seeds the tail
tokens through the sampler's forced queue so ordinary decode steps compute
their KV in place before generation starts (sampler.py:admit_data).

Fetching is asynchronous: worker threads pull pages over the store's RDMA
path into per-worker pinned+registered staging; the engine loop installs
completed fetches (block allocation + one H2D + per-layer scatters) and only
then admits the request to the scheduler, so the loop never blocks on the
network. A request whose pages cannot be fetched is rejected (finished result
with no tokens) — there is no fallback prefill by design.
"""

import hashlib
import os
import queue
import struct
import threading
from dataclasses import dataclass, field
from typing import List

import torch

from disagmoe.utils.logger import get_logger

PAGE_TOKENS = 16


def page_chain_hashes(token_ids: List[int]) -> List[str]:
    """Hex digests of the sha256 page chain over the FULL 16-token pages."""
    hashes = []
    digest = b""
    for p in range(0, len(token_ids) // PAGE_TOKENS * PAGE_TOKENS, PAGE_TOKENS):
        page = token_ids[p: p + PAGE_TOKENS]
        h = hashlib.sha256()
        h.update(digest)
        h.update(struct.pack(f"<{len(page)}I", *page))
        digest = h.digest()
        hashes.append(digest.hex())
    return hashes


@dataclass
class KVLoadJob:
    req_id: int
    prompt_token_ids: List[int]
    ctx: object                    # opaque engine context (TokenizedRequest)


@dataclass
class KVLoadDone:
    job: KVLoadJob
    worker: int
    n_pages: int
    ok: bool
    # install() hands the worker its staging buffer back through this event
    released: threading.Event = field(default_factory=threading.Event)


class MooncakeKVLoader:

    def __init__(self, master_addr: str, local_ip: str,
                 kv_pool, block_mgr_c, block_size: int,
                 num_layers: int, kv_heads: int, head_dim: int,
                 max_prompt_tokens: int = 8192, num_workers: int = 4):
        from mooncake.store import MooncakeDistributedStore

        assert block_size == PAGE_TOKENS, \
            f"HiCache contract is {PAGE_TOKENS}-token pages; block_size={block_size}"
        self.kv_pool = kv_pool          # disagmoe MHATokenToKVPool
        self.block_mgr = block_mgr_c    # C++ BlockManager
        self.num_layers = num_layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        # one K or V object: bf16 [16, num_layers, kv_heads, head_dim]
        self.obj_bytes = PAGE_TOKENS * num_layers * kv_heads * head_dim * 2

        protocol = os.environ.get("STREAMEP_KV_PROTOCOL", "rdma")
        device = os.environ.get("STREAMEP_KV_DEVICE", "mlx5_1")
        self.store = MooncakeDistributedStore()
        ret = self.store.setup({
            "local_hostname": local_ip,
            "metadata_server": "P2PHANDSHAKE",
            "global_segment_size": 16 * 1024 * 1024,  # pure requester
            "local_buffer_size": 256 * 1024 * 1024,
            "protocol": protocol,
            "rdma_devices": device if protocol == "rdma" else "",
            "master_server_addr": master_addr,
        })
        assert ret == 0, f"[M-KVL] mooncake setup failed: {ret}"
        # Plug this process's own mandatory 16 MB segment (the allocator floor
        # forces mounting one; unmount_segment needs a UUID setup() never
        # exposes) with a pinned junk object so the master can never place KV
        # pages here — pages on engine segments would die at engine teardown.
        from mooncake.store import ReplicateConfig
        pin = ReplicateConfig()
        pin.replica_num = 1
        pin.preferred_segment = self.store.get_hostname()
        pin.with_soft_pin = True
        for mb in (15, 14, 13, 12):
            if self.store.put(f"kvloadpin:{self.store.get_hostname()}",
                              b"\0" * (mb * 1024 * 1024), pin) == 0:
                break
        else:
            get_logger().warning("[M-KVL] could not plug own segment")

        max_pages = max_prompt_tokens // PAGE_TOKENS
        stage_bytes = max_pages * 2 * self.obj_bytes
        # explicit device: engines run under a cuda default-device context,
        # and pin_memory requires a CPU tensor
        self._staging: List[torch.Tensor] = []
        for _ in range(num_workers):
            buf = torch.empty(stage_bytes, dtype=torch.uint8, device="cpu",
                              pin_memory=True)
            assert self.store.register_buffer(buf.data_ptr(), buf.numel()) == 0
            self._staging.append(buf)

        self._jobs: "queue.Queue[KVLoadJob]" = queue.Queue()
        self._done: "queue.Queue[KVLoadDone]" = queue.Queue()
        for w in range(num_workers):
            threading.Thread(target=self._worker_loop, args=(w,),
                             daemon=True, name=f"kv-loader-{w}").start()
        self.n_loaded = 0
        self.n_rejected = 0
        get_logger().info(
            f"[M-KVL] loader up: master={master_addr} protocol={protocol} "
            f"workers={num_workers} staging={stage_bytes >> 20}MB/worker")

    def submit(self, req_id: int, prompt_token_ids: List[int], ctx) -> None:
        self._jobs.put(KVLoadJob(req_id, prompt_token_ids, ctx))

    def poll(self, max_items: int = 8) -> List[KVLoadDone]:
        """Non-blocking drain of completed fetches (engine loop)."""
        out = []
        while len(out) < max_items:
            try:
                out.append(self._done.get_nowait())
            except queue.Empty:
                break
        return out

    def _worker_loop(self, w: int) -> None:
        buf = self._staging[w]
        base = buf.data_ptr()
        while True:
            job = self._jobs.get()
            n_pages = len(job.prompt_token_ids) // PAGE_TOKENS
            ok = True
            if n_pages > 0:
                assert 2 * n_pages * self.obj_bytes <= buf.numel(), \
                    f"prompt of {len(job.prompt_token_ids)} tokens exceeds staging"
                keys, ptrs, sizes = [], [], []
                for p, hx in enumerate(page_chain_hashes(job.prompt_token_ids)):
                    keys += [f"{hx}_0_k", f"{hx}_0_v"]
                    ptrs += [base + (2 * p) * self.obj_bytes,
                             base + (2 * p + 1) * self.obj_bytes]
                    sizes += [self.obj_bytes, self.obj_bytes]
                ok = False
                for attempt in range(3):
                    try:
                        codes = self.store.batch_get_into(keys, ptrs, sizes)
                        bad = [(k, c) for k, c in zip(keys, codes)
                               if c != self.obj_bytes]
                        if not bad:
                            ok = True
                            break
                        get_logger().warning(
                            f"[M-KVL] req {job.req_id}: {len(bad)}/{len(keys)} "
                            f"objects failed (attempt {attempt}); "
                            f"first: {bad[0][0]} rc={bad[0][1]}")
                    except Exception as e:  # noqa: BLE001
                        get_logger().warning(
                            f"[M-KVL] req {job.req_id} fetch attempt "
                            f"{attempt} raised: {e}")
            done = KVLoadDone(job, w, n_pages, ok)
            self._done.put(done)
            done.released.wait()

    @torch.inference_mode()
    def install(self, done: KVLoadDone) -> bool:
        """Engine-loop side: block allocation + one H2D + per-layer scatters.
        Always releases the worker's staging buffer."""
        try:
            if not done.ok:
                self.n_rejected += 1
                return False
            n_pages = done.n_pages
            # pre-admission allocation: the first scheduled batch skips
            # allocation (BlockManager::update_block_table sees the seq list)
            self.block_mgr.allocate(done.job.req_id, n_pages * PAGE_TOKENS)
            if n_pages == 0:
                self.n_loaded += 1
                return True
            block_ids = self.block_mgr.get_seq_block_ids(done.job.req_id)
            assert len(block_ids) >= n_pages
            buf = self._staging[done.worker]
            staged = buf[: 2 * n_pages * self.obj_bytes] \
                .view(torch.bfloat16) \
                .view(n_pages, 2, PAGE_TOKENS, self.num_layers,
                      self.kv_heads, self.head_dim).cuda(non_blocking=False)
            # page_first objects are token-major; pool blocks want
            # [pages, 16, H, D] per layer per K/V
            g = staged.permute(1, 3, 0, 2, 4, 5).contiguous()
            blocks = torch.tensor(block_ids[:n_pages], dtype=torch.long,
                                  device="cuda")
            for l in range(self.num_layers):
                layer_buf = self.kv_pool.get_kv_buffer(l)  # [2, blocks+1, 16, H, D]
                layer_buf[0].index_copy_(0, blocks, g[0, l])
                layer_buf[1].index_copy_(0, blocks, g[1, l])
            torch.cuda.current_stream().synchronize()
            self.n_loaded += 1
            return True
        except Exception as e:  # noqa: BLE001
            get_logger().error(f"[M-KVL] install for req {done.job.req_id} "
                               f"raised: {e}")
            self.n_rejected += 1
            return False
        finally:
            done.released.set()
