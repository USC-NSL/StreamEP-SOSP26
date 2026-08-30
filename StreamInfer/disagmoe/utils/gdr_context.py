from disagmoe_c import GdrContext as GdrContextImpl
from disagmoe.utils.tensor_utils import get_cuda_aligned_tensor
import os
import torch

# GDR staging writes metadata into BAR1-mapped GPU memory, bypassing the
# CUDA stream; the ring below keeps a slot from being reused while its
# consumer kernel is still queued.
use_gdrcopy_optimization = os.environ.get("DMOE_USE_GDRCOPY", "1") == "1"

class GdrContext:
    
    def __init__(self, tensor: torch.Tensor):
        self.gdr_context = GdrContextImpl(tensor)
        self.tensor = tensor
        
    def copy_from_host(self, src: int, nbytes: int, dst_offset: int = 0) -> None:
        self.gdr_context.copy_from_host(src, nbytes, dst_offset)
        
    def copy_from_host_tensor(self, src: torch.Tensor, nbytes: int = 0) -> None:
        self.gdr_context.copy_from_host_tensor(src, nbytes)
        
    def copy_to_host(self, dest: int, nbytes: int, src_offset: int = 0) -> None:
        self.gdr_context.copy_to_host(dest, nbytes, src_offset)
        
    def copy_to_host_tensor(self, dst: torch.Tensor, nbytes: int = 0) -> None:
        self.gdr_context.copy_to_host_tensor(dst, nbytes)
        
    def fill(self, value: int, nbytes: int, dst_offset: int = 0) -> None:
        self.gdr_context.fill(value, nbytes, dst_offset)
        
    def copy_from_host_int32(self, src: list[int]) -> None:
        self.gdr_context.copy_from_host_int32(src)
        
    def copy_from_host_int64(self, src: list[int]) -> None:
        self.gdr_context.copy_from_host_int64(src)
        
    def copy_from_host_float(self, src: list[float]) -> None:
        self.gdr_context.copy_from_host_float(src)
        
    def copy_to_host_int32(self, nelems: int) -> list[int]:
        return self.gdr_context.copy_to_host_int32(nelems)
    
    def copy_to_host_float(self, nelems: int) -> list[float]:
        return self.gdr_context.copy_to_host_float(nelems)
        
    def copy_to_host_int64(self, nelems: int) -> list[int]:
        return self.gdr_context.copy_to_host_int64(nelems)
    
class GdrRingBuffer:
    """Large N-slot ring of GDR-staged buffers (deadlock-free).

    GdrContext.copy_from_host writes GPU memory via a raw memcpy into the
    BAR1-mapped pointer, bypassing the CUDA stream; the consumer kernel is
    queued on the stream and runs later. The previous 2-slot double buffer let
    the CPU lap the GPU after just 2 steps and overwrite a slot whose consumer
    was still in flight -> the D20 concurrency corruption.

    A blocking per-slot event fence is NOT usable here: get_one_handle runs on
    the engine's main loop thread, which also drives cross-engine NCCL/ZMQ
    transport; blocking it on a CUDA event stalls the whole disaggregated
    pipeline and deadlocks (observed at bs=20). Instead we make the ring large
    enough that the CPU cannot lap it: the dispatcher's max_pending_sends
    backpressure bounds how far the CPU runs ahead of the GPU, and N slots
    (default 256 >> any realistic in-flight batch depth) means a slot is only
    reused long after its consumer kernel has retired. No CPU blocking, so no
    deadlock; correct as long as N exceeds the CPU-ahead depth (raise
    DMOE_GDR_RING_SLOTS if corruption ever reappears under extreme load)."""

    def __init__(self, nelems: int, dtype: torch.dtype, device: str = "cuda",
                 n_slots: int = None):
        n = int(n_slots if n_slots is not None
                else os.environ.get("DMOE_GDR_RING_SLOTS", "256"))
        assert n >= 2
        self.buffers = [get_cuda_aligned_tensor(nelems, dtype, device=device)
                        for _ in range(n)]
        self.gdr_contexts = [GdrContext(b) for b in self.buffers]
        self.n = n
        self.idx = -1

    def get_one_handle(self) -> GdrContext:
        self.idx = (self.idx + 1) % self.n
        return self.gdr_contexts[self.idx]


# back-compat alias: call sites construct GdrDoubleBuffer(nelems, dtype, device)
GdrDoubleBuffer = GdrRingBuffer
