#ifndef GDR_RING_HPP
#define GDR_RING_HPP

#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>
#include <memory>
#include <cstdlib>

#include "gdr_context.hpp"
#include "tensor_utils.hpp"

// Large N-slot ring of GDR-staged buffers (deadlock-free). Mirrors the Python
// GdrRingBuffer (disagmoe/utils/gdr_context.py).
//
// GdrContext::copy_from_host writes GPU memory via a raw memcpy into the
// BAR1-mapped pointer, bypassing the CUDA stream; the consumer kernel is
// queued on the stream and runs later. A 2-slot alternating buffer let the CPU
// lap the GPU after 2 steps and overwrite a slot whose consumer was still in
// flight -> the D20 concurrency corruption.
//
// A blocking per-slot cudaEvent fence is NOT usable: these rings are advanced
// on threads that also drive cross-engine transport, so blocking on a CUDA
// event stalls the disaggregated pipeline and deadlocks (observed at bs=20).
// Instead the ring is made large enough (default 256 >> any realistic
// in-flight batch depth, bounded by the dispatcher's max_pending_sends
// backpressure) that a slot is only reused long after its consumer kernel has
// retired. No blocking, so no deadlock; correct as long as N exceeds the
// CPU-ahead depth (raise DMOE_GDR_RING_SLOTS if corruption reappears).
class GdrRing {
public:
    GdrRing(int64_t nelems, torch::Dtype dtype, int n_slots = -1) {
        if (n_slots < 0) {
            const char* env = std::getenv("DMOE_GDR_RING_SLOTS");
            n_slots = env ? std::atoi(env) : 256;
            if (n_slots < 2) n_slots = 256;
        }
        n_ = n_slots;
        ctxs_.reserve(n_);
        for (int i = 0; i < n_; i++) {
            auto t = get_cuda_aligned_tensor(nelems, dtype);
            ctxs_.push_back(std::make_shared<GdrContext>(t));
        }
    }

    // stream param kept for call-site symmetry; no fencing is performed
    gdr_context_t next(cudaStream_t /*stream*/ = nullptr) {
        idx_ = (idx_ + 1) % n_;
        return ctxs_[idx_];
    }

private:
    int n_ = 0;
    int idx_ = -1;
    std::vector<gdr_context_t> ctxs_;
};

#endif  // GDR_RING_HPP
