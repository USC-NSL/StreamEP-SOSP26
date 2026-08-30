# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import ctypes
import logging
import time
from abc import ABC
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.distributed

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cudaEventRecordWithFlags via ctypes
#
# torch.cuda.Event.record() uses cudaEventRecord() which does NOT set the
# CU_EVENT_RECORD_EXTERNAL flag.  Without this flag, events recorded inside
# a CUDA graph capture produce no valid timing data on graph replay — the
# events become "internal" to the graph and elapsed_time() returns 0.
#
# Fix: call cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal)
# so the events are treated as external synchronization points whose timing
# is observable after replay.
#
# See: https://github.com/pytorch/pytorch/issues/115339
# ---------------------------------------------------------------------------
_cudart_lib = None
try:
    _cudart_lib = ctypes.CDLL("libcudart.so")
except OSError:
    for _suffix in ("libcudart.so.12", "libcudart.so.11"):
        try:
            _cudart_lib = ctypes.CDLL(_suffix)
            break
        except OSError:
            continue

_CUDA_EVENT_RECORD_EXTERNAL = 0x1


def _event_record_for_graph(event: torch.cuda.Event):
    """Record a CUDA event with cudaEventRecordExternal flag.

    Must be used instead of event.record() for events that are recorded
    inside CUDA graph capture regions and need valid elapsed_time() after
    graph replay.
    """
    stream = torch.cuda.current_stream()
    handle = event.cuda_event
    if _cudart_lib is not None and handle != 0:
        ret = _cudart_lib.cudaEventRecordWithFlags(
            ctypes.c_void_p(handle),
            ctypes.c_void_p(stream.cuda_stream),
            ctypes.c_uint(_CUDA_EVENT_RECORD_EXTERNAL),
        )
        if ret != 0:
            logger.warning(
                "cudaEventRecordWithFlags returned %d, falling back to event.record()",
                ret,
            )
            event.record(stream)
    else:
        event.record(stream)


_OutputMode = Literal["file", "object"]

# ---------------------------------------------------------------------------
# Global GPU buffer for local-token counting.
#
# FusedMoE.forward() writes to this buffer using **pure tensor ops** so the
# writes are captured inside torch.compile / CUDA-graph regions.  After each
# model-forward step the model-runner calls ``capture_step`` on the recorder
# which async-copies the buffer to CPU and appends it to the per-step list.
# ---------------------------------------------------------------------------
_local_tokens_gpu_buffer: Optional[torch.Tensor] = None  # shape [num_layers], int32


def get_local_tokens_gpu_buffer() -> Optional[torch.Tensor]:
    return _local_tokens_gpu_buffer


def init_local_tokens_gpu_buffer(num_layers: int, device: str = "cuda"):
    global _local_tokens_gpu_buffer
    _local_tokens_gpu_buffer = torch.zeros(num_layers, dtype=torch.int32, device=device)


class MoEKernelBalanceRecorder(ABC):
    """Records MoE kernel execution time per layer per forward step.

    Usage from each MoE layer's forward method::

        recorder = get_global_moe_kernel_balance_recorder()
        recorder.record_start(self.layer_id)
        output = self.run_moe_core(...)
        recorder.record_end(self.layer_id)

    Step boundaries are detected automatically: ``layer_idx == 0`` signals the
    start of a new forward step (finalizing the previous one).

    ``set_forward_mode`` should be called once per forward pass (e.g. from
    the model runner) so that dump-time filtering to decode-only steps works.

    ``capture_step`` should be called from the model runner **after** each
    forward pass.  It async-copies the GPU local-tokens buffer to CPU and
    stores the snapshot for the current step.

    At dump time, gathers data from all ranks, filters to steps where all ranks
    are in decode mode, and produces a 3D tensor of shape
    [#all_decode_steps, #layers, #EP_ranks].
    """

    @staticmethod
    def init_new(
        num_layers: int,
        rank: int,
        world_size: int,
        enabled: bool = False,
    ) -> "MoEKernelBalanceRecorder":
        if enabled:
            return _MoEKernelBalanceRecorderReal(num_layers, rank, world_size)
        return _MoEKernelBalanceRecorderNoop()

    def set_forward_mode(self, forward_mode: ForwardMode):
        pass

    def record_start(self):
        pass

    def record_end(self):
        pass

    def record_moe_start(self, layer_id: int):
        pass

    def record_moe_end(self, layer_id: int):
        pass

    def record_attn_start(self, layer_id: int):
        pass

    def record_attn_end(self, layer_id: int):
        pass

    def record_ag_start(self, layer_id: int):
        pass

    def record_ag_end(self, layer_id: int):
        pass

    def record_ar_start(self, layer_id: int):
        pass

    def record_ar_end(self, layer_id: int):
        pass

    def record_metadata_ar_start(self, layer_id: int):
        pass

    def record_metadata_ar_end(self, layer_id: int):
        pass

    def record_fwd_start(self, layer_id: int):
        pass

    def record_fwd_end(self, layer_id: int):
        pass

    def capture_step(self, batch_size: int = 0):
        """Called from model_runner after each forward to snapshot the GPU buffer."""
        pass

    def start_record(self):
        pass

    def stop_record(self):
        pass

    def dump(self, output_mode: _OutputMode = "file"):
        pass

    @property
    def recording(self):
        return False


class _MoEKernelBalanceRecorderNoop(MoEKernelBalanceRecorder):
    pass


class _MoEKernelBalanceRecorderReal(MoEKernelBalanceRecorder):
    def __init__(self, num_layers: int, rank: int, world_size: int):
        self._num_layers = num_layers
        self._rank = rank
        self._world_size = world_size
        self._recording = False

        self._forward_modes: List[int] = []
        self._batch_sizes: List[int] = []
        self._timestamps: List[float] = []
        self._local_tokens_per_step: List[torch.Tensor] = []
        self._step_times: List[torch.Tensor] = []  # [step] -> tensor[num_layers] ms

        # Per-step phase timing buffers (attn / ag / ar / fwd).
        self._attn_step_times: List[torch.Tensor] = []
        self._ag_step_times: List[torch.Tensor] = []
        self._ar_step_times: List[torch.Tensor] = []
        self._fwd_step_times: List[torch.Tensor] = []

        self._current_forward_mode_value: int = -1

        # Pre-allocated CUDA events for per-layer timing.
        # These are recorded unconditionally in FusedMoE.forward() so that
        # the event.record() calls are captured inside CUDA graphs during
        # warmup.  During graph replay the same events are re-recorded
        # automatically by the replayed graph.
        self._start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]
        self._end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]

        # Per-layer CUDA events for attn / ag / ar phase timing.
        self._attn_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]
        self._attn_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]
        self._ag_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]
        self._ag_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]
        self._ar_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]
        self._ar_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]
        # Per-step timing for metadata all-reduce (positions + req_ids reconstruction
        # in baseline, or routing-gather all-reduce after local DP gating).
        self._metadata_ar_step_times: List[torch.Tensor] = []
        self._metadata_ar_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]
        self._metadata_ar_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]
        self._fwd_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]
        self._fwd_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(num_layers)
        ]

    def set_forward_mode(self, forward_mode: ForwardMode):
        if self._recording:
            self._current_forward_mode_value = forward_mode.value

    def record_start(self):
        if not self._recording:
            return
        self._timestamps.append(time.time())

    def record_end(self):
        pass

    def record_moe_start(self, layer_id: int):
        # No _recording guard — must always run so event.record() calls
        # are captured inside CUDA graphs during warmup/capture.
        #
        # During graph capture we must use cudaEventRecordWithFlags with
        # CU_EVENT_RECORD_EXTERNAL so that the events produce valid timing
        # data on graph replay.  During normal (non-graph) execution,
        # regular event.record() suffices and is cheaper.
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._start_events[layer_id])
        else:
            self._start_events[layer_id].record()

    def record_moe_end(self, layer_id: int):
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._end_events[layer_id])
        else:
            self._end_events[layer_id].record()

    def record_attn_start(self, layer_id: int):
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._attn_start_events[layer_id])
        else:
            self._attn_start_events[layer_id].record()

    def record_attn_end(self, layer_id: int):
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._attn_end_events[layer_id])
        else:
            self._attn_end_events[layer_id].record()

    def record_ag_start(self, layer_id: int):
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._ag_start_events[layer_id])
        else:
            self._ag_start_events[layer_id].record()

    def record_ag_end(self, layer_id: int):
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._ag_end_events[layer_id])
        else:
            self._ag_end_events[layer_id].record()

    def record_ar_start(self, layer_id: int):
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._ar_start_events[layer_id])
        else:
            self._ar_start_events[layer_id].record()

    def record_ar_end(self, layer_id: int):
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._ar_end_events[layer_id])
        else:
            self._ar_end_events[layer_id].record()

    def record_metadata_ar_start(self, layer_id: int):
        # Times the metadata all-reduce (positions + req_ids reconstruction in
        # baseline; routing-gather all-reduce after local DP gating).
        # No _recording guard — must always run so event.record() calls are
        # captured inside CUDA graphs during warmup/capture.
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._metadata_ar_start_events[layer_id])
        else:
            self._metadata_ar_start_events[layer_id].record()

    def record_metadata_ar_end(self, layer_id: int):
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._metadata_ar_end_events[layer_id])
        else:
            self._metadata_ar_end_events[layer_id].record()

    def record_fwd_start(self, layer_id: int):
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._fwd_start_events[layer_id])
        else:
            self._fwd_start_events[layer_id].record()

    def record_fwd_end(self, layer_id: int):
        if torch.cuda.is_current_stream_capturing():
            _event_record_for_graph(self._fwd_end_events[layer_id])
        else:
            self._fwd_end_events[layer_id].record()

    def capture_step(self, batch_size: int = 0):
        if not self._recording:
            return
        gpu_buf = get_local_tokens_gpu_buffer()
        if gpu_buf is None:
            return
        cpu_snapshot = gpu_buf.to("cpu", non_blocking=True)
        moe_batch_size = int(cpu_snapshot.sum().item())
        self._forward_modes.append(self._current_forward_mode_value)
        self._batch_sizes.append(moe_batch_size)
        self._local_tokens_per_step.append(cpu_snapshot)

        # Read per-layer MoE timing from pre-allocated events.
        # Must sync because the same events are overwritten on the next
        # forward step (or graph replay).
        torch.cuda.synchronize()
        times = torch.zeros(self._num_layers, dtype=torch.float32)
        attn_times = torch.zeros(self._num_layers, dtype=torch.float32)
        ag_times = torch.zeros(self._num_layers, dtype=torch.float32)
        ar_times = torch.zeros(self._num_layers, dtype=torch.float32)
        fwd_times = torch.zeros(self._num_layers, dtype=torch.float32)
        for i in range(self._num_layers):
            try:
                times[i] = self._start_events[i].elapsed_time(self._end_events[i])
            except (RuntimeError, ValueError):
                times[i] = 0.0
            try:
                attn_times[i] = self._attn_start_events[i].elapsed_time(
                    self._attn_end_events[i]
                )
            except (RuntimeError, ValueError):
                attn_times[i] = 0.0
            try:
                ag_times[i] = self._ag_start_events[i].elapsed_time(
                    self._ag_end_events[i]
                )
            except (RuntimeError, ValueError):
                ag_times[i] = 0.0
            try:
                ar_times[i] = self._ar_start_events[i].elapsed_time(
                    self._ar_end_events[i]
                )
            except (RuntimeError, ValueError):
                ar_times[i] = 0.0
            try:
                fwd_times[i] = self._fwd_start_events[i].elapsed_time(
                    self._fwd_end_events[i]
                )
            except (RuntimeError, ValueError):
                fwd_times[i] = 0.0
        metadata_ar_times = torch.zeros(self._num_layers, dtype=torch.float32)
        for i in range(self._num_layers):
            try:
                metadata_ar_times[i] = self._metadata_ar_start_events[i].elapsed_time(
                    self._metadata_ar_end_events[i]
                )
            except (RuntimeError, ValueError):
                metadata_ar_times[i] = 0.0
        self._step_times.append(times)
        self._attn_step_times.append(attn_times)
        self._ag_step_times.append(ag_times)
        self._ar_step_times.append(ar_times)
        self._fwd_step_times.append(fwd_times)
        self._metadata_ar_step_times.append(metadata_ar_times)

    def start_record(self):
        self._recording = True
        self._reset()

    def stop_record(self):
        self._recording = False

    def _reset(self):
        self._forward_modes.clear()
        self._batch_sizes.clear()
        self._timestamps.clear()
        self._local_tokens_per_step.clear()
        self._step_times.clear()
        self._attn_step_times.clear()
        self._ag_step_times.clear()
        self._ar_step_times.clear()
        self._fwd_step_times.clear()
        self._metadata_ar_step_times.clear()
        self._current_forward_mode_value = -1

    @property
    def recording(self):
        return self._recording

    def dump(self, output_mode: _OutputMode = "file"):
        num_steps = len(self._forward_modes)
        device = "cuda"

        logger.warning(
            "MoEKernelBalanceRecorder.dump(): rank=%d, num_steps=%d",
            self._rank, num_steps,
        )

        torch.cuda.synchronize()

        local_times_cpu = torch.zeros(
            (num_steps, self._num_layers), dtype=torch.float32
        )
        for step_idx in range(min(len(self._step_times), num_steps)):
            local_times_cpu[step_idx] = self._step_times[step_idx]

        local_attn_times_cpu = torch.zeros(
            (num_steps, self._num_layers), dtype=torch.float32
        )
        for step_idx in range(min(len(self._attn_step_times), num_steps)):
            local_attn_times_cpu[step_idx] = self._attn_step_times[step_idx]

        local_ag_times_cpu = torch.zeros(
            (num_steps, self._num_layers), dtype=torch.float32
        )
        for step_idx in range(min(len(self._ag_step_times), num_steps)):
            local_ag_times_cpu[step_idx] = self._ag_step_times[step_idx]

        local_ar_times_cpu = torch.zeros(
            (num_steps, self._num_layers), dtype=torch.float32
        )
        for step_idx in range(min(len(self._ar_step_times), num_steps)):
            local_ar_times_cpu[step_idx] = self._ar_step_times[step_idx]

        local_fwd_times_cpu = torch.zeros(
            (num_steps, self._num_layers), dtype=torch.float32
        )
        for step_idx in range(min(len(self._fwd_step_times), num_steps)):
            local_fwd_times_cpu[step_idx] = self._fwd_step_times[step_idx]

        local_metadata_ar_times_cpu = torch.zeros(
            (num_steps, self._num_layers), dtype=torch.float32
        )
        for step_idx in range(min(len(self._metadata_ar_step_times), num_steps)):
            local_metadata_ar_times_cpu[step_idx] = self._metadata_ar_step_times[step_idx]

        local_num_steps = torch.tensor([num_steps], dtype=torch.int64, device=device)
        all_num_steps_list = [
            torch.zeros(1, dtype=torch.int64, device=device)
            for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_num_steps_list, local_num_steps)
        max_steps = int(max(s.item() for s in all_num_steps_list))

        if max_steps == 0:
            logger.warning(
                "MoEKernelBalanceRecorder: no steps recorded, skipping dump."
            )
            return

        local_modes = torch.full(
            (max_steps,), fill_value=-1, dtype=torch.int32, device=device
        )
        local_times = torch.zeros(
            (max_steps, self._num_layers), dtype=torch.float32, device=device
        )
        local_attn = torch.zeros(
            (max_steps, self._num_layers), dtype=torch.float32, device=device
        )
        local_ag = torch.zeros(
            (max_steps, self._num_layers), dtype=torch.float32, device=device
        )
        local_ar = torch.zeros(
            (max_steps, self._num_layers), dtype=torch.float32, device=device
        )
        local_fwd = torch.zeros(
            (max_steps, self._num_layers), dtype=torch.float32, device=device
        )
        local_metadata_ar = torch.zeros(
            (max_steps, self._num_layers), dtype=torch.float32, device=device
        )
        local_ltok = torch.zeros(
            (max_steps, self._num_layers), dtype=torch.int32, device=device
        )
        local_ts = torch.zeros(max_steps, dtype=torch.float64, device=device)

        if num_steps > 0:
            local_modes[:num_steps] = torch.tensor(
                self._forward_modes, dtype=torch.int32, device=device
            )
            local_times[:num_steps] = local_times_cpu.to(device)
            local_attn[:num_steps] = local_attn_times_cpu.to(device)
            local_ag[:num_steps] = local_ag_times_cpu.to(device)
            local_ar[:num_steps] = local_ar_times_cpu.to(device)
            local_fwd[:num_steps] = local_fwd_times_cpu.to(device)
            local_metadata_ar[:num_steps] = local_metadata_ar_times_cpu.to(device)
            num_ts = min(len(self._timestamps), num_steps)
            if num_ts > 0:
                local_ts[:num_ts] = torch.tensor(
                    self._timestamps[:num_ts], dtype=torch.float64, device=device
                )
            if self._local_tokens_per_step:
                ltok_stacked = torch.stack(self._local_tokens_per_step)
                local_ltok[:num_steps] = ltok_stacked.to(
                    device=device, dtype=torch.int32
                )

        all_modes_list = [
            torch.zeros_like(local_modes) for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_modes_list, local_modes)
        all_modes = torch.stack(all_modes_list)

        decode_value = ForwardMode.DECODE.value
        all_decode_mask = (all_modes == decode_value).all(dim=0)

        all_times_list = [
            torch.zeros_like(local_times) for _ in range(self._world_size)
        ]
        torch.distributed.all_gather(all_times_list, local_times)
        all_times = torch.stack(all_times_list)

        all_attn_list = [torch.zeros_like(local_attn) for _ in range(self._world_size)]
        torch.distributed.all_gather(all_attn_list, local_attn)
        all_attn = torch.stack(all_attn_list)

        all_ag_list = [torch.zeros_like(local_ag) for _ in range(self._world_size)]
        torch.distributed.all_gather(all_ag_list, local_ag)
        all_ag = torch.stack(all_ag_list)

        all_ar_list = [torch.zeros_like(local_ar) for _ in range(self._world_size)]
        torch.distributed.all_gather(all_ar_list, local_ar)
        all_ar = torch.stack(all_ar_list)

        all_fwd_list = [torch.zeros_like(local_fwd) for _ in range(self._world_size)]
        torch.distributed.all_gather(all_fwd_list, local_fwd)
        all_fwd = torch.stack(all_fwd_list)

        all_metadata_ar_list = [torch.zeros_like(local_metadata_ar) for _ in range(self._world_size)]
        torch.distributed.all_gather(all_metadata_ar_list, local_metadata_ar)
        all_metadata_ar = torch.stack(all_metadata_ar_list)

        all_ltok_list = [torch.zeros_like(local_ltok) for _ in range(self._world_size)]
        torch.distributed.all_gather(all_ltok_list, local_ltok)
        all_ltok = torch.stack(all_ltok_list)

        all_ts_list = [torch.zeros_like(local_ts) for _ in range(self._world_size)]
        torch.distributed.all_gather(all_ts_list, local_ts)
        all_ts = torch.stack(all_ts_list)

        decode_times = all_times[:, all_decode_mask, :]
        result = decode_times.permute(1, 2, 0).contiguous()
        decode_attn = all_attn[:, all_decode_mask, :].permute(1, 2, 0).contiguous()
        decode_ag = all_ag[:, all_decode_mask, :].permute(1, 2, 0).contiguous()
        decode_ar = all_ar[:, all_decode_mask, :].permute(1, 2, 0).contiguous()
        decode_fwd = all_fwd[:, all_decode_mask, :].permute(1, 2, 0).contiguous()
        decode_metadata_ar = all_metadata_ar[:, all_decode_mask, :].permute(1, 2, 0).contiguous()
        decode_ltok = all_ltok[:, all_decode_mask, :]
        decode_ltok = decode_ltok.permute(1, 2, 0).contiguous()
        decode_ts = all_ts[:, all_decode_mask].permute(1, 0).contiguous()

        output = dict(
            rank=self._rank,
            moe_times=result,
            attn_times=decode_attn,
            ag_times=decode_ag,
            ar_times=decode_ar,
            fwd_times=decode_fwd,
            metadata_ar_times=decode_metadata_ar,
            local_token_counts=decode_ltok,
            timestamps=decode_ts,
            num_total_steps=max_steps,
            num_decode_steps=result.shape[0],
        )

        self._reset()

        if output_mode == "file":
            if self._rank == 0:
                _dump_to_file(f"moe_kernel_balance_{time.time()}.pt", output)
        elif output_mode == "object":
            return output
        else:
            raise NotImplementedError


def _dump_to_file(name: str, data):
    save_dir = Path(envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.get())
    path_output = save_dir / name
    logger.info(f"Write MoE kernel balance data to {path_output}")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path_output))


_global_moe_kernel_balance_recorder: Optional[MoEKernelBalanceRecorder] = (
    _MoEKernelBalanceRecorderNoop()
)


def get_global_moe_kernel_balance_recorder() -> MoEKernelBalanceRecorder:
    return _global_moe_kernel_balance_recorder


def set_global_moe_kernel_balance_recorder(value: MoEKernelBalanceRecorder):
    global _global_moe_kernel_balance_recorder
    _global_moe_kernel_balance_recorder = value
