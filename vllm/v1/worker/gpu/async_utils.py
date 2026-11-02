# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors
from vllm.v1.worker.utils import raise_if_nan_logits

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.worker import PendingAuxOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass(frozen=True)
class StepTimingSample:
    forward_ms: float
    drafter_ms: float
    num_target_tokens: int
    num_reqs: int
    full_cudagraph: bool


def _timing_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


@dataclass
class StepTimingEvents:
    forward_start: torch.cuda.Event = field(default_factory=_timing_event)
    forward_end: torch.cuda.Event = field(default_factory=_timing_event)
    drafter_start: torch.cuda.Event = field(default_factory=_timing_event)
    drafter_end: torch.cuda.Event = field(default_factory=_timing_event)


class StepTimingCollector:
    """Times the steps run inside ``collect``; record calls no-op outside it.

    Every step gets its own events, so the steps queue back-to-back and the
    block resolves them all behind one sync on the way out.
    """

    def __init__(self):
        self._collecting = False
        self._step: StepTimingEvents | None = None
        self._batch = (False, 0, 0)
        self._timed: list[tuple[StepTimingEvents, tuple[bool, int, int]]] = []

    @contextlib.contextmanager
    def collect(self) -> Iterator[list[StepTimingSample]]:
        """Time every step run in this block.

        The yielded list holds one sample per timed step once the block exits;
        it stays empty inside the block, where the timings are still on device.
        """
        samples: list[StepTimingSample] = []
        self._collecting = True
        try:
            yield samples
        finally:
            self._collecting = False
            timed, self._timed, self._step = self._timed, [], None
        if not timed:
            return
        # Same stream, issue order: once the last step is done, so are the rest.
        timed[-1][0].drafter_end.synchronize()
        samples.extend(
            StepTimingSample(
                events.forward_start.elapsed_time(events.forward_end),
                events.drafter_start.elapsed_time(events.drafter_end),
                num_target_tokens,
                num_reqs,
                full_cudagraph,
            )
            for events, (full_cudagraph, num_target_tokens, num_reqs) in timed
        )

    def record_batch(self, input_batch: "InputBatch", full_cudagraph: bool) -> None:
        """Costs from different execution modes must not share a cost curve."""
        self._batch = (full_cudagraph, input_batch.num_tokens, input_batch.num_reqs)

    def forward_start(self) -> None:
        if self._collecting:
            self._step = StepTimingEvents()
            self._step.forward_start.record()

    def forward_end(self) -> None:
        if self._step is not None:
            self._step.forward_end.record()

    def drafter_start(self) -> None:
        if self._step is not None:
            self._step.drafter_start.record()

    def drafter_end(self) -> None:
        """Ends the step: only steps that reach here have a draft cost."""
        if self._step is not None:
            self._step.drafter_end.record()
            self._timed.append((self._step, self._batch))
            self._step = None


@dataclass(frozen=True)
class AsyncOutputEventSlot:
    """Two reusable events for one in-flight asynchronous output copy."""

    handoff_event: torch.cuda.Event
    copy_done_event: torch.cuda.Event


class AsyncOutputEventPool:
    """Bounded pool of CUDA events used by :class:`AsyncOutput`.

    A slot is leased until the associated D2H copy has completed. Re-recording
    either event before then would change the synchronization target observed by
    an older ``AsyncOutput``, so acquisition deliberately applies backpressure
    when all slots are in flight.
    """

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError(
                f"Async output event pool size must be positive, got {size}"
            )
        self._available = deque(
            AsyncOutputEventSlot(
                handoff_event=torch.cuda.Event(),
                copy_done_event=torch.cuda.Event(blocking=True),
            )
            for _ in range(size)
        )
        self._condition = threading.Condition()
        self._size = size
        self._in_use = 0
        self._high_watermark = 0
        self._wait_count = 0
        self._miss_count = 0

    def acquire(self, *, block: bool = True) -> AsyncOutputEventSlot | None:
        with self._condition:
            if not self._available:
                if not block:
                    self._miss_count += 1
                    return None
                self._wait_count += 1
            while not self._available:
                self._condition.wait()
            slot = self._available.popleft()
            self._in_use += 1
            self._high_watermark = max(self._high_watermark, self._in_use)
            return slot

    def release(self, slot: AsyncOutputEventSlot) -> None:
        with self._condition:
            self._available.append(slot)
            self._in_use -= 1
            self._condition.notify()

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "size": self._size,
                "in_use": self._in_use,
                "high_watermark": self._high_watermark,
                "wait_count": self._wait_count,
                "miss_count": self._miss_count,
            }


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        check_ep_fault: bool,
        pending_aux_output: "PendingAuxOutput | None",
        event_pool: AsyncOutputEventPool | None = None,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.pending_aux_output = pending_aux_output
        self._event_pool = event_pool
        # Blocking here can deadlock async scheduling: the output holding the
        # last slot may only be consumed after this Worker RPC returns. Fall
        # back to the legacy one-shot events when a deliberately undersized
        # pool is exhausted; a correctly sized pool should report zero misses.
        self._event_slot = (
            event_pool.acquire(block=False) if event_pool is not None else None
        )
        # Blocking (sleep) event avoids busy-polling the CUDA driver lock. The
        # legacy path intentionally remains unchanged when pooling is disabled.
        self.copy_event = (
            self._event_slot.copy_done_event
            if self._event_slot is not None
            else torch.cuda.Event(blocking=True)
        )
        self._has_fault: torch.Tensor | None = None

        try:
            with stream(copy_stream, main_stream):
                if self._event_slot is None:
                    copy_stream.wait_stream(main_stream)
                else:
                    # Avoid Stream.wait_stream(), which creates a fresh
                    # temporary CUDA event for every output.
                    self._event_slot.handoff_event.record(main_stream)
                    copy_stream.wait_event(self._event_slot.handoff_event)

                self.sampled_token_ids = async_copy_to_np(
                    sampler_output.sampled_token_ids
                )
                self.logprobs_tensors: LogprobsTensors | None = None
                if sampler_output.logprobs_tensors is not None:
                    self.logprobs_tensors = (
                        sampler_output.logprobs_tensors.to_cpu_nonblocking()
                    )
                self.num_nans: np.ndarray | None = None
                if sampler_output.num_nans is not None:
                    self.num_nans = async_copy_to_np(sampler_output.num_nans)
                self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
                self.sampling_mask_tensors: SamplingMaskTensors | None = None
                if sampler_output.sampling_mask_tensors is not None:
                    self.sampling_mask_tensors = (
                        sampler_output.sampling_mask_tensors.to_cpu_nonblocking()
                    )
                self.prompt_logprobs_dict = {
                    k: v.to_cpu_nonblocking() if v is not None else None
                    for k, v in self.model_runner_output.prompt_logprobs_dict.items()
                }
                token_id_logprobs = (
                    self.model_runner_output.prompt_token_id_logprobs_dict
                )
                self.prompt_token_id_logprobs_dict = {
                    k: v.to("cpu", non_blocking=True)
                    for k, v in token_id_logprobs.items()
                }
                if self.pending_aux_output is not None:
                    self.pending_aux_output.enqueue_cpu_copy(
                        num_sampled=self.num_sampled_tokens_np,
                        num_rejected=async_copy_to_np(sampler_output.num_rejected),
                    )
                if check_ep_fault:
                    has_fault = get_ep_all2all_manager().query_fault()
                    self._has_fault = has_fault.to("cpu", non_blocking=True)
                self.copy_event.record(copy_stream)
        except BaseException:
            if self._event_slot is not None:
                # Do not make a slot containing unfinished work reusable.
                copy_stream.synchronize()
                self._event_pool.release(self._event_slot)
                self._event_slot = None
            raise

    def get_output(self) -> ModelRunnerOutput:
        try:
            self.copy_event.synchronize()
        finally:
            if self._event_slot is not None:
                assert self._event_pool is not None
                self._event_pool.release(self._event_slot)
                self._event_slot = None

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.sampling_mask_tensors is not None:
            self.model_runner_output.sampling_masks = (
                self.sampling_mask_tensors.tolists()
            )

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )
            if envs.VLLM_RAISE_ON_LOGIT_NANS:
                raise_if_nan_logits(self.model_runner_output.num_nans_in_logits)

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        self.model_runner_output.prompt_token_id_logprobs_dict = (
            self.prompt_token_id_logprobs_dict
        )
        if self.pending_aux_output is not None:
            self.model_runner_output.aux_output_connector_output = (
                self.pending_aux_output.process_output()
            )

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: PoolerOutput,
        finished_mask: list[bool],
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            if isinstance(self.pooler_output, torch.Tensor) and all(finished_mask):
                self.pooler_output_cpu: PoolerOutput = self.pooler_output.to(
                    "cpu", non_blocking=True
                )
            else:
                outputs = (
                    self.pooler_output.unbind()
                    if isinstance(self.pooler_output, torch.Tensor)
                    else self.pooler_output
                )
                self.pooler_output_cpu = [
                    None
                    if output is None or not is_finished
                    else output.to("cpu", non_blocking=True)
                    for output, is_finished in zip(outputs, finished_mask, strict=True)
                ]
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        if isinstance(self.pooler_output_cpu, torch.Tensor):
            pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        else:
            pooler_output = self.pooler_output_cpu
        self.copy_event.synchronize()
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
