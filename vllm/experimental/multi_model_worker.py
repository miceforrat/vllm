# SPDX-License-Identifier: Apache-2.0
"""Experimental model-runner registry hosted by an existing vLLM worker."""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Mapping

from vllm.experimental.communicator_registry import CommunicatorRegistry
from vllm.experimental.model_instance_registry import (
    CommunicationGroupSpec,
    ModelInstance,
    ModelInstanceRegistry,
    ModelSpec,
)
from vllm.experimental.multi_model_topology import TopologyPlan


@dataclass(frozen=True)
class SchedulerModelRequest:
    """Request contract for a model managed by the vLLM scheduler."""

    prompt: Any
    sampling_params: Any
    request_id: str
    generate_kwargs: Mapping[str, Any] = field(default_factory=dict)


class MultiModelWorkerExtension:
    """Own a unified registry of scheduler-managed and runtime-added models."""

    def _model_instance_registry(self) -> ModelInstanceRegistry:
        registry = getattr(self, "_vllm_model_instance_registry", None)
        if registry is not None:
            return registry

        from vllm.distributed.parallel_state import get_tp_group

        registry = ModelInstanceRegistry()
        config = getattr(self, "vllm_config", None)
        additional = getattr(config, "additional_config", None)
        runtime = (
            additional.get("multi_model_runtime", {})
            if isinstance(additional, dict)
            else {}
        )
        primary_model_id = str(runtime.get("primary_model_id", "reasoner"))
        model_config = getattr(self, "model_config", None)
        model_path = str(getattr(model_config, "model", primary_model_id))
        tp_ranks = tuple(int(rank) for rank in get_tp_group().ranks)
        primary_spec = ModelSpec(
            model_id=primary_model_id,
            model_path=model_path,
            backend="vllm",
            capabilities=("text_generation",),
            default_task="text_generation",
            worker_ranks=tp_ranks,
        )
        registry.register(
            ModelInstance(
                spec=primary_spec,
                runner=self.model_runner,
                scheduler_managed=True,
                runtime_added=False,
            )
        )
        self._vllm_model_instance_registry = registry
        return registry

    @staticmethod
    def _describe_group(group: Any) -> dict[str, Any]:
        return {
            "ranks": list(group.ranks),
            "world_size": int(group.world_size),
            "rank_in_group": int(group.rank_in_group),
            "local_rank": int(group.local_rank),
            "unique_name": str(group.unique_name),
        }

    def inspect_multi_model_topology(
        self, worker_ranks: tuple[int, ...] = ()
    ) -> dict[str, Any]:
        """Describe the vLLM topology an auxiliary runner would inherit."""
        from vllm.distributed.parallel_state import (
            get_pp_group,
            get_tp_group,
            get_world_group,
        )

        placement = tuple(int(rank) for rank in worker_ranks)
        world = get_world_group()
        tp = get_tp_group()
        pp = get_pp_group()
        return {
            "rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "pid": os.getpid(),
            "device": str(self.device),
            "active_for_placement": not placement or self.rank in placement,
            "requested_worker_ranks": list(placement),
            "world": self._describe_group(world),
            "tp": self._describe_group(tp),
            "pp": self._describe_group(pp),
        }

    def register_model(
        self,
        payload: Mapping[str, Any],
        topology_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Materialize one runtime-added model in the unified registry."""
        unplaced_spec = ModelSpec.from_rpc_payload(payload)
        plan = TopologyPlan.from_rpc_payload(topology_payload)
        spec = plan.apply(unplaced_spec)
        if self.rank not in spec.worker_ranks:
            # 不需要创建并行组
            return {
                "model_id": spec.model_id,
                "active": False,
                "rank": self.rank,
            }
        registry = self._model_instance_registry()
        if spec.model_id in registry:
            raise ValueError(f"Model {spec.model_id!r} is already registered")
        if spec.backend != "vllm_omni":
            raise ValueError(f"Unsupported auxiliary backend: {spec.backend!r}")

        from vllm.distributed.parallel_state import get_tp_group

        # 目前只支持同态TP
        tp_ranks = tuple(int(rank) for rank in get_tp_group().ranks)
        if tuple(spec.worker_ranks) != tp_ranks:
            raise ValueError(
                "The homogeneous TP gate requires Generator placement to match "
                f"the vLLM TP group exactly: placement={spec.worker_ranks}, "
                f"tp={tp_ranks}"
            )

        tp_specs = tuple(
            group
            for group in spec.communication_groups
            if group.parallel_dimension == "tp"
        )
        if len(tp_specs) != 1:
            raise ValueError(
                "An auxiliary runner requires exactly one Engine-planned TP "
                f"communication group, got {len(tp_specs)}"
            )
        tp_spec = tp_specs[0]
        if tuple(tp_spec.ranks) != tp_ranks:
            raise ValueError(
                "The homogeneous TP gate requires the Engine-planned group "
                f"to match vLLM TP ranks: planned={tp_spec.ranks}, tp={tp_ranks}"
            )

        from vllm.experimental.omni_runner_adapter import (
            EmbeddedOmniDiffusionRunner,
            resolve_omni_diffusion_stage,
        )

        resolved = resolve_omni_diffusion_stage(
            spec.model_path,
            cli_overrides=spec.backend_options,
        )
        # 创建辅助runner
        adapter = EmbeddedOmniDiffusionRunner(
            resolved,
            device=str(self.device),
            memory_pool_tag_prefix=spec.model_id,
        )
        tp_group = None
        try:
            tp_group = self._communicator_registry().acquire(
                tp_spec, spec.model_id
            )
            runner = adapter.start_on_worker(
                self,
                tp_group=tp_group,
                load_format=spec.load_format,
            )
            execution_stream = getattr(
                self, "_multi_model_generator_stream", None
            )
            if execution_stream is not None:
                adapter.set_execution_stream(execution_stream)
            phase_selector = (
                getattr(self, "acquire_generator_phase_lease", None)
                if getattr(self, "_multi_model_resource_arbiter", None)
                is not None
                else None
            )
            if phase_selector is not None:
                adapter.set_generator_phase_stream_selector(phase_selector)
        except Exception:
            adapter.close()
            if tp_group is not None:
                self._communicator_registry().release(
                    spec.model_id, (tp_spec.group_id,)
                )
            raise
        instance = ModelInstance(
            spec=spec,
            adapter=adapter,
            runner=runner,
            runtime_added=True,
            communication_group_ids=(tp_spec.group_id,),
        )
        registry.register(instance)
        return self._describe_model_instance(instance)

    def register_auxiliary_runner(
        self,
        payload: Mapping[str, Any],
        topology_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Compatibility alias for clients created before ``register_model``."""
        return self.register_model(payload, topology_payload)

    def execute_auxiliary_runner(
        self, model_id: str, request: Any, profile_memory: bool = False
    ) -> Any:
        slot = self._get_auxiliary_slot(model_id)
        if profile_memory:
            output = slot.adapter.execute_model_profiled(request)
        else:
            output = slot.adapter.execute_model(request)
        # Every TP rank must execute the forward and its collectives, but only
        # one rank should serialize the assembled DiffusionOutput to the client.
        output_rank = min(slot.spec.worker_ranks)
        return output if self.rank == output_rank else None

    def execute_auxiliary_runner_timed(
        self, model_id: str, request: Any, profile_memory: bool = False
    ) -> dict[str, Any] | None:
        """Time an auxiliary forward without returning its media payload."""
        slot = self._get_auxiliary_slot(model_id)
        host_start = time.perf_counter()
        output = self.execute_auxiliary_runner(model_id, request, profile_memory)
        # execute_model records a completion event and installs a stream wait,
        # but that wait is asynchronous on the host. Timing-only RPC removes
        # the media serialization that previously synchronized the payload, so
        # wait for this runner's stream before recording host_end.
        slot.adapter.synchronize_execution_stream()
        host_end = time.perf_counter()
        if output is None:
            return None
        # collective_rpc must not serialize the raw image/video tensor for a
        # timing-only request. The regular execute path still returns output.
        del output
        return {
            "host_start": host_start,
            "host_end": host_end,
            "seconds": host_end - host_start,
            "worker_thread_ident": threading.get_ident(),
            "worker_thread_name": threading.current_thread().name,
        }

    def _auxiliary_thread_pool(self) -> ThreadPoolExecutor:
        pool = getattr(self, "_vllm_auxiliary_thread_pool", None)
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"aux-runner-rank-{self.rank}",
            )
            self._vllm_auxiliary_thread_pool = pool
        return pool

    def prepare_auxiliary_runner(
        self,
        model_id: str,
        request: Any,
        profile_memory: bool = False,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Prepare a worker-local asynchronous forward behind a start gate."""
        slot = self._get_auxiliary_slot(model_id)
        if slot.pending_future is not None:
            raise RuntimeError(f"Runner {model_id!r} already has pending work")
        start = threading.Event()
        started = threading.Event()
        resolved_job_id = job_id or f"prepared-{model_id}"

        def run() -> Any:
            if not start.wait(timeout=300.0):
                raise TimeoutError(
                    f"Timed out waiting to release auxiliary runner {model_id!r}"
                )
            import torch

            torch.cuda.set_device(self.device)
            started.set()
            return self.execute_auxiliary_runner_timed(
                model_id, request, profile_memory
            )

        slot.pending_start = start
        slot.pending_started = started
        slot.pending_job_id = resolved_job_id
        slot.pending_future = self._auxiliary_thread_pool().submit(run)
        return {
            "model_id": model_id,
            "job_id": resolved_job_id,
            "rank": int(self.rank),
            "prepared": True,
        }

    def start_auxiliary_runner(
        self,
        model_id: str,
        request: Any,
        profile_memory: bool = False,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Start an auxiliary forward independently of primary-model work."""
        prepared = self.prepare_auxiliary_runner(
            model_id,
            request,
            profile_memory,
            job_id,
        )
        slot = self._get_auxiliary_slot(model_id)
        assert slot.pending_start is not None
        assert slot.pending_started is not None
        slot.pending_start.set()
        slot.pending_start = None
        if not slot.pending_started.wait(timeout=30.0):
            raise TimeoutError(
                f"Timed out starting auxiliary runner {model_id!r}"
            )
        return {
            **prepared,
            "prepared": False,
            "started": True,
        }

    def release_prepared_auxiliary_runners(self) -> list[str]:
        released = []
        for model_id, slot in self._model_instance_registry().runtime_items():
            if slot.pending_start is not None:
                slot.pending_start.set()
                slot.pending_start = None
                released.append(model_id)
        return released

    def auxiliary_runner_job_status(
        self, model_id: str, job_id: str | None = None
    ) -> dict[str, Any]:
        slot = self._get_auxiliary_slot(model_id)
        if job_id is not None and slot.pending_job_id != job_id:
            raise RuntimeError(
                f"Runner {model_id!r} has job {slot.pending_job_id!r}, "
                f"not {job_id!r}"
            )
        future = slot.pending_future
        return {
            "model_id": model_id,
            "job_id": slot.pending_job_id,
            "rank": int(self.rank),
            "pending": future is not None,
            "started": (
                slot.pending_started is not None
                and slot.pending_started.is_set()
            ),
            "done": future is not None and future.done(),
        }

    def collect_auxiliary_runner(
        self,
        model_id: str,
        job_id: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        slot = self._get_auxiliary_slot(model_id)
        if job_id is not None and slot.pending_job_id != job_id:
            raise RuntimeError(
                f"Runner {model_id!r} has job {slot.pending_job_id!r}, "
                f"not {job_id!r}"
            )
        future = slot.pending_future
        if future is None:
            raise RuntimeError(f"Runner {model_id!r} has no pending work")
        try:
            return future.result(timeout=timeout)
        finally:
            if future.done():
                slot.pending_future = None
                slot.pending_start = None
                slot.pending_started = None
                slot.pending_job_id = None

    def create_auxiliary_execution_stream(
        self,
        model_id: str,
        kind: str = "dedicated",
        sm_count: int | None = None,
    ) -> dict[str, Any]:
        slot = self._get_auxiliary_slot(model_id)
        if kind == "default":
            slot.adapter.set_execution_stream(None)
        elif kind == "dedicated":
            slot.adapter.create_execution_stream()
        elif kind == "green":
            if sm_count is None:
                raise ValueError("sm_count is required for a Green Context stream")
            slot.adapter.create_green_context_execution_stream(sm_count)
        else:
            raise ValueError(f"Unsupported execution stream kind: {kind!r}")
        return slot.adapter.stream_control_stats()

    def auxiliary_runner_metadata(self, model_id: str) -> dict[str, Any]:
        return self._describe_model_instance(self._get_auxiliary_slot(model_id))

    def model_registry_metadata(self) -> list[dict[str, Any]]:
        """Describe every model hosted by this Worker, including primary."""
        return [
            self._describe_model_instance(instance)
            for instance in self._model_instance_registry().values()
        ]

    def close_auxiliary_runners(self) -> list[str]:
        registry = self._model_instance_registry()
        slots = list(registry.runtime_instances())
        model_ids = [slot.spec.model_id for slot in slots]
        self.release_prepared_auxiliary_runners()
        for slot in reversed(slots):
            if slot.pending_future is not None:
                slot.pending_future.result()
                slot.pending_future = None
                slot.pending_start = None
                slot.pending_started = None
                slot.pending_job_id = None
            # Drop the registry's duplicate runner reference before adapter
            # teardown.  This lets the runner/pipeline destruct while the Omni
            # compatibility group views are still installed.
            slot.runner = None
            slot.adapter.close()
            self._communicator_registry().release(
                slot.spec.model_id, slot.communication_group_ids
            )
            registry.remove(slot.spec.model_id)
        pool = getattr(self, "_vllm_auxiliary_thread_pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
            self._vllm_auxiliary_thread_pool = None
        return model_ids

    def _communicator_registry(self) -> CommunicatorRegistry:
        registry = getattr(self, "_vllm_communicator_registry", None)
        if registry is None:
            registry = CommunicatorRegistry()
            self._vllm_communicator_registry = registry
        return registry

    def _auxiliary_runner_slots(self) -> dict[str, ModelInstance]:
        """Compatibility view containing only runtime-added models."""
        registry = getattr(self, "_vllm_model_instance_registry", None)
        if registry is None:
            return {}
        return dict(registry.runtime_items())

    def _get_auxiliary_slot(self, model_id: str) -> ModelInstance:
        try:
            instance = self._model_instance_registry().get(model_id)
        except KeyError as exc:
            raise KeyError(f"Unknown auxiliary runner: {model_id!r}") from exc
        if not instance.runtime_added:
            raise KeyError(f"Model {model_id!r} is not a runtime-added runner")
        return instance

    def _describe_model_instance(self, slot: ModelInstance) -> dict[str, Any]:
        runner = slot.runner
        description = {
            "model_id": slot.spec.model_id,
            "active": True,
            "rank": self.rank,
            "pid": os.getpid(),
            "worker_id": id(self),
            "device": str(self.device),
            "primary_runner_type": (
                f"{type(self.model_runner).__module__}."
                f"{type(self.model_runner).__name__}"
            ),
            "backend": slot.spec.backend,
            "capabilities": list(slot.spec.capabilities),
            "default_task": slot.spec.default_task,
            "scheduler_managed": slot.scheduler_managed,
            "runtime_added": slot.runtime_added,
            "runner_type": (
                f"{type(runner).__module__}.{type(runner).__name__}"
            ),
        }
        if slot.adapter is None:
            return description
        description.update({
            "pipeline_type": type(runner.pipeline).__name__,
            "parallel_topology": slot.adapter.parallel_topology_stats(),
            "communication_groups": [
                self._describe_group(
                    self._communicator_registry().get(group_id)
                )
                for group_id in slot.communication_group_ids
            ],
            "memory_pool_usage_bytes": slot.adapter.memory_pool_usage(),
            "stream_control": slot.adapter.stream_control_stats(),
        })
        return description


class WorkerHostedRunnerClient:
    """Frontend control plane for scheduler- and Worker-managed models."""

    def __init__(self, model_frontend: Any):
        self._model_frontend = model_frontend
        self._llm_engine = getattr(model_frontend, "llm_engine", model_frontend)
        self._runtime_model_specs: dict[str, ModelSpec] = {}

    def _runtime_config(self) -> Mapping[str, Any]:
        config = getattr(self._llm_engine, "vllm_config", None)
        additional = getattr(config, "additional_config", None)
        if not isinstance(additional, Mapping):
            return {}
        runtime = additional.get("multi_model_runtime", {})
        return runtime if isinstance(runtime, Mapping) else {}

    @property
    def primary_model_id(self) -> str:
        return str(self._runtime_config().get("primary_model_id", "reasoner"))

    def _dynamic_arbitration_enabled(self) -> bool:
        green_context = self._runtime_config().get("green_context", {})
        return (
            isinstance(green_context, Mapping)
            and green_context.get("partition_policy") == "dynamic"
        )

    def submit(
        self,
        model_id: str,
        request: Any,
        *,
        profile_memory: bool = False,
    ) -> "ModelRequestFuture":
        """Submit one request without exposing the model's execution owner."""
        if model_id in self._runtime_model_specs:
            return self.start_async(
                model_id, request, profile_memory=profile_memory
            )
        if model_id != self.primary_model_id:
            raise KeyError(f"Unknown model: {model_id!r}")
        if not isinstance(request, SchedulerModelRequest):
            raise TypeError(
                "Scheduler-managed models require SchedulerModelRequest"
            )
        generate = getattr(self._model_frontend, "generate", None)
        if generate is None:
            raise RuntimeError(
                "Submitting the primary model requires the LLM frontend, not "
                "only its llm_engine"
            )
        dynamic = self._dynamic_arbitration_enabled()
        if dynamic:
            self.set_primary_demand(True)

        future: Future[Any] = Future()
        try:
            future.set_result(
                generate(
                    request.prompt,
                    request.sampling_params,
                    **dict(request.generate_kwargs),
                )
            )
        except BaseException as exc:
            future.set_exception(exc)
        finally:
            if dynamic:
                self.set_primary_demand(False)
        return SchedulerModelFuture(model_id, request.request_id, future)

    def register(self, spec: ModelSpec) -> list[dict[str, Any]]:
        plan = self._topology_planner().plan(spec)
        planned = plan.apply(spec)
        result = self._rpc(
            "register_model", spec.to_rpc_payload(), plan.to_rpc_payload()
        )
        self._runtime_model_specs[spec.model_id] = planned
        return result

    def _topology_planner(self) -> Any:
        planner = getattr(
            self._llm_engine, "multi_model_topology_planner", None
        )
        if planner is None:
            raise RuntimeError(
                "The owning Engine did not provide a multi-model topology planner"
            )
        return planner

    def topology(
        self, worker_ranks: tuple[int, ...] = ()
    ) -> list[dict[str, Any]]:
        return self._rpc("inspect_multi_model_topology", worker_ranks)

    def models(self) -> list[dict[str, Any]]:
        """Return the unified Worker-local model registry on every rank."""
        return self._rpc("model_registry_metadata")

    def configure_stream(
        self,
        model_id: str,
        *,
        kind: str,
        sm_count: int | None = None,
    ) -> list[dict[str, Any]]:
        return self._rpc(
            "create_auxiliary_execution_stream", model_id, kind, sm_count
        )

    def execute(
        self, model_id: str, request: Any, *, profile_memory: bool = False
    ) -> Any:
        results = self._rpc(
            "execute_auxiliary_runner", model_id, request, profile_memory
        )
        outputs = [result for result in results if result is not None]
        if len(outputs) != 1:
            raise RuntimeError(
                "Expected exactly one output-owner rank, got "
                f"{len(outputs)} outputs from {len(results)} workers"
            )
        return outputs[0]

    def execute_timed(
        self, model_id: str, request: Any, *, profile_memory: bool = False
    ) -> dict[str, Any]:
        """Execute on workers and return timing only, never raw media tensors."""
        results = self._rpc(
            "execute_auxiliary_runner_timed", model_id, request, profile_memory
        )
        outputs = [result for result in results if result is not None]
        if len(outputs) != 1:
            raise RuntimeError(
                "Expected exactly one timed output-owner result, got "
                f"{len(outputs)} outputs from {len(results)} workers"
            )
        return outputs[0]

    def prepare(
        self, model_id: str, request: Any, *, profile_memory: bool = False
    ) -> list[dict[str, Any]]:
        return self._rpc(
            "prepare_auxiliary_runner", model_id, request, profile_memory
        )

    def start_async(
        self, model_id: str, request: Any, *, profile_memory: bool = False
    ) -> "WorkerHostedRunnerFuture":
        """Start Generator work now and return without waiting for completion."""
        job_id = uuid.uuid4().hex
        results = self._rpc(
            "start_auxiliary_runner",
            model_id,
            request,
            profile_memory,
            job_id,
        )
        if not results or not all(
            row.get("job_id") == job_id and row.get("started")
            for row in results
        ):
            raise RuntimeError(
                f"Auxiliary job {job_id!r} did not start on every worker: "
                f"{results!r}"
            )
        return WorkerHostedRunnerFuture(self, model_id, job_id)

    def release_prepared(self) -> list[list[str]]:
        return self._rpc("release_prepared_auxiliary_runners")

    def collect(
        self,
        model_id: str,
        *,
        job_id: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        results = self._rpc(
            "collect_auxiliary_runner", model_id, job_id, timeout
        )
        outputs = [result for result in results if result is not None]
        if len(outputs) != 1:
            raise RuntimeError(
                "Expected exactly one asynchronous output-owner rank, got "
                f"{len(outputs)} outputs from {len(results)} workers"
            )
        return outputs[0]

    def job_status(
        self, model_id: str, *, job_id: str | None = None
    ) -> list[dict[str, Any]]:
        return self._rpc("auxiliary_runner_job_status", model_id, job_id)

    def metadata(self, model_id: str) -> list[dict[str, Any]]:
        return self._rpc("auxiliary_runner_metadata", model_id)

    def resource_plan(self) -> list[dict[str, Any]]:
        return self._rpc("multi_model_resource_plan")

    def set_primary_demand(self, active: bool) -> list[dict[str, Any]]:
        return self._rpc("set_primary_model_demand", active)

    def close(self) -> list[list[str]]:
        result = self._rpc("close_auxiliary_runners")
        self._runtime_model_specs.clear()
        return result

    def _rpc(self, method: str, *args: Any) -> list[Any]:
        return self._llm_engine.collective_rpc(method, args=args)


class WorkerHostedRunnerFuture:
    """External handle for one Worker-hosted auxiliary-model request."""

    def __init__(
        self,
        client: WorkerHostedRunnerClient,
        model_id: str,
        job_id: str,
    ) -> None:
        self._client = client
        self.model_id = model_id
        self.job_id = job_id
        self._resolved = False
        self._result: Any = None

    def done(self) -> bool:
        if self._resolved:
            return True
        statuses = self._client.job_status(
            self.model_id, job_id=self.job_id
        )
        return bool(statuses) and all(row["done"] for row in statuses)

    def result(self, timeout: float | None = None) -> Any:
        if not self._resolved:
            self._result = self._client.collect(
                self.model_id,
                job_id=self.job_id,
                timeout=timeout,
            )
            self._resolved = True
        return self._result


class SchedulerModelFuture:
    """Synchronous handle for one scheduler-managed model request."""

    def __init__(
        self, model_id: str, request_id: str, future: Future[Any]
    ) -> None:
        self.model_id = model_id
        self.request_id = request_id
        self._future = future

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> Any:
        return self._future.result(timeout=timeout)


class AsyncWorkerHostedRunnerClient(WorkerHostedRunnerClient):
    """AsyncLLM frontend for runners owned by the same GPU workers.

    This client deliberately mirrors :class:`WorkerHostedRunnerClient`, but
    awaits ``AsyncLLM.collective_rpc`` instead of blocking the caller.  Keeping
    the two clients separate preserves the synchronous benchmark path while
    allowing a serving-style event loop to submit Reasoner and Generator work
    independently.
    """

    async def register(
        self, spec: ModelSpec
    ) -> list[dict[str, Any]]:
        plan = self._topology_planner().plan(spec)
        planned = plan.apply(spec)
        result = await self._rpc_async(
            "register_model", spec.to_rpc_payload(), plan.to_rpc_payload()
        )
        self._runtime_model_specs[planned.model_id] = planned
        return result

    async def submit(
        self,
        model_id: str,
        request: Any,
        *,
        profile_memory: bool = False,
    ) -> "AsyncModelRequestFuture":
        """Submit to AsyncLLM or a Worker-hosted model through one entry."""
        if model_id in self._runtime_model_specs:
            return await self.start_async(
                model_id, request, profile_memory=profile_memory
            )
        if model_id != self.primary_model_id:
            raise KeyError(f"Unknown model: {model_id!r}")
        if not isinstance(request, SchedulerModelRequest):
            raise TypeError(
                "Scheduler-managed models require SchedulerModelRequest"
            )
        dynamic = self._dynamic_arbitration_enabled()
        if dynamic:
            await self.set_primary_demand(True)

        async def run() -> Any:
            final_output = None
            try:
                async for output in self._model_frontend.generate(
                    prompt=request.prompt,
                    sampling_params=request.sampling_params,
                    request_id=request.request_id,
                    **dict(request.generate_kwargs),
                ):
                    final_output = output
                if final_output is None:
                    raise RuntimeError(
                        "Scheduler-managed request produced no output"
                    )
                return final_output
            finally:
                if dynamic:
                    await self.set_primary_demand(False)

        task = asyncio.create_task(run())
        return AsyncSchedulerModelFuture(model_id, request.request_id, task)

    async def topology(
        self, worker_ranks: tuple[int, ...] = ()
    ) -> list[dict[str, Any]]:
        return await self._rpc_async(
            "inspect_multi_model_topology", worker_ranks
        )

    async def configure_stream(
        self,
        model_id: str,
        *,
        kind: str,
        sm_count: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self._rpc_async(
            "create_auxiliary_execution_stream", model_id, kind, sm_count
        )

    async def execute(
        self, model_id: str, request: Any, *, profile_memory: bool = False
    ) -> Any:
        results = await self._rpc_async(
            "execute_auxiliary_runner", model_id, request, profile_memory
        )
        return self._one_output(results, "output-owner")

    async def execute_timed(
        self, model_id: str, request: Any, *, profile_memory: bool = False
    ) -> dict[str, Any]:
        results = await self._rpc_async(
            "execute_auxiliary_runner_timed",
            model_id,
            request,
            profile_memory,
        )
        return self._one_output(results, "timed output-owner")

    async def prepare(
        self, model_id: str, request: Any, *, profile_memory: bool = False
    ) -> list[dict[str, Any]]:
        return await self._rpc_async(
            "prepare_auxiliary_runner", model_id, request, profile_memory
        )

    async def release_prepared(self) -> list[list[str]]:
        return await self._rpc_async("release_prepared_auxiliary_runners")

    async def start_async(
        self, model_id: str, request: Any, *, profile_memory: bool = False
    ) -> "AsyncWorkerHostedRunnerFuture":
        job_id = uuid.uuid4().hex
        results = await self._rpc_async(
            "start_auxiliary_runner",
            model_id,
            request,
            profile_memory,
            job_id,
        )
        if not results or not all(
            row.get("job_id") == job_id and row.get("started")
            for row in results
        ):
            raise RuntimeError(
                f"Auxiliary job {job_id!r} did not start on every worker: "
                f"{results!r}"
            )
        return AsyncWorkerHostedRunnerFuture(self, model_id, job_id)

    async def collect(
        self,
        model_id: str,
        *,
        job_id: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        results = await self._rpc_async(
            "collect_auxiliary_runner", model_id, job_id, timeout
        )
        return self._one_output(results, "asynchronous output-owner")

    async def job_status(
        self, model_id: str, *, job_id: str | None = None
    ) -> list[dict[str, Any]]:
        return await self._rpc_async(
            "auxiliary_runner_job_status", model_id, job_id
        )

    async def metadata(self, model_id: str) -> list[dict[str, Any]]:
        return await self._rpc_async("auxiliary_runner_metadata", model_id)

    async def resource_plan(self) -> list[dict[str, Any]]:
        return await self._rpc_async("multi_model_resource_plan")

    async def set_primary_demand(
        self, active: bool
    ) -> list[dict[str, Any]]:
        return await self._rpc_async("set_primary_model_demand", active)

    async def close(self) -> list[list[str]]:
        result = await self._rpc_async("close_auxiliary_runners")
        self._runtime_model_specs.clear()
        return result

    async def _rpc_async(self, method: str, *args: Any) -> list[Any]:
        return await self._llm_engine.collective_rpc(method, args=args)

    @staticmethod
    def _one_output(results: list[Any], label: str) -> Any:
        outputs = [result for result in results if result is not None]
        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected exactly one {label} rank, got "
                f"{len(outputs)} outputs from {len(results)} workers"
            )
        return outputs[0]


class AsyncWorkerHostedRunnerFuture:
    """Awaitable handle for one Worker-hosted auxiliary-model request."""

    def __init__(
        self,
        client: AsyncWorkerHostedRunnerClient,
        model_id: str,
        job_id: str,
    ) -> None:
        self._client = client
        self.model_id = model_id
        self.job_id = job_id
        self._resolved = False
        self._result: Any = None

    async def done(self) -> bool:
        if self._resolved:
            return True
        statuses = await self._client.job_status(
            self.model_id, job_id=self.job_id
        )
        return bool(statuses) and all(row["done"] for row in statuses)

    async def result(self, timeout: float | None = None) -> Any:
        if not self._resolved:
            self._result = await self._client.collect(
                self.model_id,
                job_id=self.job_id,
                timeout=timeout,
            )
            self._resolved = True
        return self._result


class AsyncSchedulerModelFuture:
    """Async handle for one scheduler-managed model request."""

    def __init__(
        self, model_id: str, request_id: str, task: asyncio.Task[Any]
    ) -> None:
        self.model_id = model_id
        self.request_id = request_id
        self._task = task

    async def done(self) -> bool:
        return self._task.done()

    async def result(self, timeout: float | None = None) -> Any:
        if timeout is None:
            return await self._task
        return await asyncio.wait_for(self._task, timeout=timeout)


ModelRequestFuture = WorkerHostedRunnerFuture | SchedulerModelFuture
AsyncModelRequestFuture = (
    AsyncWorkerHostedRunnerFuture | AsyncSchedulerModelFuture
)
