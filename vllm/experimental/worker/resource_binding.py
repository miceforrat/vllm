# SPDX-License-Identifier: Apache-2.0
"""MiniContainer 与不同 Runner 后端之间的资源接线实现。"""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import Any, Protocol

from vllm.experimental.worker.binding import ResourceAssignment


class _SingleRankGroupView:
    """把 vLLM 的单 rank PP group 暴露为 Omni 禁用并行维度。"""

    world_size = 1
    rank_in_group = 0
    ulysses_world_size = 1
    ulysses_rank = 0
    ring_world_size = 1
    ring_rank = 0
    allgather_world_size = 1
    allgather_rank = 0
    is_first_rank = True
    is_last_rank = True

    def __init__(self, coordinator: Any) -> None:
        self.coordinator = coordinator

    @property
    def ulysses_group(self) -> Any:
        return self.coordinator.device_group

    @property
    def ring_group(self) -> Any:
        return self.coordinator.device_group

    @property
    def allgather_group(self) -> Any:
        return self.coordinator.device_group

    def __getattr__(self, name: str) -> Any:
        return getattr(self.coordinator, name)


class _OmniParallelGroupProvider:
    """补全 Omni 所需的 WORLD、TP 和禁用并行维度视图。"""

    _SINGLETON_DIMENSIONS = {
        "sp",
        "pp",
        "cfg",
        "dp",
        "fs",
        "hsdp_replicate",
    }

    def __init__(self, binding: Any) -> None:
        from vllm.distributed.parallel_state import get_pp_group

        self._binding = binding
        self._singleton = _SingleRankGroupView(get_pp_group())

    def get(self, parallel_dimension: str) -> Any | None:
        group = self._binding.get(parallel_dimension)
        if group is not None:
            return group
        if parallel_dimension == "world":
            from vllm.distributed.parallel_state import get_world_group

            return get_world_group()
        if parallel_dimension == "tp":
            from vllm.distributed.parallel_state import get_tp_group

            return get_tp_group()
        if parallel_dimension in self._SINGLETON_DIMENSIONS:
            return self._singleton
        return None

    @contextmanager
    def activate(self):
        from vllm_omni.diffusion.distributed.parallel_state import (
            use_parallel_group_provider,
        )

        with self._binding.activate():
            with use_parallel_group_provider(self):
                yield self

    def communication_context(self):
        return self.activate()


class RunnerResourceBinder(Protocol):
    """把既有物理资源接到一个已经创建的 Runner。"""

    def initialize(
        self,
        assignment: ResourceAssignment,
        *,
        load_format: str = "default",
    ) -> None: ...

    def activate(
        self,
        assignment: ResourceAssignment,
        *,
        resource_transition: bool = True,
    ) -> AbstractContextManager[Any]: ...

    def close(self) -> None: ...


class VllmRunnerResourceBinder:
    """原生 vLLM Runner 的资源接线实现。"""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def initialize(
        self,
        assignment: ResourceAssignment,
        *,
        load_format: str = "default",
    ) -> None:
        """向原生 Worker 生命周期暴露模型级显存 Binding。"""
        del load_format
        self._runner._vllm_memory_binding = assignment.memory

    @contextmanager
    def activate(
        self,
        assignment: ResourceAssignment,
        *,
        resource_transition: bool = True,
    ):
        if assignment.graph is not None:
            assignment.graph.validate_stream(assignment.streams)
        with ExitStack() as stack:
            # stack用来动态组合多个 context manager，并保证按相反顺序退出。
            stack.enter_context(assignment.communicator.activate())
            stack.enter_context(assignment.memory.activate())
            stack.enter_context(
                assignment.streams.activate(
                    self._runner,
                    synchronize_transition=resource_transition,
                )
            )
            if assignment.graph is not None and resource_transition:
                # 如果decode开启graph执行，还需要额外切换graph环境
                assignment.graph.activate(self._runner)
            yield

    def close(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None:
            shutdown = getattr(runner, "shutdown", None)
            if shutdown is not None:
                shutdown()


class OmniRunnerResourceBinder:
    """Omni Runner 的资源接线实现；不接受 CUDA Graph binding。"""

    def __init__(
        self,
        runner: Any,
        *,
        diffusion_vllm_config: Any,
        omni_diffusion_config: Any,
    ) -> None:
        self._runner = runner
        self._diffusion_vllm_config = diffusion_vllm_config
        self._omni_diffusion_config = omni_diffusion_config
        self._communicator_provider: Any | None = None
        self._base_runtime_providers: Any | None = None

    def initialize(
        self,
        assignment: ResourceAssignment,
        *,
        load_format: str,
    ) -> None:
        """创建并接入 Omni runtime providers；不加载模型。"""
        del load_format
        from vllm_omni.diffusion.worker.runtime_providers import (
            DiffusionRuntimeProviders,
        )

        communicator = _OmniParallelGroupProvider(
            assignment.communicator
        )
        self._communicator_provider = communicator
        providers = DiffusionRuntimeProviders(
            memory=assignment.memory,
            stream=assignment.streams,
            communicator=communicator,
        )
        self._base_runtime_providers = providers
        self._runner._container_runtime_providers = providers

    def _providers_for_assignment(
        self, assignment: ResourceAssignment
    ) -> Any:
        """为本次资源变体派生 Omni provider 视图。"""
        from vllm_omni.diffusion.worker.runtime_providers import (
            DiffusionRuntimeProviders,
        )

        base = self._base_runtime_providers
        if base is None:
            raise RuntimeError("Omni runtime providers are not initialized")
        return DiffusionRuntimeProviders(
            memory=assignment.memory,
            stream=assignment.streams,
            communicator=base.communicator,
            phase=base.phase,
        )

    @contextmanager
    def activate(
        self,
        assignment: ResourceAssignment,
        *,
        resource_transition: bool = True,
    ):
        if assignment.graph is not None:
            raise ValueError("Omni Runner does not accept a CUDA Graph binding")

        from vllm.config import set_current_vllm_config
        from vllm_omni.diffusion.forward_context import set_forward_context

        runner = self._runner
        if runner is None:
            raise RuntimeError("Omni Runner binding is closed")
        selected_providers = self._providers_for_assignment(assignment)
        previous_runtime = getattr(runner, "runtime_providers", None)
        previous_container = getattr(
            runner, "_container_runtime_providers", None
        )
        # DiffusionModelRunner 会在 execute_model 内再次进入 provider；必须让
        # 该内层上下文看到本次 Container 选择的 stream，而不是初始化时的
        # 默认变体。
        runner.runtime_providers = selected_providers
        runner._container_runtime_providers = selected_providers

        # 激活绑定
        try:
            with ExitStack() as stack:
                # stack用来动态组合多个 context manager，并保证按相反顺序退出。
                communicator = self._communicator_provider
                if communicator is None:
                    raise RuntimeError(
                        "Omni communicator provider is not initialized"
                    )
                stack.enter_context(communicator.activate())
                stack.enter_context(assignment.memory.activate())
                stack.enter_context(
                    assignment.streams.activate(
                        runner,
                        synchronize_transition=resource_transition,
                    )
                )
                # Omni 执行所需的软件运行时上下文，不是物理资源
                stack.enter_context(
                    set_current_vllm_config(self._diffusion_vllm_config)
                )
                stack.enter_context(
                    set_forward_context(
                        vllm_config=self._diffusion_vllm_config,
                        omni_diffusion_config=self._omni_diffusion_config,
                    )
                )
                yield
        finally:
            runner._container_runtime_providers = (
                self._base_runtime_providers
                if previous_container is None
                else previous_container
            )
            runner.runtime_providers = (
                self._base_runtime_providers
                if previous_runtime is None
                else previous_runtime
            )

    def close(self) -> None:
        runner, self._runner = self._runner, None
        if runner is None:
            return
        close = getattr(runner, "close", None)
        if close is None:
            close = getattr(runner, "shutdown", None)
        if close is not None:
            close()


class RunnerResourceBinderFactory:
    """仅在 Container 创建时解析一次后端差异。"""

    @staticmethod
    def create_binder(
        exec_unit: Any,
    ) -> RunnerResourceBinder:
        backend = exec_unit.spec.backend
        if backend == "vllm":
            return VllmRunnerResourceBinder(exec_unit.runner)
        elif backend == "vllm_omni":
            return OmniRunnerResourceBinder(
                exec_unit.runner,
                diffusion_vllm_config=exec_unit.diffusion_vllm_config,
                omni_diffusion_config=exec_unit.omni_diffusion_config,
            )
        raise ValueError(
            f"No RunnerResourceBinder is registered for {backend!r}"
        )

    @staticmethod
    def create_container(
        exec_unit: Any,
        *,
        initialization_assignment: ResourceAssignment | None = None,
    ) -> MiniContainer:
        """兼容旧测试的非托管 Container 构造入口。"""
        binder = RunnerResourceBinderFactory.create_binder(exec_unit)
        if initialization_assignment is not None:
            binder.initialize(
                initialization_assignment,
                load_format=getattr(exec_unit.spec, "load_format", "default"),
            )
        return MiniContainer(exec_unit, binder)


# 兼容原有导入路径；新代码应直接从 mini_container 导入。
from vllm.experimental.worker.mini_container import MiniContainer
