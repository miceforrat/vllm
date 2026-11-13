# SPDX-License-Identifier: Apache-2.0
"""Worker-owned model specifications and runtime instance registry."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from vllm.experimental.communication import CommunicationGroupDescription

# 仅供 archived 代码兼容；现役代码统一使用 CommunicationGroupDescription。
CommunicationGroupSpec = CommunicationGroupDescription

@dataclass(frozen=True)
class ModelSpec:
    """ModelSpec是跨越多个rank的部署描述

    ``capabilities`` 描述了一个模型可以执行的任务——这样一个模型可以用于服务多种请求
    """

    model_id: str
    model_path: str
    backend: str
    # 该模型能处理的请求类型集合
    capabilities: tuple[str, ...] = ()
    default_task: str | None = None

    # 该模型放在哪些rank上
    worker_ranks: tuple[int, ...] = ()
    # 该模型使用的通信组
    communication_groups: tuple[CommunicationGroupDescription, ...] = ()
    load_format: str = "default"
    # 仅对特定后端生效的配置项
    backend_options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rpc_payload(cls, payload: Mapping[str, Any]) -> "ModelSpec":
        values = dict(payload)
        capabilities = values.get("capabilities", ())
        values["capabilities"] = tuple(str(item) for item in capabilities)
        values["worker_ranks"] = tuple(
            int(rank) for rank in values.get("worker_ranks", ())
        )
        values["communication_groups"] = tuple(
            CommunicationGroupDescription.from_rpc_payload(group)
            for group in values.get("communication_groups", ())
        )
        return cls(**values)

    def to_rpc_payload(self) -> dict[str, Any]:
        return asdict(self)

    def with_placement(
        self,
        *,
        worker_ranks: tuple[int, ...],
        communication_groups: tuple[CommunicationGroupDescription, ...],
    ) -> "ModelSpec":
        return replace(
            self,
            worker_ranks=worker_ranks,
            communication_groups=communication_groups,
        )


# 这个类负责管理模型的整体spec
# 由 EngineCore 唯一持有，负责模型注册、查询和请求路由。
class ModelRegistry:
    """Engine-owned registry of every deployed model specification.

    该注册表只保存 Engine 级部署意图；Worker 本地 Runner 的生命周期由
    对应的组件管理。启动模型与运行时追加模型使用同一接口，
    ``default_model_id`` 仅用于兼容默认路由，不表示特殊生命周期所有权。
    """

    def __init__(self, *, default_model_id: str | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._default_model_id = default_model_id
        self._lock = threading.RLock()

    @property
    def default_model_id(self) -> str:
        with self._lock:
            if self._default_model_id is None:
                raise RuntimeError("The Engine ModelRegistry has no default model")
            return self._default_model_id

    def register(self, spec: ModelSpec, *, make_default: bool = False) -> None:
        """注册全局模型"""
        with self._lock:
            if spec.model_id in self._specs:
                raise ValueError(f"Model {spec.model_id!r} is already registered")
            self._specs[spec.model_id] = spec
            if make_default or self._default_model_id is None:
                self._default_model_id = spec.model_id

    def get(self, model_id: str) -> ModelSpec:
        with self._lock:
            try:
                return self._specs[model_id]
            except KeyError as exc:
                raise KeyError(f"Unknown model: {model_id!r}") from exc

    def remove(self, model_id: str) -> ModelSpec:
        with self._lock:
            if model_id == self._default_model_id:
                raise ValueError("Cannot remove the compatibility default model")
            try:
                return self._specs.pop(model_id)
            except KeyError as exc:
                raise KeyError(f"Unknown model: {model_id!r}") from exc

    def values(self) -> tuple[ModelSpec, ...]:
        with self._lock:
            return tuple(self._specs.values())

    def __contains__(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._specs


def create_engine_core_model_registry(vllm_config: Any) -> ModelRegistry:
    """在 EngineCore 内规划启动部署，并建立唯一模型注册表。"""
    from vllm.experimental.deployment import (
        EngineDeploymentPlan,
        ModelDeployment,
    )
    from vllm.experimental.worker.resource_desc import (
        ExecutionVariantDescription,
        MemoryResourceDescription,
        ResourceDescription,
        StreamResourceDescription,
    )
    from vllm.experimental.multi_model_topology import TopologyPlanner

    topology_planner = TopologyPlanner(vllm_config.parallel_config)

    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict):
        additional = {}
        vllm_config.additional_config = additional
    runtime = additional.setdefault("multi_model_runtime", {})
    plan_payload = runtime.get("deployment_plan")
    if plan_payload is None:
        model_id = str(runtime.get("default_model_id", "reasoner"))
        capabilities = ("text_generation",)
        deployment_plan = EngineDeploymentPlan(
            deployments=(
                ModelDeployment(
                    spec=ModelSpec(
                        model_id=model_id,
                        model_path=str(vllm_config.model_config.model),
                        backend="vllm",
                        capabilities=capabilities,
                        default_task=capabilities[0],
                    ),
                    resources=ResourceDescription(
                        memory=MemoryResourceDescription(kind="default"),
                        execution_variants=(
                            ExecutionVariantDescription(
                                name="default",
                                stream=StreamResourceDescription(
                                    kind="default"
                                ),
                            ),
                        ),
                        default_variant="default",
                    ),
                ),
            ),
            default_model_id=model_id,
        )
    else:
        deployment_plan = EngineDeploymentPlan.from_rpc_payload(plan_payload)

    native = [
        deployment
        for deployment in deployment_plan.deployments
        if deployment.spec.backend == "vllm"
    ]
    if len(native) != 1:
        raise ValueError(
            "The compatibility Engine currently requires exactly one "
            f"scheduler-managed vllm deployment, got {len(native)}"
        )
    planned_deployments: list[ModelDeployment] = []
    for deployment in deployment_plan.deployments:
        spec = deployment.spec
        if spec.backend == "vllm":
            # EngineArgs 可能已把 Hugging Face 模型 ID 解析为本地 snapshot
            # 路径；scheduler-managed Runner 必须复用这一权威解析结果。
            spec = replace(
                spec, model_path=str(vllm_config.model_config.model)
            )
        planned_spec = topology_planner.plan(spec).apply(spec)
        planned_groups = tuple(
            CommunicationGroupDescription.from_rpc_payload(group)
            for group in planned_spec.communication_groups
        )
        resources = replace(
            deployment.resources,
            communication_groups=planned_groups,
        )
        planned_deployments.append(
            ModelDeployment(spec=planned_spec, resources=resources)
        )

    planned_plan = EngineDeploymentPlan(
        deployments=tuple(planned_deployments),
        default_model_id=deployment_plan.default_model_id,
        revision=deployment_plan.revision,
    )
    registry = ModelRegistry(default_model_id=planned_plan.default_model_id)
    for deployment in planned_plan.deployments:
        registry.register(
            deployment.spec,
            make_default=(
                deployment.spec.model_id == planned_plan.default_model_id
            ),
        )

    # deployment_plan 是新接口；另两项是当前 Worker/旧分支的兼容视图。
    runtime["deployment_plan"] = planned_plan.to_rpc_payload()
    runtime["default_model_id"] = planned_plan.default_model_id
    runtime["startup_models"] = [
        deployment.spec.to_rpc_payload()
        for deployment in planned_plan.deployments
    ]
    runtime["startup_resources"] = {
        deployment.spec.model_id: deployment.resources.to_rpc_payload()
        for deployment in planned_plan.deployments
    }
    return registry
