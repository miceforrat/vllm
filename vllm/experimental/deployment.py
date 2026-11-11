# SPDX-License-Identifier: Apache-2.0
"""Engine 侧多模型部署计划的数据契约。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from vllm.experimental.model_instance_registry import ModelSpec
from vllm.experimental.worker.resource_desc import ResourceDescription


@dataclass(frozen=True)
class ModelDeployment:
    """一个模型实例及其规范化物理资源需求。
    这个描述是可以总体而非单worker的，详见model spec
    """
    spec: ModelSpec
    resources: ResourceDescription

    @classmethod
    def from_rpc_payload(
        cls, payload: Mapping[str, Any] | ModelDeployment
    ) -> ModelDeployment:
        if isinstance(payload, cls):
            return payload
        return cls(
            spec=ModelSpec.from_rpc_payload(payload["spec"]),
            resources=ResourceDescription.from_rpc_payload(
                payload["resources"]
            ),
        )

    def to_rpc_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EngineDeploymentPlan:
    """Engine 对一次 Worker 启动所做的完整部署规划。

    本对象只描述模型、放置和资源需求。物理资源仍由各 Worker 的
    ResourceManager 创建，逐 step 的推进顺序仍由 Worker 调度器决定。

    这里是整体的部署计划，所以放了多个model的deployment
    """

    deployments: tuple[ModelDeployment, ...]
    default_model_id: str
    revision: int = 0

    def __post_init__(self) -> None:
        """校验部署计划"""
        if not self.deployments:
            raise ValueError("An Engine deployment plan cannot be empty")
        model_ids = [deployment.spec.model_id for deployment in self.deployments]
        # 每个模型的id唯一
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Deployment model ids must be unique")
        # 默认模型id必须存在于计划中
        if self.default_model_id not in set(model_ids):
            raise ValueError(
                f"Unknown default deployment model {self.default_model_id!r}"
            )
        
        if self.revision < 0:
            raise ValueError("Deployment plan revision cannot be negative")

        # 收集用到gc的执行变体，按照partition_group分组
        # 目前只支持恰好两个成员
        partition_members: dict[str, list[tuple[str, str]]] = {}
        for deployment in self.deployments:
            for variant in deployment.resources.execution_variants:
                stream = variant.stream
                if stream.kind != "green_context":
                    continue
                assert stream.partition_group is not None
                partition_members.setdefault(stream.partition_group, []).append(
                    (deployment.spec.model_id, variant.name)
                )
        invalid = {
            group: members
            for group, members in partition_members.items()
            if len(members) != 2
        }
        if invalid:
            raise ValueError(
                "Each Green Context partition group requires exactly two "
                f"members, got {invalid!r}"
            )

    @classmethod
    def from_rpc_payload(
        cls, payload: Mapping[str, Any] | EngineDeploymentPlan
    ) -> EngineDeploymentPlan:
        if isinstance(payload, cls):
            return payload
        values = dict(payload)
        values["deployments"] = tuple(
            ModelDeployment.from_rpc_payload(deployment)
            for deployment in values.get("deployments", ())
        )
        return cls(**values)

    def to_rpc_payload(self) -> dict[str, Any]:
        return asdict(self)

