# SPDX-License-Identifier: Apache-2.0
"""EngineCore-owned execution-slice scheduling and admission."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from vllm.experimental.execution_slice import ExecutionSliceDescriptor


@dataclass(frozen=True)
class ScheduledExecutionSlice:
    """EngineCore 为一次模型推进下发的调度结果。

    这个类和``ExecUnit``不同，``ExecUnit`` 是 Worker 中长期存活的模型实例；本类只描述
    一次 batch、denoising step 或 VAE block 的逻辑资源决策。
    """

    model_id: str
    request_id: str
    descriptor: ExecutionSliceDescriptor
    # 要求采用什么资源执行
    resource_variant: str
    #  EngineCore 给每次资源调度决策分配的全局单调递增编号（仅用于日志观测）
    execution_epoch: int
    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any] | ScheduledExecutionSlice
    ) -> ScheduledExecutionSlice:
        if isinstance(payload, cls):
            return payload
        return cls(
            model_id=str(payload["model_id"]),
            request_id=str(payload["request_id"]),
            descriptor=ExecutionSliceDescriptor.from_payload(
                payload["descriptor"]
            ),
            resource_variant=str(payload["resource_variant"]),
            execution_epoch=int(payload["execution_epoch"]),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "request_id": self.request_id,
            "descriptor": self.descriptor.to_payload(),
            "resource_variant": self.resource_variant,
            "execution_epoch": self.execution_epoch,
        }


class ExecutionSliceScheduler:
    """根据跨模型 demand 为某一个 execution slice 选择资源变体。"""

    def __init__(self) -> None:
        self._primary_model_id: str | None = None
        self._model_demand: dict[str, int] = {}
        self._model_variants: dict[str, tuple[str, str | None]] = {}
        self._next_execution_epoch = 0
        self._revision = 0
        self._lock = threading.RLock()

    def configure_model(
        self,
        model_id: str,
        *,
        full_variant: str,
        shared_variant: str | None,
        primary: bool = False,
    ) -> None:
        """Register Engine-planned logical variants for one model."""
        with self._lock:
            self._model_variants[str(model_id)] = (
                str(full_variant),
                None if shared_variant is None else str(shared_variant),
            )
            if primary:
                self._primary_model_id = str(model_id)

    @property
    def has_resource_variants(self) -> bool:
        return any(shared is not None for _, shared in self._model_variants.values())

    @property
    def primary_model_id(self) -> str | None:
        return self._primary_model_id

    def set_model_demand(self, model_id: str, active: bool) -> dict[str, Any]:
        """记录请求需求
            由于缺乏进一步的调度策略，所以这里只需要管“有没有请求”即可"""
        with self._lock:
            model_id = str(model_id)
            count = self._model_demand.get(model_id, 0)
            if active:
                count += 1
            elif count <= 0:
                raise RuntimeError(
                    f"Model demand is not active for {model_id!r}"
                )
            else:
                count -= 1
            if count:
                self._model_demand[model_id] = count
            else:
                self._model_demand.pop(model_id, None)
            self._revision += 1
            return self.snapshot()

    def validate_requested_variant(
        self, model_id: str, requested_variant: str | None
    ) -> None:
        """在请求进入 Worker 前校验显式资源变体。"""
        if requested_variant in (None, "auto"):
            return
        with self._lock:
            model_id = str(model_id)
            full_variant, shared_variant = self._model_variants.get(
                model_id, ("generator_full", "shared")
            )
            allowed = {full_variant}
            if shared_variant is not None:
                allowed.add(shared_variant)
            if str(requested_variant) not in allowed:
                raise ValueError(
                    f"Unknown resource variant {requested_variant!r} for "
                    f"model {model_id!r}; expected one of "
                    f"{sorted(allowed)!r} or 'auto'"
                )

    def schedule_slice(
        self,
        model_id: str,
        request_id: str,
        descriptor: ExecutionSliceDescriptor,
    ) -> ScheduledExecutionSlice:
        """根据slice选择一个资源变体
            目前的决策依据只有“是否存在多个请求”这一条
            未来考虑收束约束：SLO、资源利用情况
            同时未来建模还要考虑sm scaling、干扰等
            """
        with self._lock:
            model_id = str(model_id)
            full_variant, shared_variant = self._model_variants.get(
                model_id, ("generator_full", "shared")
            )
            contended = any(
                other != model_id and count > 0
                for other, count in self._model_demand.items()
            )
            variant = (
                shared_variant
                if contended and shared_variant is not None
                else full_variant
            )
            requested = descriptor.metadata.get("requested_variant")
            if requested not in (None, "auto"):
                requested = str(requested)
                self.validate_requested_variant(model_id, requested)
                variant = requested
            self._next_execution_epoch += 1
            return ScheduledExecutionSlice(
                model_id=model_id,
                request_id=str(request_id),
                descriptor=descriptor,
                resource_variant=variant,
                execution_epoch=self._next_execution_epoch,
            )

    def schedule_model_batch(
        self, model_id: str, request_ids: Sequence[str]
    ) -> ScheduledExecutionSlice:
        """Schedule one native model batch as an execution slice."""
        joined = ",".join(str(item) for item in request_ids)
        return self.schedule_slice(
            model_id,
            joined or f"batch-{self._next_execution_epoch + 1}",
            ExecutionSliceDescriptor(
                sequence_no=self._next_execution_epoch + 1,
                tags=frozenset(("model:batch",)),
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "revision": self._revision,
                "primary_model_id": self._primary_model_id,
                "model_demand": dict(self._model_demand),
                "model_variants": {
                    model_id: {
                        "full": variants[0],
                        "shared": variants[1],
                    }
                    for model_id, variants in self._model_variants.items()
                },
                "next_execution_epoch": self._next_execution_epoch,
            }
