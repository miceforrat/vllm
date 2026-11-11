# SPDX-License-Identifier: Apache-2.0
"""Engine-planned topology for Worker-hosted model instances."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from vllm.experimental.model_instance_registry import (
    CommunicationGroupSpec,
    ModelSpec,
)


@dataclass(frozen=True)
class TopologyPlan:
    """对拓扑规划的描述，从模型视角转到部署视角"""

    model_id: str
    worker_ranks: tuple[int, ...]
    communication_groups: tuple[CommunicationGroupSpec, ...]

    @classmethod
    def from_rpc_payload(
        cls, payload: Mapping[str, Any] | "TopologyPlan"
    ) -> "TopologyPlan":
        if isinstance(payload, cls):
            return payload
        values = dict(payload)
        values["worker_ranks"] = tuple(
            int(rank) for rank in values["worker_ranks"]
        )
        values["communication_groups"] = tuple(
            CommunicationGroupSpec.from_rpc_payload(group)
            for group in values.get("communication_groups", ())
        )
        return cls(**values)

    def to_rpc_payload(self) -> dict[str, Any]:
        return asdict(self)

    def apply(self, spec: ModelSpec) -> ModelSpec:
        if spec.model_id != self.model_id:
            raise ValueError(
                f"Topology plan for {self.model_id!r} cannot be applied to "
                f"model {spec.model_id!r}"
            )
        return spec.with_placement(
            worker_ranks=self.worker_ranks,
            communication_groups=self.communication_groups,
        )


class TopologyPlanner:
    """Engine-owned planner for Worker-hosted model placement.

    The first implementation deliberately preserves the verified homogeneous
    TP gate.  Keeping this policy behind an Engine-owned object avoids making
    frontend clients or Workers reverse-read the Engine's parallel config.
    """

    def __init__(self, parallel_config: Any) -> None:
        self._tp_size = int(parallel_config.tensor_parallel_size)
        self._pp_size = int(parallel_config.pipeline_parallel_size)

    def plan(self, spec: ModelSpec) -> TopologyPlan:
        if spec.backend == "vllm":
            # Native vLLM already materializes its TP/PP communicators. The
            # Engine plan still makes placement explicit, but must not create
            # a second set of equivalent groups in the Worker registry.
            ranks = tuple(range(self._tp_size * self._pp_size))
            if spec.worker_ranks and tuple(spec.worker_ranks) != ranks:
                raise ValueError(
                    "Requested vLLM placement conflicts with Engine topology: "
                    f"requested={spec.worker_ranks}, engine={ranks}"
                )
            return TopologyPlan(
                model_id=spec.model_id,
                worker_ranks=ranks,
                communication_groups=spec.communication_groups,
            )
        if self._pp_size != 1:
            raise ValueError(
                "The homogeneous auxiliary-runner Gate currently requires PP=1, "
                f"got PP={self._pp_size}"
            )
        ranks = tuple(range(self._tp_size))
        if spec.worker_ranks and tuple(spec.worker_ranks) != ranks:
            raise ValueError(
                "Requested placement conflicts with Engine TP topology: "
                f"requested={spec.worker_ranks}, engine_tp={ranks}"
            )
        if spec.communication_groups:
            groups = spec.communication_groups
        else:
            groups = (
                CommunicationGroupSpec(
                    group_id=f"{spec.model_id}-tp",
                    parallel_dimension="tp",
                    ranks=ranks,
                ),
            )
        return TopologyPlan(
            model_id=spec.model_id,
            worker_ranks=ranks,
            communication_groups=groups,
        )
