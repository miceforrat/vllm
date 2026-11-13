# SPDX-License-Identifier: Apache-2.0
"""Engine 规划、Worker 实例化的共享通信资源契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class CommunicationGroupDescription:
    """一个跨 rank 通信域的完整、可序列化描述。"""

    group_id: str
    parallel_dimension: str
    ranks: tuple[int, ...]
    backend: str | None = None
    use_device_communicator: bool = True

    @classmethod
    def from_rpc_payload(
        cls,
        payload: Mapping[str, Any] | CommunicationGroupDescription | Any,
    ) -> CommunicationGroupDescription:
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, Mapping):
            values = dict(payload)
        else:
            values = {
                "group_id": payload.group_id,
                "parallel_dimension": payload.parallel_dimension,
                "ranks": payload.ranks,
                "backend": payload.backend,
                "use_device_communicator": payload.use_device_communicator,
            }
        values["ranks"] = tuple(int(rank) for rank in values["ranks"])
        return cls(**values)
