# SPDX-License-Identifier: Apache-2.0
"""Worker-local materialization of Engine-planned communication groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from vllm.experimental.model_instance_registry import CommunicationGroupSpec


@dataclass
class _CommunicatorSlot:
    spec: CommunicationGroupSpec
    group: Any
    owners: set[str] = field(default_factory=set)


class CommunicatorRegistry:
    """Create, reuse and release communicators inside one Worker process."""

    def __init__(
        self,
        group_factory: Callable[[CommunicationGroupSpec], Any] | None = None,
    ) -> None:
        self._group_factory = group_factory or self._create_group
        self._slots: dict[str, _CommunicatorSlot] = {}

    @staticmethod
    def _create_group(spec: CommunicationGroupSpec) -> Any:
        from vllm.distributed.parallel_state import (
            get_tp_group,
            get_world_group,
            init_model_parallel_group,
        )

        world = get_world_group()
        default_tp = get_tp_group()
        backend = spec.backend
        if backend is None:
            import torch

            backend = str(torch.distributed.get_backend(default_tp.device_group))
        return init_model_parallel_group(
            group_ranks=[list(spec.ranks)],
            local_rank=int(world.local_rank),
            backend=backend,
            group_name=spec.group_id,
            use_device_communicator=spec.use_device_communicator,
        )

    def acquire(self, spec: CommunicationGroupSpec, owner: str) -> Any:
        existing = self._slots.get(spec.group_id)
        if existing is not None:
            if existing.spec != spec:
                raise ValueError(
                    f"Communication group {spec.group_id!r} was already created "
                    "with a different topology"
                )
            existing.owners.add(owner)
            return existing.group

        group = self._group_factory(spec)
        self._slots[spec.group_id] = _CommunicatorSlot(
            spec=spec, group=group, owners={owner}
        )
        return group

    def release(self, owner: str, group_ids: tuple[str, ...]) -> None:
        for group_id in reversed(group_ids):
            slot = self._slots[group_id]
            slot.owners.discard(owner)
            if not slot.owners:
                slot.group.destroy()
                del self._slots[group_id]

    def get(self, group_id: str) -> Any:
        return self._slots[group_id].group

    def owners(self, group_id: str) -> frozenset[str]:
        return frozenset(self._slots[group_id].owners)
