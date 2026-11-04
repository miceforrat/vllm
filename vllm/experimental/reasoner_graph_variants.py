# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph variant registry for a Worker-hosted primary model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ReasonerGraphVariant:
    """One resource-specific executable variant of the same model graph."""

    name: str
    stream: torch.cuda.Stream
    # 每个stream资源变体一个graph manager,可以存多个bs的graph
    manager: Any
    piecewise_entry_sets: tuple[tuple[Any, dict[Any, Any]], ...] = ()
    capture_count: int = 0


class ReasonerGraphVariantRegistry:
    """Own and select graph managers captured for different GPU resources.

    The registry does not imply that CUDA graph executables can be shared:
    every variant remains captured on its own stream/context.  It provides a
    single Worker-owned lifecycle and dispatch surface for those executables.
    """

    def __init__(self) -> None:
        self._variants: dict[str, ReasonerGraphVariant] = {}
        self.active_name: str | None = None

    def register(
        self,
        name: str,
        stream: torch.cuda.Stream,
        manager: Any,
        *,
        piecewise_entry_sets: tuple[tuple[Any, dict[Any, Any]], ...] = (),
        capture_count: int | None = None,
    ) -> None:
        if name in self._variants:
            raise RuntimeError(f"Reasoner graph variant {name!r} already exists")
        if capture_count is None:
            capture_count = len(manager.graphs)
        self._variants[name] = ReasonerGraphVariant(
            name,
            stream,
            manager,
            piecewise_entry_sets,
            capture_count,
        )

    def activate(self, runner: Any, name: str) -> ReasonerGraphVariant:
        try:
            variant = self._variants[name]
        except KeyError as exc:
            raise RuntimeError(
                f"Reasoner graph variant {name!r} was not captured; "
                f"available={sorted(self._variants)}"
            ) from exc
        runner.cudagraph_manager = variant.manager
        for wrapper, entries in variant.piecewise_entry_sets:
            wrapper.concrete_cudagraph_entries = entries
        self.active_name = name
        return variant

    def get(self, name: str) -> ReasonerGraphVariant | None:
        return self._variants.get(name)

    def counts(self) -> dict[str, int]:
        return {
            name: variant.capture_count
            for name, variant in self._variants.items()
        }

    def clear(self) -> None:
        self._variants.clear()
        self.active_name = None
