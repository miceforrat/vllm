# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph variant registry for a Worker-hosted primary model."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import torch


@dataclass(frozen=True)
class ModelGraphVariant:
    """One resource-specific executable variant of the same model graph."""

    name: str
    stream: torch.cuda.Stream
    # 每个stream资源变体一个graph manager,可以存多个bs的graph
    manager: Any
    piecewise_entry_sets: tuple[tuple[Any, dict[Any, Any]], ...] = ()
    capture_count: int = 0


class ModelGraphVariantManager:
    """Capture, own and select equivalent resource-specific graph variants.

    The registry does not imply that CUDA graph executables can be shared:
    every variant remains captured on its own stream/context.  It provides a
    single Worker-owned lifecycle and dispatch surface for those executables.
    """

    def __init__(self) -> None:
        self._variants: dict[str, ModelGraphVariant] = {}
        self.active_name: str | None = None
        self._capture_handler: Callable[..., int] | None = None

    def __getattr__(self, name: str) -> Any:
        """Expose the active native manager through one stable facade."""
        active_name = self.active_name
        if active_name is None:
            raise AttributeError(name)
        return getattr(self._variants[active_name].manager, name)

    def set_capture_handler(self, handler: Callable[..., int]) -> None:
        """Install the Worker-specific multi-variant capture implementation."""
        if self._capture_handler is not None:
            raise RuntimeError("A graph variant capture handler is already set")
        self._capture_handler = handler

    def clear_capture_handler(self) -> None:
        self._capture_handler = None

    def capture_all(
        self,
        runner: Any,
        capture_native: Callable[..., int],
        **kwargs: Any,
    ) -> int:
        """Runner-facing provider API used by ``GPUModelRunner.capture_model``."""
        if self._capture_handler is None:
            return capture_native(**kwargs)
        return self._capture_handler(
            runner,
            capture_native,
            **kwargs,
        )

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
        self._variants[name] = ModelGraphVariant(
            name,
            stream,
            manager,
            piecewise_entry_sets,
            capture_count,
        )

    def activate(self, runner: Any, name: str) -> ModelGraphVariant:
        try:
            variant = self._variants[name]
        except KeyError as exc:
            raise RuntimeError(
                f"Reasoner graph variant {name!r} was not captured; "
                f"available={sorted(self._variants)}"
            ) from exc
        # Keep one stable manager object installed on the Runner. Native graph
        # managers remain per-resource executable stores behind this facade.
        runner.cudagraph_manager = self
        for wrapper, entries in variant.piecewise_entry_sets:
            wrapper.concrete_cudagraph_entries = entries
        self.active_name = name
        return variant

    def get(self, name: str) -> ModelGraphVariant | None:
        return self._variants.get(name)

    def counts(self) -> dict[str, int]:
        return {
            name: variant.capture_count
            for name, variant in self._variants.items()
        }

    def clear(self) -> None:
        self._variants.clear()
        self.active_name = None
        self._capture_handler = None
