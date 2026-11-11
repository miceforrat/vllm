# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

from vllm.experimental.archived.reasoner_graph_variants import (
    ModelGraphVariantManager,
)


class _NativeManager:
    def __init__(self, label: str) -> None:
        self.label = label
        self.graphs = {label: object()}

    def execute_model(self) -> str:
        return self.label


def test_stable_manager_facade_switches_native_variant() -> None:
    manager = ModelGraphVariantManager()
    runner = SimpleNamespace(cudagraph_manager=None)
    shared = _NativeManager("shared")
    full = _NativeManager("reasoner_full")

    manager.register("shared", object(), shared)
    manager.register("reasoner_full", object(), full)

    manager.binding("shared").activate(runner)
    assert runner.cudagraph_manager is manager
    assert manager.execute_model() == "shared"

    manager.binding("reasoner_full").activate(runner)
    assert runner.cudagraph_manager is manager
    assert manager.execute_model() == "reasoner_full"


def test_capture_provider_uses_one_multi_variant_entry() -> None:
    manager = ModelGraphVariantManager()
    calls: list[tuple[Any, Any, bool]] = []

    def handler(
        runner: Any,
        capture_native: Any,
        *,
        profile_only: bool,
    ) -> int:
        calls.append((runner, capture_native, profile_only))
        return 17

    def native(**_: Any) -> int:
        return 1
    runner = object()
    manager.set_capture_handler(handler)

    assert manager.capture_all(
        runner, native, profile_only=True
    ) == 17
    assert calls == [(runner, native, True)]

    manager.clear_capture_handler()
    assert manager.capture_all(
        runner, native, profile_only=False
    ) == 1
