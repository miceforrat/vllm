# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.experimental.worker.model_lifecycle import OmniModelLifecycle


def test_omni_lifecycle_loads_only_when_container_requests_it() -> None:
    calls = []

    class _Runner:
        _container_runtime_providers = object()

        def load_model(self, **kwargs):
            calls.append(kwargs)

    unit = SimpleNamespace(
        model_id="generator",
        runner=_Runner(),
        spec=SimpleNamespace(load_format="default"),
    )
    lifecycle = OmniModelLifecycle(unit)

    assert calls == []
    lifecycle.load_model()
    assert calls == [
        {
            "runtime_providers": unit.runner._container_runtime_providers,
            "load_format": "default",
        }
    ]


def test_omni_lifecycle_rejects_vllm_kv_cache_protocol() -> None:
    unit = SimpleNamespace(model_id="generator", runner=object())
    lifecycle = OmniModelLifecycle(unit)

    with pytest.raises(NotImplementedError, match="KV-cache"):
        lifecycle.initialize_from_config(object())
