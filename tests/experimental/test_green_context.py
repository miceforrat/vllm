# SPDX-License-Identifier: Apache-2.0

import ctypes

import pytest

from vllm.experimental.green_context import _DevResource


def test_dev_resource_abi_size_is_stable() -> None:
    assert ctypes.sizeof(_DevResource) == 144


def test_balanced_workqueue_config_layout() -> None:
    resource = _DevResource()

    resource.set_balanced_workqueue_config(device=3, concurrency_limit=2)

    assert resource.type == 1000
    assert resource.workqueue_config == {
        "device": 3,
        "concurrency_limit": 2,
        "sharing_scope": 1,
    }


def test_workqueue_concurrency_limit_must_be_positive() -> None:
    resource = _DevResource()

    with pytest.raises(ValueError, match="must be positive"):
        resource.set_balanced_workqueue_config(device=0, concurrency_limit=0)
