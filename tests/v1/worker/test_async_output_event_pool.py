# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from unittest.mock import Mock

import torch

from vllm.v1.worker.gpu.async_utils import AsyncOutputEventPool


def test_async_output_event_pool_reuses_released_slot(monkeypatch):
    events = []

    def make_event(**kwargs):
        event = Mock(blocking=kwargs.get("blocking", False))
        events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", make_event)
    pool = AsyncOutputEventPool(2)

    first = pool.acquire()
    second = pool.acquire()
    assert first is not second
    assert pool.stats() == {
        "size": 2,
        "in_use": 2,
        "high_watermark": 2,
        "wait_count": 0,
        "miss_count": 0,
    }

    pool.release(first)
    assert pool.acquire() is first
    assert len(events) == 4


def test_async_output_event_pool_applies_backpressure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: Mock())
    pool = AsyncOutputEventPool(1)
    first = pool.acquire()
    acquired = threading.Event()

    def acquire_second():
        pool.acquire()
        acquired.set()

    thread = threading.Thread(target=acquire_second)
    thread.start()
    assert not acquired.wait(timeout=0.05)

    pool.release(first)
    assert acquired.wait(timeout=1)
    thread.join(timeout=1)
    assert pool.stats()["wait_count"] == 1


def test_async_output_event_pool_nonblocking_miss(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: Mock())
    pool = AsyncOutputEventPool(1)
    first = pool.acquire()

    assert pool.acquire(block=False) is None
    assert pool.stats()["miss_count"] == 1

    assert first is not None
    pool.release(first)
    assert pool.acquire(block=False) is first
