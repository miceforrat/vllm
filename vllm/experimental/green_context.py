"""Minimal CUDA Driver API wrapper for SM-partitioned Green Context streams."""

from __future__ import annotations

import ctypes
import ctypes.util


class GreenContextError(RuntimeError):
    pass


class _DevResource(ctypes.Structure):
    # cuda.h: enum (4) + internal padding (92) + tagged-union payload (48).
    _fields_ = [("type", ctypes.c_int), ("_padding", ctypes.c_ubyte * 92), ("_payload", ctypes.c_ubyte * 48)]

    @property
    def sm_count(self) -> int:
        return int(ctypes.c_uint.from_buffer(self._payload).value)


class GreenContextStream:
    """Own a Green Context and a non-blocking stream backed by its SM set."""

    _SM_RESOURCE = 1
    _DEFAULT_STREAM = 1
    _STREAM_NON_BLOCKING = 1

    def __init__(self, requested_sm_count: int, device: int = 0) -> None:
        if requested_sm_count <= 0:
            raise ValueError("requested_sm_count must be positive")
        self._cuda = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
        self._configure_signatures()
        self._check(self._cuda.cuInit(0), "cuInit")

        cuda_device = ctypes.c_int()
        self._check(self._cuda.cuDeviceGet(ctypes.byref(cuda_device), device), "cuDeviceGet")
        available = _DevResource()
        self._check(
            self._cuda.cuDeviceGetDevResource(cuda_device, ctypes.byref(available), self._SM_RESOURCE),
            "cuDeviceGetDevResource",
        )
        if requested_sm_count > available.sm_count:
            raise ValueError(f"requested {requested_sm_count} SMs; device exposes {available.sm_count}")

        group = _DevResource()
        remainder = _DevResource()
        group_count = ctypes.c_uint(1)
        self._check(
            self._cuda.cuDevSmResourceSplitByCount(
                ctypes.byref(group), ctypes.byref(group_count), ctypes.byref(available),
                ctypes.byref(remainder), 0, requested_sm_count
            ),
            "cuDevSmResourceSplitByCount",
        )
        if group_count.value != 1:
            raise GreenContextError(f"CUDA returned {group_count.value} partitions; expected one")

        descriptor = ctypes.c_void_p()
        self._check(
            self._cuda.cuDevResourceGenerateDesc(ctypes.byref(descriptor), ctypes.byref(group), 1),
            "cuDevResourceGenerateDesc",
        )
        self._green_context = ctypes.c_void_p()
        self._check(
            self._cuda.cuGreenCtxCreate(
                ctypes.byref(self._green_context), descriptor, cuda_device, self._DEFAULT_STREAM
            ),
            "cuGreenCtxCreate",
        )
        self._stream = ctypes.c_void_p()
        try:
            self._check(
                self._cuda.cuGreenCtxStreamCreate(
                    ctypes.byref(self._stream), self._green_context, self._STREAM_NON_BLOCKING, 0
                ),
                "cuGreenCtxStreamCreate",
            )
        except Exception:
            self._cuda.cuGreenCtxDestroy(self._green_context)
            raise
        self.device = device
        self.requested_sm_count = requested_sm_count
        self.actual_sm_count = group.sm_count

    @property
    def stream_ptr(self) -> int:
        if self._stream.value is None:
            raise GreenContextError("stream is closed")
        return self._stream.value

    def close(self) -> None:
        if getattr(self, "_stream", None) is not None and self._stream.value is not None:
            self._check(self._cuda.cuStreamDestroy_v2(self._stream), "cuStreamDestroy")
            self._stream = ctypes.c_void_p()
        if getattr(self, "_green_context", None) is not None and self._green_context.value is not None:
            self._check(self._cuda.cuGreenCtxDestroy(self._green_context), "cuGreenCtxDestroy")
            self._green_context = ctypes.c_void_p()

    def _configure_signatures(self) -> None:
        resource_p = ctypes.POINTER(_DevResource)
        self._cuda.cuInit.argtypes = [ctypes.c_uint]
        self._cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._cuda.cuDeviceGetDevResource.argtypes = [ctypes.c_int, resource_p, ctypes.c_int]
        self._cuda.cuDevSmResourceSplitByCount.argtypes = [
            resource_p, ctypes.POINTER(ctypes.c_uint), resource_p, resource_p, ctypes.c_uint, ctypes.c_uint
        ]
        self._cuda.cuDevResourceGenerateDesc.argtypes = [ctypes.POINTER(ctypes.c_void_p), resource_p, ctypes.c_uint]
        self._cuda.cuGreenCtxCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
        ]
        self._cuda.cuGreenCtxStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint, ctypes.c_int
        ]
        self._cuda.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
        self._cuda.cuGreenCtxDestroy.argtypes = [ctypes.c_void_p]
        self._cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._cuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def _check(self, status: int, operation: str) -> None:
        if status == 0:
            return
        name = ctypes.c_char_p()
        message = ctypes.c_char_p()
        self._cuda.cuGetErrorName(status, ctypes.byref(name))
        self._cuda.cuGetErrorString(status, ctypes.byref(message))
        error_name = name.value.decode() if name.value else f"CUDA error {status}"
        error_message = message.value.decode() if message.value else "unknown error"
        raise GreenContextError(f"{operation} failed: {error_name}: {error_message}")


class GreenContextPair:
    """Own two disjoint Green Context streams made from one SM split.

    This mirrors FastVideo's proven ``GreenContextPairPool`` allocation: the
    first resource receives ``requested_first_sm_count`` SMs and the second
    context is created from the remainder returned by the same driver call.
    Creating two independent ``GreenContextStream`` objects would not provide
    that non-overlap guarantee.
    """

    _SM_RESOURCE = 1
    _DEFAULT_STREAM = 1
    _STREAM_NON_BLOCKING = 1
    _IGNORE_SM_COSCHEDULING = 1

    def __init__(
        self,
        requested_first_sm_count: int,
        device: int = 0,
        *,
        ignore_sm_coscheduling: bool = False,
    ) -> None:
        self._cuda = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
        GreenContextStream._configure_signatures(self)
        GreenContextStream._check(self, self._cuda.cuInit(0), "cuInit")

        cuda_device = ctypes.c_int()
        GreenContextStream._check(
            self, self._cuda.cuDeviceGet(ctypes.byref(cuda_device), device), "cuDeviceGet"
        )
        available = _DevResource()
        GreenContextStream._check(
            self,
            self._cuda.cuDeviceGetDevResource(
                cuda_device, ctypes.byref(available), self._SM_RESOURCE
            ),
            "cuDeviceGetDevResource",
        )
        if requested_first_sm_count <= 0 or requested_first_sm_count >= available.sm_count:
            raise ValueError(
                "requested_first_sm_count must be positive and smaller than "
                f"the device SM count ({available.sm_count})"
            )

        first_resource = _DevResource()
        second_resource = _DevResource()
        group_count = ctypes.c_uint(1)
        flags = self._IGNORE_SM_COSCHEDULING if ignore_sm_coscheduling else 0
        GreenContextStream._check(
            self,
            self._cuda.cuDevSmResourceSplitByCount(
                ctypes.byref(first_resource),
                ctypes.byref(group_count),
                ctypes.byref(available),
                ctypes.byref(second_resource),
                flags,
                requested_first_sm_count,
            ),
            "cuDevSmResourceSplitByCount",
        )
        if group_count.value != 1:
            raise GreenContextError(
                f"CUDA returned {group_count.value} first partitions; expected one"
            )

        self._contexts: list[ctypes.c_void_p] = []
        self._streams: list[ctypes.c_void_p] = []
        try:
            for resource in (first_resource, second_resource):
                descriptor = ctypes.c_void_p()
                GreenContextStream._check(
                    self,
                    self._cuda.cuDevResourceGenerateDesc(
                        ctypes.byref(descriptor), ctypes.byref(resource), 1
                    ),
                    "cuDevResourceGenerateDesc",
                )
                context = ctypes.c_void_p()
                GreenContextStream._check(
                    self,
                    self._cuda.cuGreenCtxCreate(
                        ctypes.byref(context), descriptor, cuda_device, self._DEFAULT_STREAM
                    ),
                    "cuGreenCtxCreate",
                )
                self._contexts.append(context)
                stream = ctypes.c_void_p()
                GreenContextStream._check(
                    self,
                    self._cuda.cuGreenCtxStreamCreate(
                        ctypes.byref(stream), context, self._STREAM_NON_BLOCKING, 0
                    ),
                    "cuGreenCtxStreamCreate",
                )
                self._streams.append(stream)
        except Exception:
            self.close()
            raise

        self.device = device
        self.requested_first_sm_count = requested_first_sm_count
        self.first_sm_count = first_resource.sm_count
        self.second_sm_count = second_resource.sm_count
        self.total_sm_count = available.sm_count

    @property
    def first_stream_ptr(self) -> int:
        return self._stream_ptr(0)

    @property
    def second_stream_ptr(self) -> int:
        return self._stream_ptr(1)

    def _stream_ptr(self, index: int) -> int:
        if index >= len(self._streams) or self._streams[index].value is None:
            raise GreenContextError("Green Context pair stream is closed")
        return int(self._streams[index].value)

    def close(self) -> None:
        for stream in reversed(getattr(self, "_streams", [])):
            if stream.value is not None:
                GreenContextStream._check(
                    self, self._cuda.cuStreamDestroy_v2(stream), "cuStreamDestroy"
                )
                stream.value = None
        for context in reversed(getattr(self, "_contexts", [])):
            if context.value is not None:
                GreenContextStream._check(
                    self, self._cuda.cuGreenCtxDestroy(context), "cuGreenCtxDestroy"
                )
                context.value = None
