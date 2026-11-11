# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.experimental.archived.execution_coordinator import (
    ContainerExecutionCoordinator,
)
from vllm.experimental.worker.resource_desc import (
    ExecutionVariantDescription,
    MemoryResourceDescription,
    ResourceDescription,
    StreamResourceDescription,
)


def _container(model_id: str, full_variant: str) -> SimpleNamespace:
    description = ResourceDescription(
        memory=MemoryResourceDescription("default"),
        execution_variants=(
            ExecutionVariantDescription(
                "shared", StreamResourceDescription("dedicated")
            ),
            ExecutionVariantDescription(
                full_variant, StreamResourceDescription("dedicated")
            ),
        ),
        default_variant=full_variant,
    )
    return SimpleNamespace(
        model_id=model_id,
        resource_description=description,
    )


def test_coordinator_selects_full_without_competing_work() -> None:
    coordinator = ContainerExecutionCoordinator()
    reasoner = _container("reasoner", "reasoner_full")
    generator = _container("generator", "full")

    assert coordinator.select_variant(reasoner, "auto") == "reasoner_full"
    assert coordinator.select_variant(generator, "auto") == "full"


def test_coordinator_selects_shared_symmetrically() -> None:
    coordinator = ContainerExecutionCoordinator()
    reasoner = _container("reasoner", "reasoner_full")
    generator = _container("generator", "full")

    coordinator.set_demand("reasoner", True)
    assert coordinator.select_variant(generator, "auto") == "shared"
    coordinator.set_demand("reasoner", False)

    with coordinator.request_scope("generator"):
        assert coordinator.select_variant(reasoner, "auto") == "shared"
    assert coordinator.select_variant(reasoner, "auto") == "reasoner_full"
