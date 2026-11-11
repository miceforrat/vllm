# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.experimental.deployment import EngineDeploymentPlan, ModelDeployment
from vllm.experimental.model_instance_registry import (
    ModelSpec,
    create_engine_core_model_registry,
)
from vllm.experimental.worker.resource_desc import (
    ExecutionVariantDescription,
    MemoryResourceDescription,
    ResourceDescription,
    StreamResourceDescription,
)


def _resources(
    owner: str,
    sm_count: int,
) -> ResourceDescription:
    return ResourceDescription(
        memory=MemoryResourceDescription(
            "default" if owner == "reasoner" else "tagged_pool"
        ),
        execution_variants=(
            ExecutionVariantDescription(
                "shared",
                StreamResourceDescription(
                    "green_context", sm_count, "reasoner-generator"
                ),
            ),
        ),
        default_variant="shared",
    )


def _config(
    plan: EngineDeploymentPlan | None = None,
    *, resolved_model: str = "cosmos3",
):
    runtime = {}
    if plan is not None:
        runtime["deployment_plan"] = plan.to_rpc_payload()
    return SimpleNamespace(
        model_config=SimpleNamespace(model=resolved_model),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        additional_config={"multi_model_runtime": runtime},
    )


def test_engine_plans_models_and_resources_as_one_deployment() -> None:
    plan = EngineDeploymentPlan(
        deployments=(
            ModelDeployment(
                ModelSpec("reasoner", "cosmos3", "vllm"),
                _resources("reasoner", 34),
            ),
            ModelDeployment(
                ModelSpec("generator", "cosmos3", "vllm_omni"),
                _resources("generator", 136),
            ),
        ),
        default_model_id="reasoner",
        revision=3,
    )
    config = _config(plan)

    registry = create_engine_core_model_registry(config)

    assert tuple(spec.model_id for spec in registry.values()) == (
        "reasoner",
        "generator",
    )
    runtime = config.additional_config["multi_model_runtime"]
    restored = EngineDeploymentPlan.from_rpc_payload(
        runtime["deployment_plan"]
    )
    assert restored.revision == 3
    assert restored.deployments[0].spec.worker_ranks == (0, 1)
    assert restored.deployments[1].spec.worker_ranks == (0, 1)
    assert restored.deployments[1].spec.communication_groups[0].ranks == (
        0,
        1,
    )
    assert (
        restored.deployments[1].resources.communication_groups[0].group_id
        == "generator-tp"
    )
    assert len(runtime["startup_models"]) == 2
    assert set(runtime["startup_resources"]) == {"reasoner", "generator"}


def test_compatibility_path_synthesizes_single_vllm_deployment() -> None:
    config = _config()

    registry = create_engine_core_model_registry(config)

    assert tuple(spec.model_id for spec in registry.values()) == ("reasoner",)
    runtime = config.additional_config["multi_model_runtime"]
    plan = EngineDeploymentPlan.from_rpc_payload(runtime["deployment_plan"])
    assert len(plan.deployments) == 1
    assert plan.deployments[0].resources.variant().stream.kind == "default"


def test_native_deployment_uses_engine_resolved_model_path() -> None:
    plan = EngineDeploymentPlan(
        deployments=(
            ModelDeployment(
                ModelSpec("reasoner", "nvidia/Cosmos3-Edge", "vllm"),
                ResourceDescription(
                    memory=MemoryResourceDescription("default"),
                    execution_variants=(
                        ExecutionVariantDescription(
                            "default", StreamResourceDescription("default")
                        ),
                    ),
                    default_variant="default",
                ),
            ),
        ),
        default_model_id="reasoner",
    )
    snapshot = "/models/Cosmos3-Edge/snapshots/revision"
    config = _config(plan, resolved_model=snapshot)

    registry = create_engine_core_model_registry(config)

    assert registry.get("reasoner").model_path == snapshot


def test_green_partition_group_requires_exactly_two_members() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        EngineDeploymentPlan(
            deployments=(
                ModelDeployment(
                    ModelSpec("reasoner", "cosmos3", "vllm"),
                    _resources("reasoner", 34),
                ),
            ),
            default_model_id="reasoner",
        )
