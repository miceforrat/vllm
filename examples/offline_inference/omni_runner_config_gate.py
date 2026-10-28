# SPDX-License-Identifier: Apache-2.0
"""Resolve an Omni diffusion model without constructing an Omni engine."""

from __future__ import annotations

import argparse
import json

from vllm.experimental.omni_runner_adapter import (
    EmbeddedOmniDiffusionRunner,
    resolve_omni_diffusion_stage,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--deploy-config")
    parser.add_argument(
        "--load-format", choices=("none", "dummy", "default"), default="none"
    )
    args = parser.parse_args()

    resolved = resolve_omni_diffusion_stage(
        args.model,
        deploy_config_path=args.deploy_config,
    )
    config = resolved.diffusion_config
    print(
        json.dumps(
            {
                "pipeline": resolved.pipeline_config.model_type,
                "stage_type": resolved.stage_config.stage_type.value,
                "model": config.model,
                "model_class_name": config.model_class_name,
                "engine_backend": config.engine_backend,
                "world_size": config.parallel_config.world_size,
                "cache_backend": config.cache_backend,
                "diffusion_compile_granularity": config.diffusion_compile_granularity,
            },
            indent=2,
        )
    )
    if args.load_format != "none":
        embedded = EmbeddedOmniDiffusionRunner(resolved)
        runner = embedded.start(load_format=args.load_format)
        print(
            json.dumps(
                {
                    "runner_type": f"{type(runner).__module__}.{type(runner).__name__}",
                    "pipeline_type": (
                        None if runner.pipeline is None else type(runner.pipeline).__name__
                    ),
                    "resource_owner": "vllm",
                },
                indent=2,
            )

        )
        embedded.close()


if __name__ == "__main__":
    main()
