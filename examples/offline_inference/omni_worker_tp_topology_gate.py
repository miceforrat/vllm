# SPDX-License-Identifier: Apache-2.0
"""Gate P1: inspect homogeneous vLLM TP topology for an Omni runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _validate(rows: list[dict[str, Any]], tp_size: int) -> dict[str, Any]:
    expected = list(range(tp_size))
    if len(rows) != tp_size:
        raise RuntimeError(
            f"Expected {tp_size} worker replies, received {len(rows)}"
        )
    rows = sorted(rows, key=lambda row: row["rank"])
    ranks = [row["rank"] for row in rows]
    devices = [row["device"] for row in rows]
    pids = [row["pid"] for row in rows]
    if ranks != expected:
        raise RuntimeError(f"Worker ranks {ranks} do not match {expected}")
    if len(set(devices)) != tp_size:
        raise RuntimeError(f"Workers do not own distinct devices: {devices}")
    for row in rows:
        if not row["active_for_placement"]:
            raise RuntimeError(f"Rank {row['rank']} rejected TP placement")
        if row["world"]["ranks"] != expected:
            raise RuntimeError(f"Unexpected world group on rank {row['rank']}")
        if row["tp"]["ranks"] != expected:
            raise RuntimeError(f"Unexpected TP group on rank {row['rank']}")
        if row["tp"]["rank_in_group"] != row["rank"]:
            raise RuntimeError(f"TP rank mismatch on rank {row['rank']}")
        if row["pp"]["world_size"] != 1:
            raise RuntimeError("Gate P1 requires pipeline_parallel_size=1")
    return {
        "verified": True,
        "tp_size": tp_size,
        "worker_ranks": ranks,
        "devices": devices,
        "worker_pids": pids,
        "process_model": (
            "single_process" if len(set(pids)) == 1 else "multi_process"
        ),
        "omni_mapping": {
            "world": "vllm.world",
            "tensor_parallel": "vllm.tp",
            "pipeline_parallel": "singleton",
            "sequence_parallel": "singleton",
            "classifier_free_guidance": "singleton",
            "data_parallel": "singleton",
            "fully_sharded": "singleton",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/Cosmos3-Edge")
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tp_size < 2:
        raise ValueError("Gate P1 is intended for tp_size >= 2")

    from vllm import LLM
    from vllm.experimental.multi_model_worker import WorkerHostedRunnerClient

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        pipeline_parallel_size=1,
        max_num_seqs=1,
        max_model_len=1024,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
        worker_extension_cls=(
            "vllm.experimental.multi_model_worker.MultiModelWorkerExtension"
        ),
    )
    client = WorkerHostedRunnerClient(llm.llm_engine)
    rows = client.topology(tuple(range(args.tp_size)))
    report = {
        "gate": "homogeneous_tp_topology",
        "model": args.model,
        "workers": rows,
        "summary": _validate(rows, args.tp_size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
