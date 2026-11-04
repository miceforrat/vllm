#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/workspace/vllm-single-framework}"
output_root="${OUTPUT_ROOT:-outputs/reasoner_graph_mode_matrix}"
python_bin="${PYTHON_BIN:-python}"

cd "${repo_root}"
mkdir -p \
  "${output_root}/full" \
  "${output_root}/piecewise" \
  "${output_root}/breakable_piecewise"

common_args=(
  examples/offline_inference/omni_worker_async_gate.py
  --model nvidia/Cosmos3-Edge
  --tp-size 1
  --reasoner-mode graph
  --reasoner-max-tokens 1350
  --reasoner-fixed-tokens
  --reasoner-input examples/offline_inference/workloads/cosmos3_agentic_t2v_matched/reasoner.json
  --generator-workload-manifest examples/offline_inference/workloads/cosmos3_agentic_t2v_matched/generator_manifest.json
  --generator-cache sea_cache
  --generator-stream green
  --generator-sm-count 136
  --partition-policy dynamic
  --reasoner-graph-pool-mode shared
  --reasoner-graph-buffer-mode shared
  --warmups 3
  --repeats 10
)

run_case() {
  local implementation="$1"
  local case_dir="${output_root}/${implementation}"

  echo "===== ${implementation} CUDA Graph ====="
  env CUBLAS_WORKSPACE_CONFIG=:4096:2 \
    "${python_bin}" -u "${common_args[@]}" \
    --reasoner-graph-capture-mode "${implementation}" \
    --output "${case_dir}/report.json" \
    2>&1 | tee "${case_dir}/run.log"
}

run_case full
run_case piecewise
run_case breakable_piecewise

"${python_bin}" - "${output_root}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
cases = {
    name: json.loads((root / name / "report.json").read_text())
    for name in ("full", "piecewise", "breakable_piecewise")
}


def mib(value: int | float) -> float:
    return float(value) / 2**20


def pool_total(snapshot: dict, pool: str, field: str) -> int:
    return int(snapshot.get("pools", {}).get(pool, {}).get(field, 0))


def extract(data: dict) -> dict[str, object]:
    startup = data["startup"]["summary"]
    plan = data["stream_control"][0]
    memory = plan["reasoner_graph_memory"]
    snapshots = plan["reasoner_graph_memory_snapshot"]
    shared_registry = snapshots["shared_registry"]
    full_registry = snapshots["full_registry"]
    after_shared = snapshots["after_shared_capture"]
    after_full = snapshots["after_full_capture_after_cleanup"]

    shared_pool = shared_registry["manager_pool"]
    full_pool = full_registry["manager_pool"]
    shared_pool_segment = pool_total(
        after_shared, shared_pool, "total_size_bytes"
    )
    full_pool_delta = (
        pool_total(after_full, full_pool, "total_size_bytes")
        - pool_total(after_shared, full_pool, "total_size_bytes")
    )
    default_allocated_delta = (
        pool_total(after_full, "(0,0)", "allocated_size_bytes")
        - pool_total(after_shared, "(0,0)", "allocated_size_bytes")
    )
    default_reserved_delta = (
        pool_total(after_full, "(0,0)", "total_size_bytes")
        - pool_total(after_shared, "(0,0)", "total_size_bytes")
    )

    measurements = data.get("measurements", [])
    allocated_growth = reserved_growth = 0
    if measurements:
        first = measurements[0]["cuda_allocator_before"][0]
        last = measurements[-1]["cuda_allocator_after"][0]
        allocated_growth = int(last["allocated_bytes"]) - int(
            first["allocated_bytes"]
        )
        reserved_growth = int(last["reserved_bytes"]) - int(
            first["reserved_bytes"]
        )

    result = dict(startup)
    result.update({
        "Shared capture total device delta": mib(
            memory["shared_graph_device_bytes"]
        ),
        "Additional full-variant total device delta": mib(
            memory["full_variant_device_bytes_after_cleanup"]
        ),
        "Dual-variant capture total device delta": mib(
            memory["total_graph_device_bytes_after_cleanup"]
        ),
        "Shared graph-pool segment": mib(shared_pool_segment),
        "Full graph-pool segment delta": mib(full_pool_delta),
        "Full default-pool allocated delta": mib(default_allocated_delta),
        "Full default-pool reserved delta": mib(default_reserved_delta),
        # Allocation-stack analysis established that this live allocation is
        # the new full-stream cuBLAS/cuBLASLt workspace for this workload.
        "Full cuBLAS workspace allocated delta": mib(default_allocated_delta),
        "Steady-state allocated growth": mib(allocated_growth),
        "Steady-state reserved growth": mib(reserved_growth),
    })
    return result


rows = [
    ("s", "关键路径", "engine_startup_total_seconds"),
    ("s", "关键路径", "config_creation_seconds"),
    ("s", "关键路径", "frontend_initialization_seconds"),
    ("s", "关键路径", "engine_core_spawn_seconds"),
    ("s", "关键路径", "engine_core_pre_worker_seconds"),
    ("s", "关键路径", "worker_startup_seconds"),
    ("s", "关键路径", "engine_ready_finalize_seconds"),
    ("s", "并行阶段", "frontend_mm_warmup_seconds"),
    ("s", "Worker", "reasoner_device_runtime_init_seconds"),
    ("s", "Worker", "colocation_resource_stream_setup_seconds"),
    ("s", "Worker", "reasoner_model_load_seconds"),
    ("s", "Worker", "profile_and_memory_preparation_seconds"),
    ("s", "Worker", "kv_cache_initialization_seconds"),
    ("s", "Worker", "primary_compile_graph_setup_seconds"),
    ("s", "Worker", "additional_full_graph_capture_seconds"),
    ("s", "Graph", "shared_graph_capture_seconds"),
    ("s", "Graph", "full_graph_capture_seconds"),
    ("s", "Graph", "dual_graph_capture_total_seconds"),
    ("bool", "配置", "memory_profiling_performed"),
    ("MiB", "显存", "Shared capture total device delta"),
    ("MiB", "显存", "Additional full-variant total device delta"),
    ("MiB", "显存", "Dual-variant capture total device delta"),
    ("MiB", "显存", "Shared graph-pool segment"),
    ("MiB", "显存", "Full graph-pool segment delta"),
    ("MiB", "显存", "Full default-pool allocated delta"),
    ("MiB", "显存", "Full default-pool reserved delta"),
    ("MiB", "显存", "Full cuBLAS workspace allocated delta"),
    ("MiB", "显存", "Steady-state allocated growth"),
    ("MiB", "显存", "Steady-state reserved growth"),
]

values = {name: extract(data) for name, data in cases.items()}
output = root / "comparison.tsv"
with output.open("w", encoding="utf-8") as f:
    print(
        "单位\t类别\t属性\tfull\tpiecewise\tbreakable_piecewise",
        file=f,
    )
    for unit, category, name in rows:
        print(
            unit,
            category,
            name,
            values["full"].get(name, ""),
            values["piecewise"].get(name, ""),
            values["breakable_piecewise"].get(name, ""),
            sep="\t",
            file=f,
        )

print(f"Wrote {output}")

performance_output = root / "performance.tsv"
with performance_output.open("w", encoding="utf-8") as f:
    print(
        "mode\treasoner_median_s\treasoner_mean_s\t"
        "generator_median_s\tgenerator_mean_s\t"
        "host_wall_median_s\thost_wall_mean_s\t"
        "distinct_reasoner_hashes\treasoner_hash",
        file=f,
    )
    for name, data in cases.items():
        measurements = data["measurements"]
        reasoner = [row["reasoner"]["seconds"] for row in measurements]
        generator = [row["generator"]["seconds"] for row in measurements]
        wall = [row["host_wall_seconds"] for row in measurements]
        hashes = sorted(
            {row["reasoner"]["text_sha256"] for row in measurements}
        )
        print(
            name,
            statistics.median(reasoner),
            statistics.mean(reasoner),
            statistics.median(generator),
            statistics.mean(generator),
            statistics.median(wall),
            statistics.mean(wall),
            len(hashes),
            ",".join(hashes),
            sep="\t",
            file=f,
        )
print(f"Wrote {performance_output}")

for name, data in cases.items():
    plan = data["stream_control"][0]
    capture_plan = plan.get("reasoner_graph_capture_plan", {})
    variants = plan.get("reasoner_graph_variants", {})
    cg = plan.get("reasoner_cudagraph", {})
    assert capture_plan.get("orchestration") == "single_vllm_capture_trigger"
    assert variants.get("shared", 0) > 0
    assert variants.get("reasoner_full", 0) > 0
    assert capture_plan.get("mode") == name
    if name != "breakable_piecewise":
        assert not cg.get("breakable_enabled", False)
    else:
        assert cg.get("breakable_enabled", False)
        assert cg.get("breakable_capture_calls", 0) > 0
        if cg.get("breakable_replay_calls", 0) == 0:
            print(
                "NOTE: breakable graphs were captured but this FULL-decode "
                "workload did not replay the PIECEWISE wrapper"
            )
    print(name, capture_plan, variants, cg)
PY
