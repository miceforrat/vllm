#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/workspace/vllm-single-framework}"
output_root="${OUTPUT_ROOT:-outputs/cublas_workspace_ab}"
python_bin="${PYTHON_BIN:-python}"

cd "${repo_root}"
mkdir -p "${output_root}"

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
  --warmups 1
  --repeats 1
)

run_case() {
  local name="$1"
  local workspace_config="$2"
  local case_dir="${output_root}/${name}"
  mkdir -p "${case_dir}"

  echo
  echo "===== ${name}: CUBLAS_WORKSPACE_CONFIG=${workspace_config:-<unset>} ====="

  if [[ -z "${workspace_config}" ]]; then
    env -u CUBLAS_WORKSPACE_CONFIG \
      "${python_bin}" -u "${common_args[@]}" \
      --output "${case_dir}/report.json" \
      2>&1 | tee "${case_dir}/run.log"
  else
    env CUBLAS_WORKSPACE_CONFIG="${workspace_config}" \
      "${python_bin}" -u "${common_args[@]}" \
      --output "${case_dir}/report.json" \
      2>&1 | tee "${case_dir}/run.log"
  fi
}

run_case default ""
run_case 4096x2 ":4096:2"
run_case 16x8 ":16:8"

"${python_bin}" - "${output_root}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])

for name in ("default", "4096x2", "16x8"):
    path = root / name / "report.json"
    data = json.loads(path.read_text())
    plan = data["stream_control"][0]
    snap = plan["reasoner_graph_memory_snapshot"]
    before = snap["after_shared_capture"]["pools"]
    after = snap["after_full_capture_after_cleanup"]["pools"]

    pool = "(0,0)"
    allocated_delta = (
        after.get(pool, {}).get("allocated_size_bytes", 0)
        - before.get(pool, {}).get("allocated_size_bytes", 0)
    ) / 2**20
    reserved_delta = (
        after.get(pool, {}).get("total_size_bytes", 0)
        - before.get(pool, {}).get("total_size_bytes", 0)
    ) / 2**20

    measurements = data.get("measurements", [])
    reasoner_samples = [
        row["reasoner"]["seconds"]
        for row in measurements
        if row.get("reasoner") is not None
    ]
    generator_samples = [
        row["generator"]["seconds"]
        for row in measurements
        if row.get("generator") is not None
    ]
    reasoner = data.get("reasoner", {})
    generator = data.get("generator", {})

    reasoner_seconds = (
        statistics.median(reasoner_samples)
        if reasoner_samples
        else reasoner.get("seconds")
    )
    generator_seconds = (
        statistics.median(generator_samples)
        if generator_samples
        else generator.get("seconds")
    )

    print(f"\n=== {name} ===")
    print(f"default-pool allocated delta: {allocated_delta:.3f} MiB")
    print(f"default-pool reserved delta:  {reserved_delta:.3f} MiB")
    print(f"reasoner seconds: {reasoner_seconds}")
    print(f"generator seconds: {generator_seconds}")
    print(f"reasoner hash: {reasoner.get('text_sha256')}")
PY
