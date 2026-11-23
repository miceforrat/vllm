# SPDX-License-Identifier: Apache-2.0
"""Generate dependency-free SVG/TSV partition scaling plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling-dir", type=Path, required=True)
    parser.add_argument("--throughput-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1350)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot take the median of an empty series")
    return float(statistics.median(values))


def load_speedup_series(
    root: Path, tokens: int
) -> tuple[dict[str, list[tuple[int, float]]], dict[str, float]]:
    decode_times: dict[int, list[float]] = defaultdict(list)
    component_times: dict[str, dict[int, list[float]]] = {
        "Diffusion sampling": defaultdict(list),
        "VAE decode": defaultdict(list),
    }
    component_names = {
        "diffusion_sampling": "Diffusion sampling",
        "vae_decode": "VAE decode",
    }

    for path in root.glob("*.json"):
        data = json.loads(path.read_text())
        workload = data.get("workload", {})
        if int(workload.get("reasoner_max_tokens", -1)) != tokens:
            continue
        component = data.get("component")
        if component not in component_names:
            continue
        partition = data["partition"]
        summary = data["summary"]
        reasoner_sm = int(partition["actual_decode_sm_count"])
        generator_sm = int(partition["actual_generator_sm_count"])
        decode_times[reasoner_sm].append(
            float(summary["decode_partitioned_alone_seconds"])
        )
        component_times[component_names[component]][generator_sm].append(
            float(summary["component_partitioned_alone_seconds"])
        )

    raw: dict[str, dict[int, float]] = {
        "LLM decode": {
            sm: _median(times) for sm, times in decode_times.items()
        }
    }
    raw.update(
        {
            name: {sm: _median(times) for sm, times in values.items()}
            for name, values in component_times.items()
        }
    )

    series: dict[str, list[tuple[int, float]]] = {}
    full_times: dict[str, float] = {}
    for name, values in raw.items():
        if 170 not in values:
            raise ValueError(
                f"{name} has no 170-SM full baseline; run the component "
                "benchmark with --full-sm-baseline first"
            )
        baseline = values[170]
        full_times[name] = baseline
        # 统一以 Generator SM 为横轴。Reasoner 与 Generator 使用互补分区，
        # 因此 LLM decode 的横坐标为 170 - reasoner_sm。
        series[name] = sorted(
            (
                170 - sm if name == "LLM decode" else sm,
                baseline / seconds,
            )
            for sm, seconds in values.items()
        )
    return series, full_times


def load_throughput_series(
    root: Path, tokens: int
) -> dict[str, list[tuple[int, float]]]:
    actual: list[tuple[int, float]] = []
    theoretical: list[tuple[int, float]] = []
    for path in root.glob(f"tok{tokens}_r*_g*.json"):
        data = json.loads(path.read_text())
        observed = int(data["reasoner_completion_tokens"])
        if observed != tokens:
            raise ValueError(
                f"{path} produced {observed} tokens, expected {tokens}"
            )
        deployment = data["deployment_plan"]["deployments"]
        generator = next(
            item for item in deployment if item["spec"]["model_id"] == "generator"
        )
        shared = next(
            item
            for item in generator["resources"]["execution_variants"]
            if item["name"] == "shared"
        )
        generator_sm = int(shared["stream"]["sm_count"])
        perf = data["performance"]
        reasoner_alone = float(perf["reasoner_alone_median"])
        generator_alone = float(perf["generator_alone_median"])
        serial = reasoner_alone + generator_alone
        wall = _median(
            [float(item["host_wall_seconds"]) for item in perf["concurrent"]]
        )
        actual.append((generator_sm, serial / wall))
        theoretical.append((generator_sm, serial / generator_alone))
    if not actual:
        raise ValueError(f"No tok{tokens} throughput reports found in {root}")
    return {
        "Measured gain": sorted(actual),
        "Theoretical gain": sorted(theoretical),
    }


def write_tsv(
    path: Path, series: dict[str, list[tuple[int, float]]], value_name: str
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["series", "sm_count", value_name])
        for name, points in series.items():
            for sm, value in points:
                writer.writerow([name, sm, f"{value:.8f}"])


def write_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[int, float]]],
    reference_y: float | None = None,
    annotate_peaks: bool = True,
    point_annotations: dict[
        str, tuple[float, float, str, float, float, str]
    ] | None = None,
) -> None:
    width, height = 960, 600
    left, right, top, bottom = 92, 28, 72, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    points = [point for values in series.values() for point in values]
    x_min, x_max = min(x for x, _ in points), max(x for x, _ in points)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    y_min = 0.0
    y_max = max(y for _, y in points)
    if reference_y is not None:
        y_max = max(y_max, reference_y)
    y_max = math.ceil(y_max * 1.08 * 2) / 2
    if y_max == y_min:
        y_max = 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{escape(title)}</title>",
        "<style>text{font-family:Arial,sans-serif;fill:#1f2937}"
        ".grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#374151;stroke-width:1.5}"
        ".tick{font-size:12px}.label{font-size:14px}"
        ".title{font-size:21px;font-weight:600}"
        ".subtitle{font-size:13px;fill:#4b5563}.legend{font-size:13px}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="30">{escape(title)}</text>',
        f'<text class="subtitle" x="{left}" y="52">{escape(subtitle)}</text>',
    ]

    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = sy(value)
        chunks.append(
            f'<line class="grid" x1="{left}" y1="{y:.1f}" '
            f'x2="{left + plot_w}" y2="{y:.1f}"/>'
        )
        chunks.append(
            f'<text class="tick" text-anchor="end" x="{left - 10}" '
            f'y="{y + 4:.1f}">{value:.1f}</text>'
        )

    x_ticks = sorted({x for x, _ in points})
    for index, value in enumerate(x_ticks):
        if index % 2 and index != len(x_ticks) - 1:
            continue
        x = sx(value)
        chunks.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top}" '
            f'x2="{x:.1f}" y2="{top + plot_h}"/>'
        )
        chunks.append(
            f'<text class="tick" text-anchor="middle" x="{x:.1f}" '
            f'y="{top + plot_h + 22}">{value}</text>'
        )

    if reference_y is not None:
        y = sy(reference_y)
        chunks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
            f'y2="{y:.1f}" stroke="#6b7280" stroke-dasharray="6 5"/>'
        )

    chunks.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" '
            f'x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{top + plot_h}"/>',
            f'<text class="label" text-anchor="middle" x="{left + plot_w / 2}" '
            f'y="{height - 22}">{escape(x_label)}</text>',
            f'<text class="label" text-anchor="middle" '
            f'transform="translate(24 {top + plot_h / 2}) rotate(-90)">'
            f"{escape(y_label)}</text>",
        ]
    )

    legend_x = left + 14
    legend_y = top + 18
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        if annotate_peaks:
            peak_x, peak_y = max(values, key=lambda point: point[1])
            peak_sx = sx(peak_x)
            peak_sy = sy(peak_y)
            chunks.append(
                f'<line x1="{left}" y1="{peak_sy:.1f}" '
                f'x2="{peak_sx:.1f}" y2="{peak_sy:.1f}" stroke="{color}" '
                'stroke-width="1.2" stroke-dasharray="5 4" opacity="0.65"/>'
            )
            chunks.append(
                f'<line x1="{peak_sx:.1f}" y1="{peak_sy:.1f}" '
                f'x2="{peak_sx:.1f}" y2="{top + plot_h}" stroke="{color}" '
                'stroke-width="1.2" stroke-dasharray="5 4" opacity="0.65"/>'
            )
            label_anchor = (
                "end" if peak_sx > left + plot_w * 0.75 else "start"
            )
            label_dx = -7 if label_anchor == "end" else 7
            chunks.append(
                f'<text class="tick" text-anchor="{label_anchor}" '
                f'x="{peak_sx + label_dx:.1f}" y="{peak_sy - 7:.1f}" '
                f'fill="{color}">max {peak_y:.3f}×</text>'
            )
        polyline = " ".join(
            f"{sx(x):.1f},{sy(y):.1f}" for x, y in values
        )
        chunks.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round"/>'
        )
        for x, y in values:
            chunks.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.4" '
                f'fill="{color}"/>'
            )
        if point_annotations and name in point_annotations:
            x, y, label, dx, dy, anchor = point_annotations[name]
            chunks.append(
                f'<text class="tick" text-anchor="{anchor}" '
                f'x="{sx(x) + dx:.1f}" y="{sy(y) + dy:.1f}" '
                f'fill="{color}">{escape(label)}</text>'
            )
        chunks.append(
            f'<line x1="{legend_x}" y1="{legend_y + index * 21}" '
            f'x2="{legend_x + 25}" y2="{legend_y + index * 21}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        chunks.append(
            f'<text class="legend" x="{legend_x + 33}" '
            f'y="{legend_y + 4 + index * 21}">{escape(name)}</text>'
        )

    chunks.append("</svg>\n")
    path.write_text("\n".join(chunks))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    speedup, full_times = load_speedup_series(args.scaling_dir, args.tokens)
    write_tsv(args.output_dir / "component_speedup_vs_sm.tsv", speedup, "speedup")
    write_svg(
        args.output_dir / "component_speedup_vs_sm.svg",
        title="Component scaling by SM allocation",
        subtitle="Performance normalized to the 170-SM full-device baseline",
        x_label="Generator SM allocation (Reasoner SM = 170 − Generator SM)",
        y_label="Speedup (×)",
        series=speedup,
        annotate_peaks=False,
        point_annotations={
            "LLM decode": (
                0,
                1.0,
                f"full {full_times['LLM decode']:.3f}s",
                7,
                -10,
                "start",
            ),
            "Diffusion sampling": (
                170,
                1.0,
                f"full {full_times['Diffusion sampling']:.3f}s",
                -7,
                -12,
                "end",
            ),
            "VAE decode": (
                170,
                1.0,
                f"full {full_times['VAE decode']:.3f}s",
                -7,
                18,
                "end",
            ),
        },
    )

    throughput = load_throughput_series(args.throughput_dir, args.tokens)
    write_tsv(
        args.output_dir / f"throughput_gain_tok{args.tokens}_vs_sm.tsv",
        throughput,
        "throughput_gain",
    )
    write_svg(
        args.output_dir / f"throughput_gain_tok{args.tokens}_vs_sm.svg",
        title=f"Throughput gain across partitions ({args.tokens} tokens)",
        subtitle=(
            "Measured concurrent gain vs. ideal full-overlap upper bound; "
            "Reasoner SM = 170 − Generator SM"
        ),
        x_label="Generator SM allocation",
        y_label="Throughput gain (×)",
        series=throughput,
        reference_y=1.0,
    )


if __name__ == "__main__":
    main()
