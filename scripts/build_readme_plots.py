#!/usr/bin/env python3
"""
Build README-friendly SVG plots from existing experiment outputs.

No third-party dependencies required.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Tuple


METHOD_ORDER = ["MAX", "TOP-10", "TOP-20", "TOP-30", "TOP-40", "TOP-50"]
THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1.0]
MODEL_COLORS = {
    "Bio_ClinicalBERT": "#1f77b4",
    "BiomedBERT": "#ff7f0e",
    "BlueBERT": "#2ca02c",
}


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_performance_files(root: str) -> List[str]:
    pattern = os.path.join(root, "Prediction_Output_*", "PerformanceIndex.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No PerformanceIndex.txt files found under Prediction_Output_*/")
    return files


def parse_metric_row(line: str) -> Tuple[float, Dict[str, float]] | None:
    parts = line.strip().split()
    if len(parts) < 7:
        return None
    try:
        threshold = float(parts[0])
        tp = float(parts[1])
        fp = float(parts[2])
        precision = float(parts[3])
        recall = float(parts[4])
        f1 = float(parts[5])
        pr = float(parts[6])
    except ValueError:
        return None
    return threshold, {
        "TP": tp,
        "FP": fp,
        "P": precision,
        "R": recall,
        "FS": f1,
        "PR": pr,
    }


def parse_performance_file(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    model_name = None
    metrics: Dict[str, Dict[float, Dict[str, float]]] = {}
    timing: Dict[str, float] = {}

    sec_pattern = re.compile(r"^([^:]+):\s+([0-9]+(?:\.[0-9]+)?)\s+seconds")
    model_pattern = re.compile(r"^MODEL:\s+([^(]+)")

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        section = re.match(r"^10-FOLD PERFORMANCE INDEX of (.+?) SIMILARITY by MAX$", line.strip())
        if section:
            method = section.group(1).strip()
            metrics[method] = {}
            i += 1
            while i < len(lines):
                parsed = parse_metric_row(lines[i])
                if parsed is None:
                    if metrics.get(method):
                        break
                    i += 1
                    continue
                threshold, row = parsed
                metrics[method][threshold] = row
                i += 1
            continue

        sec_match = sec_pattern.match(line.strip())
        if sec_match:
            key = sec_match.group(1).strip()
            timing[key] = float(sec_match.group(2))

        model_match = model_pattern.match(line.strip())
        if model_match:
            model_name = model_match.group(1).strip()

        i += 1

    if not model_name:
        model_name = os.path.basename(os.path.dirname(path)).replace("Prediction_Output_", "")

    return {"model": model_name, "path": path, "metrics": metrics, "timing": timing}


def svg_header(width: int, height: int) -> List[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def svg_footer(parts: List[str]) -> str:
    parts.append("</svg>")
    return "\n".join(parts)


def map_linear(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def draw_line_chart(
    path: str,
    title: str,
    x_values: List[float],
    x_labels: List[str],
    series: Dict[str, List[float]],
    y_min: float,
    y_max: float,
    y_ticks: List[float],
    y_label: str,
) -> None:
    width, height = 920, 520
    ml, mr, mt, mb = 80, 220, 60, 70
    pw, ph = width - ml - mr, height - mt - mb
    x0, y0 = ml, mt + ph
    x1, y1 = ml + pw, mt

    svg = svg_header(width, height)
    svg.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>')

    # Grid + y ticks.
    for t in y_ticks:
        y = map_linear(t, y_min, y_max, y0, y1)
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        svg.append(f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{t:g}</text>')

    # Axes.
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333333" stroke-width="1.5"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333333" stroke-width="1.5"/>')

    # X ticks.
    x_positions = []
    for xv, xl in zip(x_values, x_labels):
        x = map_linear(xv, min(x_values), max(x_values), x0, x1)
        x_positions.append(x)
        svg.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+6}" stroke="#333333" stroke-width="1"/>')
        svg.append(
            f'<text x="{x:.1f}" y="{y0+24}" text-anchor="middle" font-size="12" font-family="Arial">{xl}</text>'
        )

    svg.append(
        f'<text x="{x0-58}" y="{(y0+y1)/2:.1f}" transform="rotate(-90 {x0-58} {(y0+y1)/2:.1f})" '
        f'text-anchor="middle" font-size="13" font-family="Arial">{y_label}</text>'
    )

    # Series lines.
    legend_y = mt + 20
    for model, ys in series.items():
        color = MODEL_COLORS.get(model, "#444444")
        pts = []
        for x, yv in zip(x_positions, ys):
            y = map_linear(yv, y_min, y_max, y0, y1)
            pts.append(f"{x:.1f},{y:.1f}")
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(pts)}"/>')
        for x, yv in zip(x_positions, ys):
            y = map_linear(yv, y_min, y_max, y0, y1)
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')

        lx = x1 + 18
        svg.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx+24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(
            f'<text x="{lx+32}" y="{legend_y+4}" font-size="12" font-family="Arial">{model}</text>'
        )
        legend_y += 24

    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_footer(svg))


def draw_stacked_runtime_chart(
    path: str,
    title: str,
    model_names: List[str],
    load_m: List[float],
    emb_m: List[float],
    folds_m: List[float],
    total_m: List[float],
) -> None:
    width, height = 920, 560
    ml, mr, mt, mb = 80, 180, 60, 70
    pw, ph = width - ml - mr, height - mt - mb
    x0, y0 = ml, mt + ph
    x1, y1 = ml + pw, mt

    ymax = max(total_m) * 1.12 if total_m else 1.0

    svg = svg_header(width, height)
    svg.append(f'<text x="{width/2}" y="32" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>')

    # Y grid/ticks.
    y_ticks = [round(ymax * i / 5.0, 1) for i in range(6)]
    for t in y_ticks:
        y = map_linear(t, 0, ymax, y0, y1)
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        svg.append(f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial">{t:g}</text>')

    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333333" stroke-width="1.5"/>')
    svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333333" stroke-width="1.5"/>')
    svg.append(
        f'<text x="{x0-58}" y="{(y0+y1)/2:.1f}" transform="rotate(-90 {x0-58} {(y0+y1)/2:.1f})" '
        f'text-anchor="middle" font-size="13" font-family="Arial">Minutes</text>'
    )

    bar_w = 74
    gap = (pw - bar_w * len(model_names)) / (len(model_names) + 1)
    cur_x = x0 + gap

    for i, model in enumerate(model_names):
        v_load = load_m[i]
        v_emb = emb_m[i]
        v_fold = folds_m[i]
        v_total = total_m[i]

        base_y = y0
        h_load = map_linear(v_load, 0, ymax, 0, ph)
        h_emb = map_linear(v_emb, 0, ymax, 0, ph)
        h_fold = map_linear(v_fold, 0, ymax, 0, ph)

        # Stacked bars.
        svg.append(
            f'<rect x="{cur_x:.1f}" y="{base_y-h_load:.1f}" width="{bar_w}" height="{h_load:.1f}" fill="#4e79a7"/>'
        )
        svg.append(
            f'<rect x="{cur_x:.1f}" y="{base_y-h_load-h_emb:.1f}" width="{bar_w}" height="{h_emb:.1f}" fill="#59a14f"/>'
        )
        svg.append(
            f'<rect x="{cur_x:.1f}" y="{base_y-h_load-h_emb-h_fold:.1f}" width="{bar_w}" height="{h_fold:.1f}" fill="#f28e2b"/>'
        )
        svg.append(
            f'<text x="{cur_x + bar_w/2:.1f}" y="{base_y-h_load-h_emb-h_fold-8:.1f}" text-anchor="middle" '
            f'font-size="12" font-family="Arial">{v_total:.1f}m</text>'
        )
        svg.append(
            f'<text x="{cur_x + bar_w/2:.1f}" y="{y0+22}" text-anchor="middle" font-size="12" font-family="Arial">{model}</text>'
        )
        cur_x += bar_w + gap

    # Legend.
    lx, ly = x1 + 18, mt + 24
    legend_items = [("Model Loading", "#4e79a7"), ("Embeddings", "#59a14f"), ("10-Fold Eval", "#f28e2b")]
    for label, color in legend_items:
        svg.append(f'<rect x="{lx}" y="{ly-11}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx+22}" y="{ly}" font-size="12" font-family="Arial">{label}</text>')
        ly += 24

    with open(path, "w", encoding="utf-8") as f:
        f.write(svg_footer(svg))


def parse_saturation_summary(path: str) -> Dict[str, Dict[float, float]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out: Dict[str, Dict[float, float]] = {}
    in_section2 = False
    current_model = None

    for raw in lines:
        line = raw.strip()
        if line.startswith("SECTION 2: PER-PATIENT MAX SIMILARITIES"):
            in_section2 = True
            continue
        if line.startswith("SECTION 3: INTERPRETATION"):
            break
        if not in_section2:
            continue

        m_model = re.match(r"^Model:\s+(.+)$", line)
        if m_model:
            current_model = m_model.group(1).strip()
            out[current_model] = {}
            continue

        if current_model is None:
            continue

        m_pct = re.match(r"^%\s*>?=\s*([0-9.]+)\s*=\s*([0-9.]+)%$", line)
        if m_pct:
            threshold = float(m_pct.group(1))
            pct = float(m_pct.group(2))
            out[current_model][threshold] = pct

    return out


def main() -> None:
    root = project_root()
    out_dir = os.path.join(root, "docs", "readme_plots")
    os.makedirs(out_dir, exist_ok=True)

    perf_files = find_performance_files(root)
    parsed = [parse_performance_file(p) for p in perf_files]
    model_order = ["Bio_ClinicalBERT", "BiomedBERT", "BlueBERT"]
    parsed.sort(key=lambda r: model_order.index(r["model"]) if r["model"] in model_order else 999)  # type: ignore[index]

    # Chart 1: threshold sensitivity at TOP-10.
    series_top10: Dict[str, List[float]] = {}
    series_top50: Dict[str, List[float]] = {}
    for r in parsed:
        model = r["model"]  # type: ignore[index]
        metrics = r["metrics"]  # type: ignore[index]
        series_top10[model] = [metrics.get("TOP-10", {}).get(t, {}).get("FS", 0.0) for t in THRESHOLDS]  # type: ignore[union-attr]
        series_top50[model] = [metrics.get("TOP-50", {}).get(t, {}).get("FS", 0.0) for t in THRESHOLDS]  # type: ignore[union-attr]

    draw_line_chart(
        os.path.join(out_dir, "f1_vs_threshold_top10.svg"),
        "F1 vs Threshold (TOP-10)",
        THRESHOLDS,
        [str(t) for t in THRESHOLDS],
        series_top10,
        0.0,
        1.05,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "F1 Score",
    )
    draw_line_chart(
        os.path.join(out_dir, "f1_vs_threshold_top50.svg"),
        "F1 vs Threshold (TOP-50)",
        THRESHOLDS,
        [str(t) for t in THRESHOLDS],
        series_top50,
        0.0,
        1.05,
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "F1 Score",
    )

    # Chart 2: top-k effect at strict thresholds.
    x_methods = list(range(len(METHOD_ORDER)))
    labels_methods = METHOD_ORDER
    for threshold in [0.9, 1.0]:
        series_t: Dict[str, List[float]] = {}
        for r in parsed:
            model = r["model"]  # type: ignore[index]
            metrics = r["metrics"]  # type: ignore[index]
            series_t[model] = [metrics.get(m, {}).get(threshold, {}).get("FS", 0.0) for m in METHOD_ORDER]  # type: ignore[union-attr]
        draw_line_chart(
            os.path.join(out_dir, f"f1_vs_topk_t{str(threshold).replace('.', '')}.svg"),
            f"F1 vs Top-K (Threshold {threshold})",
            x_methods,
            labels_methods,
            series_t,
            0.0,
            1.05,
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "F1 Score",
        )

    # Chart 3: runtime breakdown.
    models = [r["model"] for r in parsed]  # type: ignore[index]
    timing = [r["timing"] for r in parsed]  # type: ignore[index]
    load = [t.get("Model Loading", 0.0) / 60.0 for t in timing]
    emb = [(t.get("Symptom Embeddings", 0.0) + t.get("Diagnosis Embeddings", 0.0)) / 60.0 for t in timing]
    folds = [t.get("Total Folds Processing", 0.0) / 60.0 for t in timing]
    total = [t.get("TOTAL EXECUTION TIME", 0.0) / 60.0 for t in timing]
    draw_stacked_runtime_chart(
        os.path.join(out_dir, "runtime_breakdown.svg"),
        "Runtime Breakdown (Minutes)",
        models,
        load,
        emb,
        folds,
        total,
    )

    # Chart 4: saturation diagnostic.
    summary_path = os.path.join(root, "docs", "score_distribution_analysis", "score_distribution_summary.txt")
    saturation = parse_saturation_summary(summary_path)
    sat_series: Dict[str, List[float]] = {}
    for model in models:
        points = saturation.get(model, {})
        sat_series[model] = [points.get(t, 0.0) for t in THRESHOLDS]

    draw_line_chart(
        os.path.join(out_dir, "saturation_by_threshold.svg"),
        "Per-Patient Saturation: % MAX Similarity >= Threshold",
        THRESHOLDS,
        [str(t) for t in THRESHOLDS],
        sat_series,
        0.0,
        100.0,
        [0, 20, 40, 60, 80, 100],
        "Percent",
    )

    print("Generated README plots in docs/readme_plots/:")
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".svg"):
            print(f"  - {os.path.join(out_dir, name)}")


if __name__ == "__main__":
    main()
