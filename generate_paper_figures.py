import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "blue": "#2E86AB",
    "orange": "#F18F01",
    "red": "#C73E1D",
    "green": "#59A14F",
    "teal": "#76B7B2",
    "gray": "#6C757D",
    "light_blue": "#D8EAF4",
    "light_orange": "#FCE6C9",
    "light_green": "#DFF0D8",
    "light_gray": "#F4F5F7",
}


DEFAULT_MAIN_RESULTS = [
    {
        "method": "SR-VFI baseline",
        "all7": {"PSNR": 29.716, "SSIM": 0.910},
        "odd4": {"PSNR": 30.825, "SSIM": 0.917},
        "even3": {"PSNR": 28.237, "SSIM": 0.901},
    },
    {
        "method": "RAFT fusion + edge",
        "all7": {"PSNR": 29.745, "SSIM": 0.911},
        "odd4": {"PSNR": 30.825, "SSIM": 0.917},
        "even3": {"PSNR": 28.305, "SSIM": 0.902},
    },
]


DEFAULT_ABLATION = [
    {"method": "LDMVFI pred", "PSNR": 30.7474, "SSIM": 0.9391},
    {"method": "Farneback fusion", "PSNR": 30.8399, "SSIM": 0.9400},
    {"method": "RAFT fusion", "PSNR": 30.8599, "SSIM": 0.9404},
    {"method": "RAFT fusion + edge", "PSNR": 30.8607, "SSIM": 0.9404},
    {"method": "Confidence-gated + edge", "PSNR": 30.8547, "SSIM": 0.9404},
]


DEFAULT_ORACLE = [
    {"method": "LDMVFI pred", "PSNR": 30.7474, "SSIM": 0.9391},
    {"method": "Farneback pixel oracle", "PSNR": 33.4681, "SSIM": 0.9603},
    {"method": "RAFT pixel oracle", "PSNR": 33.6767, "SSIM": 0.9612},
]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
        }
    )


def parse_formats(value: str) -> List[str]:
    formats = [item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()]
    allowed = {"png", "svg", "pdf"}
    bad = sorted(set(formats) - allowed)
    if bad:
        raise argparse.ArgumentTypeError(f"Unsupported output format(s): {', '.join(bad)}")
    return formats or ["png", "svg"]


def metric_text(metrics: Dict[str, Optional[float]], name: str) -> str:
    value = metrics.get(name)
    if value is None:
        return "-"
    if name.upper() == "SSIM" or name.upper() == "LPIPS":
        return f"{value:.3f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def present_metrics(rows: Sequence[Dict[str, object]], groups: Sequence[str]) -> List[str]:
    metrics = ["PSNR", "SSIM", "LPIPS"]
    visible = []
    for metric in metrics:
        if any(row.get(group, {}).get(metric) is not None for row in rows for group in groups):
            visible.append(metric)
    return visible or ["PSNR", "SSIM"]


def maybe_summary_dir(path: Path) -> Path:
    return path / "_summaries" if (path / "_summaries").is_dir() else path


def load_eval_overall(summary_root: str) -> Dict[str, Dict[str, float]]:
    summary_dir = maybe_summary_dir(Path(summary_root))
    paths = sorted(summary_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No summary json files found under {summary_dir}")
    summaries = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    if "averages" not in summaries[0]:
        raise ValueError(f"{paths[0]} does not contain grouped averages")

    groups = list(summaries[0]["averages"].keys())
    overall: Dict[str, Dict[str, float]] = {}
    for group in groups:
        overall[group] = {}
        metric_names = list(summaries[0]["averages"][group].keys())
        for metric in metric_names:
            weighted_sum = 0.0
            weight_sum = 0
            for item in summaries:
                value = item["averages"][group].get(metric)
                if value is None:
                    continue
                weight = item.get("metric_frame_counts", {}).get(group, {}).get(metric, item["num_samples"])
                weighted_sum += value * weight
                weight_sum += weight
            if weight_sum:
                overall[group][metric] = round(weighted_sum / weight_sum, 3)
    return overall


def main_result_rows(args) -> List[Dict[str, object]]:
    rows = []
    if args.baseline_summary_dir:
        rows.append({"method": args.baseline_name, **load_eval_overall(args.baseline_summary_dir)})
    if args.corrector_summary_dir:
        rows.append({"method": args.corrector_name, **load_eval_overall(args.corrector_summary_dir)})
    return rows or DEFAULT_MAIN_RESULTS


def load_metric_rows(path: str) -> List[Dict[str, float]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("rows", data)
    else:
        with input_path.open("r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

    parsed_rows = []
    for row in rows:
        normalized = {str(key).strip().lower(): value for key, value in row.items()}
        parsed_rows.append(
            {
                "method": str(normalized["method"]),
                "PSNR": float(normalized["psnr"]),
                "SSIM": float(normalized["ssim"]),
            }
        )
    return parsed_rows


def save_figure(fig, output_base: Path, formats: Iterable[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"))
    plt.close(fig)


def draw_box(ax, xy, width, height, text, facecolor, edgecolor=COLORS["blue"], lw=1.8):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", weight="bold")


def draw_arrow(ax, start, end, color=COLORS["gray"], rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def plot_framework(output_dir: Path, formats: Iterable[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    draw_box(ax, (0.3, 2.7), 1.7, 1.1, "LR reference\nim1, im3, im5, im7", COLORS["light_gray"])
    draw_box(ax, (2.5, 2.7), 1.6, 1.1, "RVRT\nsuper-resolution", COLORS["light_blue"])
    draw_box(ax, (4.6, 2.7), 1.8, 1.1, "SR reference\nodd4 frames", COLORS["light_gray"])
    draw_box(ax, (6.9, 2.7), 1.7, 1.1, "LDMVFI\ninterpolation", COLORS["light_orange"], COLORS["orange"])
    draw_box(ax, (9.1, 2.7), 2.3, 1.1, "Initial HR even\nim2, im4, im6", COLORS["light_gray"])

    draw_box(ax, (6.7, 0.75), 2.2, 1.1, "RAFT-aligned\nneighbor candidates", COLORS["light_blue"])
    draw_box(ax, (9.2, 0.75), 2.0, 1.1, "ReFIn-style\nfusion corrector", COLORS["light_orange"], COLORS["red"], lw=2.4)
    draw_box(ax, (5.0, 0.75), 1.3, 1.1, "GT HR\nloss", COLORS["light_green"], COLORS["green"])
    draw_box(ax, (9.7, 4.05), 1.6, 0.55, "HR 7 frames", COLORS["light_green"], COLORS["green"])

    draw_arrow(ax, (2.0, 3.25), (2.5, 3.25))
    draw_arrow(ax, (4.1, 3.25), (4.6, 3.25))
    draw_arrow(ax, (6.4, 3.25), (6.9, 3.25))
    draw_arrow(ax, (8.6, 3.25), (9.1, 3.25))
    draw_arrow(ax, (5.5, 2.7), (7.5, 1.85), color=COLORS["blue"], rad=-0.15)
    draw_arrow(ax, (8.9, 1.3), (9.2, 1.3), color=COLORS["blue"])
    draw_arrow(ax, (10.2, 2.7), (10.2, 1.85), color=COLORS["red"])
    draw_arrow(ax, (6.3, 1.3), (9.2, 1.3), color=COLORS["green"], rad=-0.1)
    draw_arrow(ax, (10.2, 3.8), (10.2, 4.05), color=COLORS["green"])

    ax.text(0.3, 4.35, "SR -> VFI with even-frame fusion refinement", fontsize=15, weight="bold")
    ax.text(
        0.3,
        0.18,
        "The corrector refines only interpolated even frames; odd reference frames come directly from RVRT SR.",
        color=COLORS["gray"],
    )
    save_figure(fig, output_dir / "fig3_method_framework", formats)


def write_main_result_tables(rows: List[Dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = ["all7", "odd4", "even3"]
    metrics = present_metrics(rows, groups)
    header = ["Method"] + [f"{group} {metric}" for group in groups for metric in metrics]

    csv_path = output_dir / "table_full_chain_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [row["method"]]
                + [metric_text(row.get(group, {}), metric) for group in groups for metric in metrics]
            )

    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        values = [str(row["method"])] + [metric_text(row.get(group, {}), metric) for group in groups for metric in metrics]
        md_lines.append("| " + " | ".join(values) + " |")
    (output_dir / "table_full_chain_results.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    latex_lines = [
        "\\begin{tabular}{l" + "c" * (len(groups) * len(metrics)) + "}",
        "\\toprule",
        "Method & "
        + " & ".join(
            f"\\multicolumn{{{len(metrics)}}}{{c}}{{{label}}}"
            for label in ("all7", "odd4/ref4", "even3/inter3")
        )
        + " \\\\",
        " & " + " & ".join(metric for _ in groups for metric in metrics) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [str(row["method"])] + [metric_text(row.get(group, {}), metric) for group in groups for metric in metrics]
        latex_lines.append(" & ".join(values) + " \\\\")
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_dir / "table_full_chain_results.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")


def plot_main_result_table(rows: List[Dict[str, object]], output_dir: Path, formats: Iterable[str]) -> None:
    groups = ["all7", "odd4", "even3"]
    metrics = present_metrics(rows, groups)
    columns = ["Method"] + [f"{group}\n{metric}" for group in groups for metric in metrics]
    cell_text = []
    for row in rows:
        cell_text.append(
            [row["method"]] + [metric_text(row.get(group, {}), metric) for group in groups for metric in metrics]
        )

    fig, ax = plt.subplots(figsize=(13, 1.6 + 0.42 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D4DC")
        if row_idx == 0:
            cell.set_facecolor("#E8ECF3")
            cell.set_text_props(weight="bold")
        elif col_idx == 0:
            cell.set_facecolor("#F8F8F6")
            cell.set_text_props(ha="left")
    ax.set_title("Full-chain evaluation summary", pad=18, weight="bold")
    save_figure(fig, output_dir / "table_full_chain_results", formats)


def plot_odd_even_gap(args, output_dir: Path, formats: Iterable[str]) -> None:
    labels = ["odd4 / ref4", "even3 / inter3"]
    values = [args.gap_odd4_psnr, args.gap_even3_psnr]
    colors = [COLORS["blue"], COLORS["orange"]]
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Quality gap between reference and interpolated frames", weight="bold")
    ax.set_ylim(min(values) - 1.2, max(values) + 1.0)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.3f}", ha="center", va="bottom")
    gap = values[0] - values[1]
    ax.text(0.5, max(values) + 0.55, f"Gap: {gap:.3f} dB", ha="center", color=COLORS["red"], weight="bold")
    save_figure(fig, output_dir / "fig4_odd_even_gap", formats)


def plot_metric_bars(rows: List[Dict[str, float]], title: str, output_base: Path, formats: Iterable[str]) -> None:
    labels = [row["method"] for row in rows]
    x = list(range(len(rows)))
    psnr = [row["PSNR"] for row in rows]
    ssim = [row["SSIM"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, values, metric, color in [
        (axes[0], psnr, "PSNR (dB)", COLORS["blue"]),
        (axes[1], ssim, "SSIM", COLORS["orange"]),
    ]:
        bars = ax.bar(x, values, color=color, edgecolor="white", linewidth=1.1)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel(metric)
        ax.set_ylim(min(values) - (0.08 if metric.startswith("PSNR") else 0.002), max(values) + (0.08 if metric.startswith("PSNR") else 0.002))
        for bar, value in zip(bars, values):
            text = f"{value:.3f}" if metric == "SSIM" else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value, text, ha="center", va="bottom", fontsize=8)
    fig.suptitle(title, weight="bold")
    fig.tight_layout()
    save_figure(fig, output_base, formats)


def load_oracle_rows(args) -> List[Dict[str, float]]:
    if not args.raft_diagnosis_json and not args.farneback_diagnosis_json:
        return DEFAULT_ORACLE

    rows: List[Dict[str, float]] = []
    pred_added = False
    for label, path in [("Farneback pixel oracle", args.farneback_diagnosis_json), ("RAFT pixel oracle", args.raft_diagnosis_json)]:
        if not path:
            continue
        with Path(path).open("r", encoding="utf-8") as f:
            item = json.load(f)
        metrics = item["metrics"]
        if not pred_added and "pred" in metrics:
            rows.append({"method": "LDMVFI pred", "PSNR": metrics["pred"]["PSNR"], "SSIM": metrics["pred"]["SSIM"]})
            pred_added = True
        oracle = metrics.get("pixel_oracle")
        if oracle:
            rows.append({"method": label, "PSNR": oracle["PSNR"], "SSIM": oracle["SSIM"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis/paper figures for the RVRT + LDMVFI corrector experiments.")
    parser.add_argument("--output-dir", default="figures/paper", help="Directory for generated figures and tables.")
    parser.add_argument("--formats", type=parse_formats, default=parse_formats("png,svg"), help="Comma-separated formats: png,svg,pdf.")
    parser.add_argument("--baseline-summary-dir", default="", help="Eval output root or _summaries dir for the baseline full-chain table.")
    parser.add_argument("--corrector-summary-dir", default="", help="Eval output root or _summaries dir for the best corrector full-chain table.")
    parser.add_argument("--baseline-name", default="SR-VFI baseline")
    parser.add_argument("--corrector-name", default="RAFT fusion + edge")
    parser.add_argument("--gap-odd4-psnr", type=float, default=35.916)
    parser.add_argument("--gap-even3-psnr", type=float, default=30.747)
    parser.add_argument("--raft-diagnosis-json", default="", help="Optional RAFT diagnose_even_corrector_cache summary json.")
    parser.add_argument("--farneback-diagnosis-json", default="", help="Optional Farneback diagnose_even_corrector_cache summary json.")
    parser.add_argument(
        "--ablation-results",
        default="",
        help="Optional CSV/JSON with columns/keys: method, PSNR, SSIM. If omitted, built-in draft ablation values are used.",
    )
    args = parser.parse_args()

    setup_style()
    output_dir = Path(args.output_dir)
    rows = main_result_rows(args)

    plot_framework(output_dir, args.formats)
    write_main_result_tables(rows, output_dir)
    plot_main_result_table(rows, output_dir, args.formats)
    plot_odd_even_gap(args, output_dir, args.formats)
    ablation_rows = load_metric_rows(args.ablation_results) if args.ablation_results else DEFAULT_ABLATION
    plot_metric_bars(ablation_rows, "Corrector ablation on interpolated even frames", output_dir / "fig5_corrector_ablation", args.formats)
    plot_metric_bars(load_oracle_rows(args), "Candidate/oracle diagnosis", output_dir / "fig6_candidate_oracle", args.formats)

    print(f"Generated paper figures under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
