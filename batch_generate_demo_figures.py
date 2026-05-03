#!/usr/bin/env python3
"""Batch-generate qualitative demo figures from saved full-chain predictions.

This is a thin wrapper around generate_demo_figure.py. It discovers common
samples from two result directories, optionally selects zoom crops
automatically, and writes one stitched comparison figure per sample.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Crop must be x,y,w,h, got: {value}")
    try:
        x, y, w, h = [int(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Crop must contain integers: {value}") from exc
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"Crop must be non-negative x/y and positive w/h: {value}")
    return x, y, w, h


def parse_crop_size(value: str) -> tuple[int, int]:
    parts = [item.strip() for item in value.lower().split("x")]
    if len(parts) == 1:
        parts = [parts[0], parts[0]]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Crop size must be N or WxH, got: {value}")
    try:
        w, h = [int(item) for item in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Crop size must contain integers: {value}") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"Crop size must be positive: {value}")
    return w, h


def rel_to_sample(rel: Path, split: str) -> str | None:
    parts = rel.parts
    if split and parts and parts[0] == split:
        parts = parts[1:]
    if len(parts) < 2:
        return None
    # Vimeo sample ids are two-level paths such as 00001/0625.
    return Path(*parts[-2:]).as_posix()


def discover_samples(root: Path, split: str, filename: str) -> set[str]:
    roots: list[Path] = []
    if split and (root / split).is_dir():
        roots.append(root / split)
    roots.append(root)

    samples: set[str] = set()
    for base in roots:
        if not base.is_dir():
            continue
        for path in base.rglob(filename):
            try:
                rel = path.parent.relative_to(root)
            except ValueError:
                rel = path.parent.relative_to(base)
            sample = rel_to_sample(rel, split)
            if sample:
                samples.add(sample)
    return samples


def resolve_image(root: Path, sample: str, split: str, filename: str) -> Path:
    candidates = []
    if split:
        candidates.append(root / split / sample / filename)
    candidates.append(root / sample / filename)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find {filename} for {sample} under {root}")


def load_rgb(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size and img.size != size:
        img = img.resize(size, Image.Resampling.BICUBIC)
    return np.asarray(img).astype(np.float32)


def integral_image(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)


def window_sum(integral: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    return float(integral[y + h, x + w] - integral[y, x + w] - integral[y + h, x] + integral[y, x])


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    inter = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def clamp_crop(crop: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = crop
    w = min(w, width)
    h = min(h, height)
    x = min(max(0, x), max(0, width - w))
    y = min(max(0, y), max(0, height - h))
    return x, y, w, h


def select_auto_crops(
    gt_path: Path,
    baseline_path: Path,
    corrector_path: Path,
    crop_size: tuple[int, int],
    count: int,
    stride: int,
    mode: str,
) -> list[tuple[int, int, int, int]]:
    gt_img = Image.open(gt_path).convert("RGB")
    width, height = gt_img.size
    crop_w, crop_h = min(crop_size[0], width), min(crop_size[1], height)

    if mode == "center":
        return [((width - crop_w) // 2, (height - crop_h) // 2, crop_w, crop_h)]

    gt = np.asarray(gt_img).astype(np.float32)
    baseline = load_rgb(baseline_path, gt_img.size)
    corrector = load_rgb(corrector_path, gt_img.size)

    baseline_err = np.abs(gt - baseline).mean(axis=2)
    corrector_err = np.abs(gt - corrector).mean(axis=2)
    improvement = baseline_err - corrector_err

    gray = gt.mean(axis=2)
    grad = np.zeros_like(gray)
    grad[:, 1:] += np.abs(gray[:, 1:] - gray[:, :-1])
    grad[1:, :] += np.abs(gray[1:, :] - gray[:-1, :])

    if mode == "improvement":
        score = np.maximum(improvement, 0.0) + 0.15 * baseline_err + 0.03 * grad
    elif mode == "error":
        score = baseline_err + 0.03 * grad
    elif mode == "texture":
        score = grad
    else:
        raise ValueError(f"Unsupported auto crop mode: {mode}")

    integ = integral_image(score)
    step = max(1, stride)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for y in range(0, max(1, height - crop_h + 1), step):
        for x in range(0, max(1, width - crop_w + 1), step):
            mean_score = window_sum(integ, x, y, crop_w, crop_h) / (crop_w * crop_h)
            candidates.append((mean_score, (x, y, crop_w, crop_h)))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[int, int, int, int]] = []
    for _, crop in candidates:
        if all(iou(crop, old) < 0.35 for old in selected):
            selected.append(crop)
        if len(selected) >= count:
            break
    return selected


def read_sample_file(path: Path) -> list[str]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        samples.append(line)
    return samples


def choose_samples(args: argparse.Namespace, filename: str) -> list[str]:
    explicit: list[str] = []
    explicit.extend(args.sample_relpath or [])
    if args.sample_file:
        explicit.extend(read_sample_file(Path(args.sample_file)))
    if explicit:
        return sorted(dict.fromkeys(explicit))

    baseline = discover_samples(Path(args.baseline_root), args.split, filename)
    corrector = discover_samples(Path(args.corrector_root), args.split, filename)
    samples = sorted(baseline & corrector)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    return samples


def build_demo_command(
    args: argparse.Namespace,
    sample: str,
    output: Path,
    crops: Iterable[tuple[int, int, int, int]],
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("generate_demo_figure.py")),
        "--sample-relpath",
        sample,
        "--split",
        args.split,
        "--dataset-root-lr",
        args.dataset_root_lr,
        "--dataset-root-hr",
        args.dataset_root_hr,
        "--method",
        f"{args.baseline_label}={args.baseline_root}",
        "--method",
        f"{args.corrector_label}={args.corrector_root}",
        "--target-frame",
        str(args.target_frame),
        "--error-scale",
        str(args.error_scale),
        "--panel-size",
        str(args.panel_size),
        "--gap",
        str(args.gap),
        "--output",
        str(output),
    ]
    if args.title:
        cmd.extend(["--title", args.title.format(sample=sample, split=args.split, frame=args.target_frame)])
    if args.show_error_maps:
        cmd.append("--show-error-maps")
    for crop in crops:
        cmd.extend(["--crop", ",".join(str(item) for item in crop)])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate demo figures for qualitative comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--corrector-root", required=True)
    parser.add_argument("--dataset-root-lr", required=True)
    parser.add_argument("--dataset-root-hr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="fast_test")
    parser.add_argument("--target-frame", type=int, choices=[2, 4, 6], default=4)
    parser.add_argument("--max-samples", type=int, default=12, help="0 means all discovered samples.")
    parser.add_argument("--sample-relpath", action="append", help="Explicit sample, e.g. 00001/0625. Repeatable.")
    parser.add_argument("--sample-file", default="", help="Optional text file with one sample relpath per line.")
    parser.add_argument("--baseline-label", default="SR-VFI baseline")
    parser.add_argument("--corrector-label", default="RAFT fusion corrector")
    parser.add_argument("--title", default="{split} {sample} im{frame}")
    parser.add_argument("--show-error-maps", type=int, choices=[0, 1], default=1)
    parser.add_argument("--error-scale", type=float, default=4.0)
    parser.add_argument("--crop", action="append", type=parse_crop, default=[], help="Manual crop x,y,w,h. Repeatable.")
    parser.add_argument("--auto-crop-count", type=int, default=2)
    parser.add_argument("--auto-crop-size", type=parse_crop_size, default=(96, 96))
    parser.add_argument("--auto-crop-stride", type=int, default=16)
    parser.add_argument("--auto-crop-mode", choices=["improvement", "error", "texture", "center"], default="improvement")
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--gap", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filename = f"output_im{args.target_frame}.png"
    samples = choose_samples(args, filename)
    if not samples:
        raise RuntimeError("No common samples found. Check result roots, split, and target frame.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"samples={len(samples)} output_dir={output_dir}")

    for index, sample in enumerate(samples, start=1):
        safe_sample = sample.replace("/", "_")
        output = output_dir / f"qual_{args.split}_{safe_sample}_im{args.target_frame}.png"

        crops = list(args.crop)
        if not crops and args.auto_crop_count > 0:
            gt_path = resolve_image(Path(args.dataset_root_hr), sample, "", f"im{args.target_frame}.png")
            baseline_path = resolve_image(Path(args.baseline_root), sample, args.split, filename)
            corrector_path = resolve_image(Path(args.corrector_root), sample, args.split, filename)
            crops = select_auto_crops(
                gt_path,
                baseline_path,
                corrector_path,
                args.auto_crop_size,
                args.auto_crop_count,
                args.auto_crop_stride,
                args.auto_crop_mode,
            )
        else:
            gt_path = resolve_image(Path(args.dataset_root_hr), sample, "", f"im{args.target_frame}.png")
            with Image.open(gt_path) as img:
                width, height = img.size
            crops = [clamp_crop(crop, width, height) for crop in crops]

        cmd = build_demo_command(args, sample, output, crops)
        print(f"[{index}/{len(samples)}] {sample} crops={crops} -> {output}")
        if args.dry_run:
            print(" ".join(cmd))
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
