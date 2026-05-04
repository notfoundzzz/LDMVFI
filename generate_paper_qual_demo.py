#!/usr/bin/env python3
"""Generate a compact paper-ready qualitative comparison figure.

Default layout:

  columns: GT / SR-VFI baseline / RAFT fusion corrector
  rows: full im4 / zoom crop 1 / zoom crop 2

The script intentionally omits error maps because, for the current experiments,
the error maps are visually similar and make the figure less readable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


CANVAS_BG = (255, 255, 255)
PANEL_BG = (245, 245, 245)
TEXT = (20, 20, 20)
BORDER = (205, 205, 205)
COLORS = [(220, 60, 60), (45, 120, 210), (40, 150, 90)]


def parse_crop_spec(value: str) -> tuple[str, tuple[int, int, int, int]]:
    label = ""
    crop_text = value
    if "=" in value:
        label, crop_text = value.split("=", 1)
        label = label.strip()
    parts = [part.strip() for part in crop_text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Crop must be LABEL=x,y,w,h or x,y,w,h, got: {value}")
    try:
        x, y, w, h = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Crop contains non-integer value: {value}") from exc
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"Crop must have non-negative x/y and positive w/h: {value}")
    return label or f"Crop", (x, y, w, h)


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(["DejaVuSans-Bold.ttf", "Arial Bold.ttf"])
    candidates.extend(["DejaVuSans.ttf", "Arial.ttf"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resolve_image(root: str, relpath: str, filename: str, split: str = "") -> Path:
    candidates: list[Path] = []
    root_path = Path(root)
    if split:
        candidates.append(root_path / split / relpath / filename)
    candidates.append(root_path / relpath / filename)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find {filename} for {relpath} under {root}")


def fit_image(img: Image.Image, box_w: int, box_h: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    img = img.convert("RGB")
    canvas = Image.new("RGB", (box_w, box_h), PANEL_BG)
    w, h = img.size
    scale = min(box_w / w, box_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (box_w - new_w) // 2
    top = (box_h - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas, (left, top, new_w, new_h)


def draw_crop_boxes(
    draw: ImageDraw.ImageDraw,
    panel_left: int,
    panel_top: int,
    fitted_rect: tuple[int, int, int, int],
    orig_size: tuple[int, int],
    crops: Sequence[tuple[str, tuple[int, int, int, int]]],
    font,
) -> None:
    fit_left, fit_top, fit_w, fit_h = fitted_rect
    orig_w, orig_h = orig_size
    scale = min(fit_w / orig_w, fit_h / orig_h)
    for idx, (label, crop) in enumerate(crops, start=1):
        x, y, w, h = crop
        color = COLORS[(idx - 1) % len(COLORS)]
        left = panel_left + fit_left + int(round(x * scale))
        top = panel_top + fit_top + int(round(y * scale))
        right = panel_left + fit_left + int(round((x + w) * scale))
        bottom = panel_top + fit_top + int(round((y + h) * scale))
        draw.rectangle([left, top, right, bottom], outline=color, width=4)
        tag = str(idx)
        draw.rectangle([left, top, left + 26, top + 24], fill=color)
        draw.text((left + 7, top + 1), tag, fill=(255, 255, 255), font=font)


def crop_image(img: Image.Image, crop: tuple[int, int, int, int]) -> Image.Image:
    x, y, w, h = crop
    x = min(max(0, x), max(0, img.width - 1))
    y = min(max(0, y), max(0, img.height - 1))
    w = min(w, img.width - x)
    h = min(h, img.height - y)
    return img.crop((x, y, x + w, y + h))


def draw_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    img: Image.Image,
    left: int,
    top: int,
    panel_w: int,
    panel_h: int,
    border: bool = True,
) -> tuple[int, int, int, int]:
    fitted, fitted_rect = fit_image(img, panel_w, panel_h)
    canvas.paste(fitted, (left, top))
    if border:
        draw.rectangle([left, top, left + panel_w - 1, top + panel_h - 1], outline=BORDER, width=2)
    return fitted_rect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a clean paper-ready qualitative figure without error maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sample-relpath", default="00001/0625")
    parser.add_argument("--split", default="fast_test")
    parser.add_argument("--target-frame", type=int, choices=[2, 4, 6], default=4)
    parser.add_argument("--dataset-root-hr", default="/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sequences")
    parser.add_argument("--baseline-root", default="/data/Shenzhen/zhahongli/LDMVFI/qual_sr_then_vfi_baseline_fast")
    parser.add_argument("--corrector-root", default="/data/Shenzhen/zhahongli/LDMVFI/qual_sr_then_vfi_raft_corrector_fast")
    parser.add_argument("--baseline-label", default="SR-VFI baseline")
    parser.add_argument("--corrector-label", default="RAFT fusion corrector")
    parser.add_argument(
        "--crop",
        action="append",
        type=parse_crop_spec,
        default=[],
        help="Zoom crop in HR coordinates. Format: LABEL=x,y,w,h. Repeatable.",
    )
    parser.add_argument(
        "--output",
        default="/data/Shenzhen/zhahongli/LDMVFI/figures/paper/qual_paper_fast_00001_0625_im4.png",
    )
    parser.add_argument("--full-panel-width", type=int, default=330)
    parser.add_argument("--full-panel-height", type=int, default=230)
    parser.add_argument("--zoom-panel-size", type=int, default=250)
    parser.add_argument("--gap", type=int, default=26)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.target_frame
    output_filename = f"output_im{target}.png"

    crops = args.crop or [
        parse_crop_spec("Pen tip=176,104,112,112"),
        parse_crop_spec("Little finger=120,20,128,128"),
    ]

    gt = Image.open(resolve_image(args.dataset_root_hr, args.sample_relpath, f"im{target}.png")).convert("RGB")
    baseline = Image.open(resolve_image(args.baseline_root, args.sample_relpath, output_filename, args.split)).convert("RGB")
    corrector = Image.open(resolve_image(args.corrector_root, args.sample_relpath, output_filename, args.split)).convert("RGB")

    methods = [
        ("GT", gt),
        (args.baseline_label, baseline),
        (args.corrector_label, corrector),
    ]

    title_font = load_font(30, bold=True)
    header_font = load_font(24, bold=True)
    label_font = load_font(22)
    tag_font = load_font(18, bold=True)
    note_font = load_font(18)

    margin_x = 36
    margin_y = 34
    gap = args.gap
    full_w = args.full_panel_width
    full_h = args.full_panel_height
    zoom_size = args.zoom_panel_size
    col_w = max(full_w, zoom_size)
    width = margin_x * 2 + 3 * col_w + 2 * gap

    title_h = 42
    header_h = 34
    full_block_h = header_h + full_h
    zoom_header_h = 42
    zoom_block_h = header_h + zoom_size
    height = margin_y + title_h + gap + full_block_h + gap
    height += len(crops) * (zoom_header_h + zoom_block_h + gap)
    height += 8

    canvas = Image.new("RGB", (width, height), CANVAS_BG)
    draw = ImageDraw.Draw(canvas)

    title = f"{args.split} {args.sample_relpath} im{target}"
    draw.text((margin_x, margin_y), title, fill=TEXT, font=title_font)
    y = margin_y + title_h + gap

    # Column headers and full-frame comparison.
    for col, (name, img) in enumerate(methods):
        x = margin_x + col * (col_w + gap) + (col_w - full_w) // 2
        draw.text((x, y), name, fill=TEXT, font=header_font)
        fitted_rect = draw_panel(canvas, draw, img, x, y + header_h, full_w, full_h)
        draw_crop_boxes(draw, x, y + header_h, fitted_rect, img.size, crops, tag_font)

    y += full_block_h + gap

    for crop_idx, (crop_label, crop) in enumerate(crops, start=1):
        color = COLORS[(crop_idx - 1) % len(COLORS)]
        draw.rectangle([margin_x, y + 8, margin_x + 28, y + 36], fill=color)
        draw.text((margin_x + 8, y + 8), str(crop_idx), fill=(255, 255, 255), font=tag_font)
        draw.text((margin_x + 40, y + 8), crop_label, fill=TEXT, font=header_font)
        draw.text((margin_x + 210, y + 12), f"crop={crop[0]},{crop[1]},{crop[2]},{crop[3]}", fill=(100, 100, 100), font=note_font)
        y += zoom_header_h
        for col, (name, img) in enumerate(methods):
            x = margin_x + col * (col_w + gap) + (col_w - zoom_size) // 2
            if crop_idx == 1:
                draw.text((x, y), name, fill=TEXT, font=label_font)
            crop_panel = crop_image(img, crop)
            draw_panel(canvas, draw, crop_panel, x, y + header_h, zoom_size, zoom_size)
        y += zoom_block_h + gap

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
