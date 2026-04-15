import argparse
import os
from dataclasses import dataclass
from typing import List

from PIL import Image, ImageDraw, ImageFont


BG = (248, 248, 246)
TEXT = (24, 24, 24)
MUTED = (90, 90, 90)
GRID = (210, 214, 222)
HEADER_BG = (232, 236, 244)
SUBHEADER_BG = (240, 242, 247)
BEST_BG = (226, 242, 226)
METHOD_BG = (255, 255, 255)
ACCENT = (196, 68, 44)


@dataclass
class ResultRow:
    method: str
    slow_psnr: float
    slow_ssim: float
    medium_psnr: float
    medium_ssim: float
    fast_psnr: float
    fast_ssim: float
    overall_psnr: float
    overall_ssim: float


def load_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def parse_row(value: str) -> ResultRow:
    parts = [p.strip() for p in value.split("|")]
    if len(parts) != 9:
        raise argparse.ArgumentTypeError(
            "Row must be METHOD|slow_psnr|slow_ssim|medium_psnr|medium_ssim|fast_psnr|fast_ssim|overall_psnr|overall_ssim"
        )
    try:
        nums = [float(x) for x in parts[1:]]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric values in row: {value}") from exc
    return ResultRow(parts[0], *nums)


def draw_centered(draw, box, text, font, fill=TEXT):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2), text, font=font, fill=fill)


def draw_left(draw, box, text, font, fill=TEXT, pad=14):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    h = bbox[3] - bbox[1]
    draw.text((x0 + pad, y0 + (y1 - y0 - h) / 2), text, font=font, fill=fill)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation table figure.")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default="Evaluation Summary", help="Figure title")
    parser.add_argument("--subtitle", default="", help="Optional subtitle")
    parser.add_argument(
        "--row",
        action="append",
        type=parse_row,
        required=True,
        help="METHOD|slow_psnr|slow_ssim|medium_psnr|medium_ssim|fast_psnr|fast_ssim|overall_psnr|overall_ssim",
    )
    args = parser.parse_args()

    rows: List[ResultRow] = args.row
    title_font = load_font(42)
    subtitle_font = load_font(24)
    header_font = load_font(24)
    method_font = load_font(24)
    cell_font = load_font(23)

    method_w = 390
    metric_w = 120
    group_w = metric_w * 2
    left = 60
    top = 48
    title_h = 70
    subtitle_h = 34 if args.subtitle else 0
    header_h1 = 46
    header_h2 = 42
    row_h = 52
    table_y = top + title_h + subtitle_h + 28
    width = left * 2 + method_w + group_w * 4
    height = table_y + header_h1 + header_h2 + row_h * len(rows) + 70

    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((left, top), args.title, font=title_font, fill=TEXT)
    if args.subtitle:
        draw.text((left, top + title_h - 8), args.subtitle, font=subtitle_font, fill=MUTED)

    x = left
    y = table_y
    # Top header row
    draw.rounded_rectangle((x, y, x + method_w, y + header_h1), radius=10, fill=HEADER_BG, outline=GRID, width=2)
    draw_left(draw, (x, y, x + method_w, y + header_h1), "Method", header_font)
    x += method_w
    for label in ("Slow Test", "Medium Test", "Fast Test", "Overall"):
        draw.rounded_rectangle((x, y, x + group_w, y + header_h1), radius=10, fill=HEADER_BG, outline=GRID, width=2)
        draw_centered(draw, (x, y, x + group_w, y + header_h1), label, header_font)
        x += group_w

    # Subheader row
    x = left
    y += header_h1
    draw.rounded_rectangle((x, y, x + method_w, y + header_h2), radius=10, fill=SUBHEADER_BG, outline=GRID, width=2)
    x += method_w
    for _ in range(4):
        draw.rounded_rectangle((x, y, x + metric_w, y + header_h2), radius=10, fill=SUBHEADER_BG, outline=GRID, width=2)
        draw_centered(draw, (x, y, x + metric_w, y + header_h2), "PSNR", header_font)
        x += metric_w
        draw.rounded_rectangle((x, y, x + metric_w, y + header_h2), radius=10, fill=SUBHEADER_BG, outline=GRID, width=2)
        draw_centered(draw, (x, y, x + metric_w, y + header_h2), "SSIM", header_font)
        x += metric_w

    # Best markers
    best = {
        "slow_psnr": max(r.slow_psnr for r in rows),
        "slow_ssim": max(r.slow_ssim for r in rows),
        "medium_psnr": max(r.medium_psnr for r in rows),
        "medium_ssim": max(r.medium_ssim for r in rows),
        "fast_psnr": max(r.fast_psnr for r in rows),
        "fast_ssim": max(r.fast_ssim for r in rows),
        "overall_psnr": max(r.overall_psnr for r in rows),
        "overall_ssim": max(r.overall_ssim for r in rows),
    }

    # Data rows
    y += header_h2
    fields = [
        "slow_psnr", "slow_ssim", "medium_psnr", "medium_ssim",
        "fast_psnr", "fast_ssim", "overall_psnr", "overall_ssim",
    ]
    for row in rows:
        x = left
        draw.rounded_rectangle((x, y, x + method_w, y + row_h), radius=10, fill=METHOD_BG, outline=GRID, width=2)
        draw_left(draw, (x, y, x + method_w, y + row_h), row.method, method_font)
        x += method_w
        for field in fields:
            value = getattr(row, field)
            fill = BEST_BG if abs(value - best[field]) < 1e-9 else METHOD_BG
            draw.rounded_rectangle((x, y, x + metric_w, y + row_h), radius=10, fill=fill, outline=GRID, width=2)
            text = f"{value:.3f}" if "ssim" in field else f"{value:.3f}".rstrip("0").rstrip(".")
            draw_centered(draw, (x, y, x + metric_w, y + row_h), text, cell_font)
            x += metric_w
        y += row_h

    note = "Green cells mark the best value in each metric column."
    draw.text((left, height - 40), note, font=subtitle_font, fill=ACCENT)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    canvas.save(args.output)
    print(f"Saved evaluation table figure to {args.output}")


if __name__ == "__main__":
    main()
