import argparse
import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


BG = (250, 250, 248)
TEXT = (24, 24, 24)
MUTED = (90, 90, 90)
EDGE = (55, 75, 120)
BOX = (255, 255, 255)
BOX_BORDER = (200, 205, 215)
FROZEN = (225, 236, 255)
TRAINABLE = (255, 236, 220)
ACCENT = (196, 68, 44)


def load_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text_to_width(draw, text: str, font, max_width: int):
    wrapped_lines = []
    for raw_line in text.split("\n"):
        words = raw_line.split()
        if not words:
            wrapped_lines.append("")
            continue
        cur = words[0]
        for word in words[1:]:
            trial = f"{cur} {word}"
            bbox = draw.textbbox((0, 0), trial, font=font)
            if bbox[2] - bbox[0] <= max_width:
                cur = trial
            else:
                wrapped_lines.append(cur)
                cur = word
        wrapped_lines.append(cur)
    return wrapped_lines


def draw_centered_text(draw, box: Tuple[int, int, int, int], text: str, font, fill=TEXT, line_gap=6, padding=16):
    x0, y0, x1, y1 = box
    lines = wrap_text_to_width(draw, text, font, max_width=(x1 - x0 - 2 * padding))
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    heights = [b[3] - b[1] for b in bboxes]
    widths = [b[2] - b[0] for b in bboxes]
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    cur_y = y0 + (y1 - y0 - total_h) // 2
    for line, h, w in zip(lines, heights, widths):
        cur_x = x0 + (x1 - x0 - w) // 2
        draw.text((cur_x, cur_y), line, font=font, fill=fill)
        cur_y += h + line_gap


def draw_box(draw, xy, text, font, fill, outline=BOX_BORDER):
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    draw_centered_text(draw, xy, text, font)


def draw_arrow(draw, start, end, color=EDGE, width=6, head_len=18, head_w=10):
    x0, y0 = start
    x1, y1 = end
    draw.line([x0, y0, x1, y1], fill=color, width=width)
    if x1 == x0 and y1 == y0:
        return
    import math
    ang = math.atan2(y1 - y0, x1 - x0)
    left = (
        x1 - head_len * math.cos(ang) + head_w * math.sin(ang),
        y1 - head_len * math.sin(ang) - head_w * math.cos(ang),
    )
    right = (
        x1 - head_len * math.cos(ang) - head_w * math.sin(ang),
        y1 - head_len * math.sin(ang) + head_w * math.cos(ang),
    )
    draw.polygon([(x1, y1), left, right], fill=color)


def main():
    parser = argparse.ArgumentParser(description="Generate RVRT + LDMVFI structure diagram.")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default="RVRT + LDMVFI Pipeline", help="Diagram title")
    parser.add_argument("--show-lora", action="store_true", help="Annotate LoRA on diffusion UNet")
    args = parser.parse_args()

    W, H = 1880, 920
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(40)
    box_font = load_font(25)
    note_font = load_font(24)
    small_font = load_font(20)

    draw.text((70, 40), args.title, font=title_font, fill=TEXT)
    draw.text((70, 96), "Two-stage stitching: spatial super-resolution first, temporal interpolation second", font=note_font, fill=MUTED)

    # Top flow
    prev_lr = (90, 205, 320, 325)
    next_lr = (90, 385, 320, 505)
    rvrt = (410, 270, 710, 440)
    prev_sr = (810, 205, 1040, 325)
    next_sr = (810, 385, 1040, 505)
    cond = (1140, 255, 1470, 455)
    unet = (1560, 170, 1820, 350)
    vq = (1560, 435, 1820, 615)
    out = (1560, 700, 1820, 820)

    draw_box(draw, prev_lr, "Prev LR\nim3.png", box_font, BOX)
    draw_box(draw, next_lr, "Next LR\nim5.png", box_font, BOX)
    draw_box(draw, rvrt, "RVRT\nVideo Super-Resolution", box_font, FROZEN)
    draw_box(draw, prev_sr, "Prev SR", box_font, BOX)
    draw_box(draw, next_sr, "Next SR", box_font, BOX)
    draw_box(draw, cond, "Condition Encoding\nget_learned_conditioning()", box_font, BOX)

    unet_text = "Latent Diffusion UNet"
    if args.show_lora:
        unet_text += "\n+ LoRA"
    draw_box(draw, unet, unet_text, box_font, TRAINABLE if args.show_lora else FROZEN)
    draw_box(draw, vq, "VQFlow First-Stage\nEncoder / Decoder", box_font, FROZEN)
    draw_box(draw, out, "Predicted HR Middle\nim4.png", box_font, BOX)

    # Arrows
    draw_arrow(draw, (320, 260), (410, 315))
    draw_arrow(draw, (320, 450), (410, 395))
    draw_arrow(draw, (710, 315), (810, 260))
    draw_arrow(draw, (710, 395), (810, 450))
    draw_arrow(draw, (1040, 260), (1140, 315))
    draw_arrow(draw, (1040, 450), (1140, 395))
    draw_arrow(draw, (1470, 355), (1560, 260))
    draw_arrow(draw, (1690, 350), (1690, 435))
    draw_arrow(draw, (1690, 615), (1690, 700))

    # Side notes
    draw.text((440, 455), "Frozen in current setup", font=small_font, fill=ACCENT)
    draw.text((1550, 635), "Decode latent to final HR frame", font=small_font, fill=MUTED)
    draw.text((1130, 485), "SR neighbors are used as LDMVFI conditions", font=small_font, fill=MUTED)

    # Bottom summary strip
    strip = (70, 850, 1810, 895)
    draw.rounded_rectangle(strip, radius=12, fill=(240, 242, 246), outline=BOX_BORDER, width=2)
    summary = (
        "Current stitching: LR prev/next -> RVRT -> SR prev/next -> LDMVFI -> HR middle. "
        "Training strategy: RVRT and VQFlow frozen; diffusion UNet frozen or LoRA-tuned depending on experiment."
    )
    draw_centered_text(draw, strip, summary, small_font, fill=TEXT, line_gap=4, padding=20)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    canvas.save(args.output)
    print(f"Saved structure diagram to {args.output}")


if __name__ == "__main__":
    main()
