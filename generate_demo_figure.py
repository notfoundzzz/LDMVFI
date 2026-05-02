import argparse
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


PANEL_BG = (245, 245, 245)
CANVAS_BG = (255, 255, 255)
TEXT = (20, 20, 20)
ACCENT = (220, 60, 60)
BORDER = (210, 210, 210)


@dataclass
class MethodSpec:
    label: str
    root: str


def parse_method_spec(value: str) -> MethodSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Method spec must be LABEL=DIR, got: {value}")
    label, root = value.split("=", 1)
    label = label.strip()
    root = root.strip()
    if not label or not root:
        raise argparse.ArgumentTypeError(f"Invalid method spec: {value}")
    return MethodSpec(label=label, root=root)


def parse_crop(value: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"Crop must be x,y,w,h, got: {value}")
    try:
        x, y, w, h = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Crop must contain integers: {value}") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"Crop width/height must be positive: {value}")
    return x, y, w, h


def parse_target_frame(value: str) -> int:
    try:
        frame_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Target frame must be an integer, got: {value}") from exc
    if frame_id not in (2, 4, 6):
        raise argparse.ArgumentTypeError("Target frame must be one of 2, 4, or 6 for even-frame comparison")
    return frame_id


def load_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resolve_dataset_image(root: str, relpath: str, filename: str, split: str = "") -> str:
    candidates = []
    if split:
        candidates.append(os.path.join(root, split, relpath, filename))
    candidates.append(os.path.join(root, relpath, filename))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Cannot find {filename} for sample {relpath} under {root}")


def resolve_method_image(root: str, relpath: str, filename: str, split: str = "") -> str:
    candidates = []
    if split:
        candidates.append(os.path.join(root, split, relpath, filename))
    candidates.append(os.path.join(root, relpath, filename))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Cannot find {filename} for sample {relpath} under {root}")


def fit_image(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    canvas = Image.new("RGB", (size, size), PANEL_BG)
    w, h = img.size
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def draw_labeled_panel(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    img: Image.Image,
    label: str,
    left: int,
    top: int,
    panel_size: int,
    title_h: int,
    label_font,
):
    panel = fit_image(img, panel_size)
    canvas.paste(panel, (left, top + title_h))
    draw.rectangle(
        [left, top + title_h, left + panel_size - 1, top + title_h + panel_size - 1],
        outline=BORDER,
        width=2,
    )
    draw.text((left, top), label, fill=TEXT, font=label_font)


def draw_crop_boxes(
    draw: ImageDraw.ImageDraw,
    image_left: int,
    image_top: int,
    panel_size: int,
    orig_size: Tuple[int, int],
    crops: Sequence[Tuple[int, int, int, int]],
):
    orig_w, orig_h = orig_size
    scale = min(panel_size / orig_w, panel_size / orig_h)
    scaled_w = int(round(orig_w * scale))
    scaled_h = int(round(orig_h * scale))
    offset_x = image_left + (panel_size - scaled_w) // 2
    offset_y = image_top + (panel_size - scaled_h) // 2
    for idx, (x, y, w, h) in enumerate(crops, start=1):
        left = offset_x + int(round(x * scale))
        top = offset_y + int(round(y * scale))
        right = offset_x + int(round((x + w) * scale))
        bottom = offset_y + int(round((y + h) * scale))
        draw.rectangle([left, top, right, bottom], outline=ACCENT, width=3)
        draw.text((left + 4, top + 4), str(idx), fill=ACCENT, font=load_font(20))


def crop_and_fit(img: Image.Image, crop: Tuple[int, int, int, int], size: int) -> Image.Image:
    x, y, w, h = crop
    cropped = img.crop((x, y, x + w, y + h))
    return fit_image(cropped, size)


def make_error_map(gt: Image.Image, pred: Image.Image, scale: float) -> Image.Image:
    if pred.size != gt.size:
        pred = pred.resize(gt.size, Image.Resampling.BICUBIC)
    diff = ImageChops.difference(gt.convert("RGB"), pred.convert("RGB")).convert("L")
    diff = diff.point(lambda p: min(255, int(round(p * scale))))
    return ImageOps.colorize(diff, black=(255, 255, 255), white=(198, 62, 46)).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Generate a stitched demo figure for thesis/PPT reporting.")
    parser.add_argument("--sample-relpath", required=True, help="Sample relative path, e.g. 00001/0001")
    parser.add_argument("--split", default="", help="Dataset split for LR/results roots, e.g. slow_test")
    parser.add_argument("--dataset-root-lr", required=True, help="Root of LR triplets")
    parser.add_argument("--dataset-root-hr", required=True, help="Root of HR triplets")
    parser.add_argument("--method", action="append", type=parse_method_spec, required=True,
                        help="Method result root in LABEL=DIR form. Repeat for multiple methods.")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--title", default="", help="Optional figure title")
    parser.add_argument("--target-frame", type=parse_target_frame, default=None,
                        help="Even target frame for full-chain outputs. Use 2, 4, or 6. "
                             "When set, method outputs default to output_im{N}.png.")
    parser.add_argument("--method-output-filename", default="",
                        help="Override method output filename. Defaults to output.png, or output_im{N}.png with --target-frame.")
    parser.add_argument("--show-error-maps", action="store_true",
                        help="Add an absolute error-map row for each method against GT.")
    parser.add_argument("--error-scale", type=float, default=4.0,
                        help="Multiplier applied before visualizing absolute error maps.")
    parser.add_argument("--crop", action="append", type=parse_crop, default=[],
                        help="Crop box x,y,w,h. Repeat for multiple zoom regions.")
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--gap", type=int, default=24)
    args = parser.parse_args()

    label_font = load_font(26)
    title_font = load_font(34)
    small_font = load_font(22)

    target_frame = args.target_frame or 4
    prev_frame = target_frame - 1
    next_frame = target_frame + 1
    method_output_filename = args.method_output_filename
    if not method_output_filename:
        method_output_filename = f"output_im{target_frame}.png" if args.target_frame else "output.png"

    prev_lr = Image.open(
        resolve_dataset_image(args.dataset_root_lr, args.sample_relpath, f"im{prev_frame}.png", args.split)
    ).convert("RGB")
    next_lr = Image.open(
        resolve_dataset_image(args.dataset_root_lr, args.sample_relpath, f"im{next_frame}.png", args.split)
    ).convert("RGB")
    gt = Image.open(resolve_dataset_image(args.dataset_root_hr, args.sample_relpath, f"im{target_frame}.png")).convert("RGB")

    method_images = []
    for method in args.method:
        out_path = resolve_method_image(method.root, args.sample_relpath, method_output_filename, args.split)
        method_images.append((method.label, Image.open(out_path).convert("RGB")))

    panel_size = args.panel_size
    gap = args.gap
    title_h = 44
    row_h = title_h + panel_size
    header_h = 56 if args.title else 0
    crop_header_h = 30 if args.crop else 0

    top_row_labels = [f"Prev LR im{prev_frame}", f"Next LR im{next_frame}", f"GT im{target_frame}"]
    top_row_images = [prev_lr, next_lr, gt]
    compare_labels = ["GT"] + [label for label, _ in method_images]
    compare_images = [gt] + [img for _, img in method_images]
    error_labels = [f"Error: {label}" for label, _ in method_images]
    error_images = [make_error_map(gt, img, args.error_scale) for _, img in method_images]

    top_cols = len(top_row_images)
    compare_cols = len(compare_images)
    crop_cols = compare_cols
    error_cols = len(error_images) if args.show_error_maps else 0

    max_cols = max(top_cols, compare_cols, crop_cols, error_cols)
    width = max_cols * panel_size + (max_cols + 1) * gap
    height = header_h + gap + row_h + gap + row_h
    if args.show_error_maps:
        height += gap + row_h
    if args.crop:
        height += gap + crop_header_h + len(args.crop) * (row_h + gap)

    canvas = Image.new("RGB", (width, height), CANVAS_BG)
    draw = ImageDraw.Draw(canvas)

    if args.title:
        draw.text((gap, gap // 2), args.title, fill=TEXT, font=title_font)

    y = header_h + gap
    for idx, (label, img) in enumerate(zip(top_row_labels, top_row_images)):
        left = gap + idx * (panel_size + gap)
        draw_labeled_panel(draw, canvas, img, label, left, y, panel_size, title_h, label_font)

    y += row_h + gap
    for idx, (label, img) in enumerate(zip(compare_labels, compare_images)):
        left = gap + idx * (panel_size + gap)
        draw_labeled_panel(draw, canvas, img, label, left, y, panel_size, title_h, label_font)
        if args.crop:
            draw_crop_boxes(draw, left, y + title_h, panel_size, img.size, args.crop)

    if args.show_error_maps:
        y += row_h + gap
        for idx, (label, img) in enumerate(zip(error_labels, error_images)):
            left = gap + idx * (panel_size + gap)
            draw_labeled_panel(draw, canvas, img, label, left, y, panel_size, title_h, small_font)

    if args.crop:
        y += row_h + gap
        draw.text((gap, y), "Zoomed Regions", fill=TEXT, font=title_font)
        y += crop_header_h
        for crop_idx, crop in enumerate(args.crop, start=1):
            for idx, img in enumerate(compare_images):
                left = gap + idx * (panel_size + gap)
                label = compare_labels[idx] if crop_idx == 1 else ""
                if label:
                    draw.text((left, y), label, fill=TEXT, font=small_font)
                crop_panel = crop_and_fit(img, crop, panel_size)
                canvas.paste(crop_panel, (left, y + title_h))
                draw.rectangle(
                    [left, y + title_h, left + panel_size - 1, y + title_h + panel_size - 1],
                    outline=BORDER,
                    width=2,
                )
            draw.text((width - gap - 90, y + title_h + 8), f"Crop {crop_idx}", fill=ACCENT, font=small_font)
            y += row_h + gap

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    canvas.save(args.output)
    print(f"Saved demo figure to {args.output}")


if __name__ == "__main__":
    main()
