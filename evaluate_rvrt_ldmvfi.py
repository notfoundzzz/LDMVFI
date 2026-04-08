import argparse
import os
from os.path import join

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from omegaconf import OmegaConf

import utility
from ldm.models.rvrt_ldmvfi_pipeline import RVRTLDMVFIPipeline


def load_triplet(folder, transform):
    candidates = [
        ("frame_00.png", "frame_01_gt.png", "frame_02.png"),
        ("frame1.png", "framet.png", "frame2.png"),
    ]
    for prev_name, gt_name, next_name in candidates:
        prev_path = join(folder, prev_name)
        gt_path = join(folder, gt_name)
        next_path = join(folder, next_name)
        if all(os.path.exists(p) for p in [prev_path, gt_path, next_path]):
            prev = transform(Image.open(prev_path)).unsqueeze(0)
            gt = transform(Image.open(gt_path)).unsqueeze(0)
            nxt = transform(Image.open(next_path)).unsqueeze(0)
            return prev, gt, nxt
    raise FileNotFoundError(f"No supported triplet naming found in {folder}")


def downsample(frame, scale):
    return torch.nn.functional.interpolate(frame, scale_factor=1.0 / scale, mode="bicubic", align_corners=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RVRT + LDMVFI pipeline on triplet folders")
    parser.add_argument("--ldm_config", required=True)
    parser.add_argument("--ldm_ckpt", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--sr_mode", choices=["bicubic", "rvrt"], default="bicubic")
    parser.add_argument("--rvrt_root", default=None)
    parser.add_argument("--rvrt_task", default="002_RVRT_videosr_bi_Vimeo_14frames")
    parser.add_argument("--rvrt_ckpt", default=None)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--metrics", nargs="+", default=["PSNR", "SSIM"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ldm_config = OmegaConf.load(args.ldm_config)
    pipeline = RVRTLDMVFIPipeline(
        ldm_config=ldm_config,
        ldm_ckpt=args.ldm_ckpt,
        sr_mode=args.sr_mode,
        scale=args.scale,
        rvrt_root=args.rvrt_root,
        rvrt_task=args.rvrt_task,
        rvrt_ckpt=args.rvrt_ckpt,
    )
    results = {metric: [] for metric in args.metrics}
    subfolders = sorted([f for f in os.listdir(args.dataset_root) if os.path.isdir(join(args.dataset_root, f))])
    for name in subfolders:
        prev_hr, gt, next_hr = load_triplet(join(args.dataset_root, name), transform)
        prev_lr = downsample(prev_hr, args.scale)
        next_lr = downsample(next_hr, args.scale)
        with torch.no_grad():
            out, prev_sr, next_sr = pipeline.interpolate(
                prev_lr,
                next_lr,
                use_ddim=args.use_ddim,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
            )

        item_dir = join(args.out_dir, name)
        os.makedirs(item_dir, exist_ok=True)
        save_image(out, join(item_dir, "output.png"), value_range=(-1, 1), normalize=True)
        save_image(prev_sr, join(item_dir, "prev_sr.png"), value_range=(-1, 1), normalize=True)
        save_image(next_sr, join(item_dir, "next_sr.png"), value_range=(-1, 1), normalize=True)

        metrics_msg = {}
        for metric in args.metrics:
            score = getattr(utility, f"calc_{metric.lower()}")(gt.to(pipeline.device), out, [prev_hr.to(pipeline.device), next_hr.to(pipeline.device)])[0].item()
            results[metric].append(score)
            metrics_msg[metric] = round(score, 3)
        print(name, metrics_msg)

    print("Average", {metric: round(float(np.mean(scores)), 3) for metric, scores in results.items()})


if __name__ == "__main__":
    main()
