import argparse
import json
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


def restore_flow_guidance_metadata(ldm_config, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    metadata = state.get("rvrt_flow_guidance_metadata", None)
    if not metadata:
        return
    params = ldm_config.model.params
    if "sr_frontend_mode" in metadata:
        params.sr_frontend_mode = metadata["sr_frontend_mode"]
    if "rvrt_flow_mode" in metadata:
        params.rvrt_flow_mode = metadata["rvrt_flow_mode"]
    if "rvrt_raft_variant" in metadata:
        params.rvrt_raft_variant = metadata["rvrt_raft_variant"]
    if "rvrt_raft_ckpt" in metadata:
        params.rvrt_raft_ckpt = metadata["rvrt_raft_ckpt"]
    if "rvrt_use_flow_adapter" in metadata:
        params.rvrt_use_flow_adapter = bool(metadata["rvrt_use_flow_adapter"])
    if "rvrt_flow_adapter_hidden_channels" in metadata:
        params.rvrt_flow_adapter_hidden_channels = int(metadata["rvrt_flow_adapter_hidden_channels"])
    if "rvrt_flow_adapter_zero_init_last" in metadata:
        params.rvrt_flow_adapter_zero_init_last = bool(metadata["rvrt_flow_adapter_zero_init_last"])
    if "lr_sequence_key" in metadata:
        params.lr_sequence_key = metadata["lr_sequence_key"]
    if "rvrt_prev_index" in metadata:
        params.rvrt_prev_index = int(metadata["rvrt_prev_index"])
    if "rvrt_next_index" in metadata:
        params.rvrt_next_index = int(metadata["rvrt_next_index"])
    if "use_flow_guidance" in metadata:
        params.use_flow_guidance = bool(metadata["use_flow_guidance"])
    if "flow_guidance_strength" in metadata:
        params.flow_guidance_strength = float(metadata["flow_guidance_strength"])
    if "flow_condition_mode" in metadata:
        params.flow_condition_mode = metadata["flow_condition_mode"]
    if "flow_backend" in metadata:
        params.flow_backend = metadata["flow_backend"]
    if "flow_raft_variant" in metadata:
        params.flow_raft_variant = metadata["flow_raft_variant"]
    if "flow_raft_ckpt" in metadata:
        params.flow_raft_ckpt = metadata["flow_raft_ckpt"]
    if "latent_motion_hidden_channels" in metadata:
        params.latent_motion_hidden_channels = int(metadata["latent_motion_hidden_channels"])
    if "latent_motion_zero_init_last" in metadata:
        params.latent_motion_zero_init_last = bool(metadata["latent_motion_zero_init_last"])
    if "use_condition_adapter" in metadata:
        params.use_condition_adapter = bool(metadata["use_condition_adapter"])
    if "condition_adapter_hidden_channels" in metadata:
        params.condition_adapter_hidden_channels = int(metadata["condition_adapter_hidden_channels"])
    if "condition_adapter_zero_init_last" in metadata:
        params.condition_adapter_zero_init_last = bool(metadata["condition_adapter_zero_init_last"])
    if "condition_adapter_lr_scale" in metadata:
        params.condition_adapter_lr_scale = float(metadata["condition_adapter_lr_scale"])
    if "motion_lr_scale" in metadata:
        params.motion_lr_scale = float(metadata["motion_lr_scale"])
    if "image_recon_loss_weight" in metadata:
        params.image_recon_loss_weight = float(metadata["image_recon_loss_weight"])
    if "image_recon_loss_type" in metadata:
        params.image_recon_loss_type = metadata["image_recon_loss_type"]


def infer_lora_rank_from_checkpoint(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    lora_down_keys = sorted(k for k in state_dict.keys() if k.endswith("lora_down.weight"))
    if not lora_down_keys:
        return None
    first_key = lora_down_keys[0]
    return int(state_dict[first_key].shape[0])


def load_triplet(folder, transform):
    candidates = [
        ("frame_00.png", "frame_01_gt.png", "frame_02.png"),
        ("frame1.png", "framet.png", "frame2.png"),
        ("im3.png", "im4.png", "im5.png"),
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


def load_lr_sequence(folder, transform):
    names = [f"im{i}.png" for i in range(1, 7)]
    paths = [join(folder, name) for name in names]
    if not all(os.path.exists(path) for path in paths):
        return None
    frames = [transform(Image.open(path)) for path in paths]
    return torch.stack(frames, dim=0).unsqueeze(0)


def collect_triplet_folders(dataset_root, list_file=None, max_samples=0):
    if list_file:
        folders = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                folder = join(dataset_root, *line.split("/"))
                folders.append((line, folder))
                if max_samples > 0 and len(folders) >= max_samples:
                    break
        folders.sort(key=lambda x: x[0])
        return folders

    triplet_markers = [
        ("frame_00.png", "frame_01_gt.png", "frame_02.png"),
        ("frame1.png", "framet.png", "frame2.png"),
        ("im3.png", "im4.png", "im5.png"),
    ]
    folders = []
    for root, _, files in os.walk(dataset_root):
        file_set = set(files)
        for marker in triplet_markers:
            if all(name in file_set for name in marker):
                rel_name = os.path.relpath(root, dataset_root)
                folders.append((rel_name, root))
                break
        if max_samples > 0 and len(folders) >= max_samples:
            break
    folders.sort(key=lambda x: x[0])
    return folders


def center_crop_to(tensor, height, width):
    _, _, h, w = tensor.shape
    top = max((h - height) // 2, 0)
    left = max((w - width) // 2, 0)
    return tensor[:, :, top : top + height, left : left + width]


def align_for_metrics(*tensors):
    min_h = min(t.shape[-2] for t in tensors)
    min_w = min(t.shape[-1] for t in tensors)
    return [center_crop_to(t, min_h, min_w) for t in tensors]


def main():
    parser = argparse.ArgumentParser(description="Evaluate RVRT + LDMVFI pipeline on triplet folders")
    parser.add_argument("--ldm_config", required=True)
    parser.add_argument("--ldm_ckpt", required=True)
    parser.add_argument("--dataset_root_hr", required=True)
    parser.add_argument("--dataset_root_lr", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--list_file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--sr_mode", choices=["bicubic", "rvrt"], default="bicubic")
    parser.add_argument("--rvrt_root", default=None)
    parser.add_argument("--rvrt_task", default="002_RVRT_videosr_bi_Vimeo_14frames")
    parser.add_argument("--rvrt_ckpt", default=None)
    parser.add_argument("--rvrt_flow_mode", choices=["spynet", "zero", "raft"], default="spynet")
    parser.add_argument("--rvrt_raft_variant", choices=["small", "large"], default="large")
    parser.add_argument("--rvrt_raft_ckpt", default=None)
    parser.add_argument("--rvrt_use_flow_adapter", type=int, choices=[0, 1], default=None)
    parser.add_argument("--rvrt_flow_adapter_hidden_channels", type=int, default=None)
    parser.add_argument("--rvrt_flow_adapter_zero_init_last", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use_ddim", dest="use_ddim", action="store_true")
    parser.add_argument("--use_ddpm", dest="use_ddim", action="store_false")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--metrics", nargs="+", default=["PSNR", "SSIM"])
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--save_sr_images", action="store_true")
    parser.add_argument("--save_max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_ema", dest="use_ema", action="store_true")
    parser.add_argument("--use_raw_weights", dest="use_ema", action="store_false")
    parser.add_argument("--allow_incomplete_ckpt", dest="strict_checkpoint", action="store_false")
    parser.add_argument("--use_flow_guidance", type=int, choices=[0, 1], default=None)
    parser.add_argument("--flow_backend", default=None)
    parser.add_argument("--flow_condition_mode", default=None)
    parser.add_argument("--flow_raft_variant", default=None)
    parser.add_argument("--flow_raft_ckpt", default=None)
    parser.set_defaults(use_ddim=True, use_ema=True, strict_checkpoint=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ldm_config = OmegaConf.load(args.ldm_config)
    restore_flow_guidance_metadata(ldm_config, args.ldm_ckpt)
    ldm_config.model.params.sr_frontend_mode = args.sr_mode
    ldm_config.model.params.rvrt_root = args.rvrt_root
    ldm_config.model.params.rvrt_task = args.rvrt_task
    ldm_config.model.params.rvrt_ckpt = args.rvrt_ckpt
    ldm_config.model.params.rvrt_flow_mode = args.rvrt_flow_mode
    ldm_config.model.params.rvrt_raft_variant = args.rvrt_raft_variant
    ldm_config.model.params.rvrt_raft_ckpt = args.rvrt_raft_ckpt
    if args.rvrt_use_flow_adapter is not None:
        ldm_config.model.params.rvrt_use_flow_adapter = bool(args.rvrt_use_flow_adapter)
    if args.rvrt_flow_adapter_hidden_channels is not None:
        ldm_config.model.params.rvrt_flow_adapter_hidden_channels = args.rvrt_flow_adapter_hidden_channels
    if args.rvrt_flow_adapter_zero_init_last is not None:
        ldm_config.model.params.rvrt_flow_adapter_zero_init_last = bool(args.rvrt_flow_adapter_zero_init_last)
    if args.use_flow_guidance is not None:
        ldm_config.model.params.use_flow_guidance = bool(args.use_flow_guidance)
    if args.flow_backend:
        ldm_config.model.params.flow_backend = args.flow_backend
    if args.flow_condition_mode:
        ldm_config.model.params.flow_condition_mode = args.flow_condition_mode
    if args.flow_raft_variant:
        ldm_config.model.params.flow_raft_variant = args.flow_raft_variant
    if args.flow_raft_ckpt:
        ldm_config.model.params.flow_raft_ckpt = args.flow_raft_ckpt
    inferred_lora_rank = infer_lora_rank_from_checkpoint(args.ldm_ckpt)
    if inferred_lora_rank is not None:
        cfg_rank = ldm_config.model.params.get("lora_rank", None)
        if cfg_rank != inferred_lora_rank:
            print(
                f"Overriding config lora_rank from {cfg_rank} to {inferred_lora_rank} "
                f"based on checkpoint {args.ldm_ckpt}"
            )
            ldm_config.model.params.lora_rank = inferred_lora_rank
    pipeline = RVRTLDMVFIPipeline(
        ldm_config=ldm_config,
        ldm_ckpt=args.ldm_ckpt,
        sr_mode=args.sr_mode,
        scale=args.scale,
        rvrt_root=args.rvrt_root,
        rvrt_task=args.rvrt_task,
        rvrt_ckpt=args.rvrt_ckpt,
        rvrt_flow_mode=args.rvrt_flow_mode,
        rvrt_raft_variant=args.rvrt_raft_variant,
        rvrt_raft_ckpt=args.rvrt_raft_ckpt,
        rvrt_use_flow_adapter=bool(args.rvrt_use_flow_adapter) if args.rvrt_use_flow_adapter is not None else False,
        rvrt_flow_adapter_hidden_channels=args.rvrt_flow_adapter_hidden_channels or 16,
        rvrt_flow_adapter_zero_init_last=bool(args.rvrt_flow_adapter_zero_init_last)
        if args.rvrt_flow_adapter_zero_init_last is not None
        else True,
        use_ema=args.use_ema,
        strict_checkpoint=args.strict_checkpoint,
    )
    results = {metric: [] for metric in args.metrics}
    dataset_root_hr = args.dataset_root_hr
    dataset_root_lr = args.dataset_root_lr
    if args.split:
        dataset_root_lr = join(dataset_root_lr, args.split)

    triplet_folders = collect_triplet_folders(dataset_root_lr, list_file=args.list_file, max_samples=args.max_samples)
    if not triplet_folders:
        raise FileNotFoundError(f"No supported triplet folders found under {dataset_root_lr}")
    print(f"Found {len(triplet_folders)} triplets")

    for idx, (name, folder) in enumerate(triplet_folders, start=1):
        print(f"[{idx}/{len(triplet_folders)}] {name}")
        prev_lr, _, next_lr = load_triplet(folder, transform)
        lr_sequence = load_lr_sequence(folder, transform)
        hr_folder = join(dataset_root_hr, *name.split("/"))
        _, gt, _ = load_triplet(hr_folder, transform)
        with torch.no_grad():
            out, prev_sr, next_sr = pipeline.interpolate(
                prev_lr,
                next_lr,
                lr_sequence=lr_sequence,
                use_ddim=args.use_ddim,
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
                seed=args.seed,
            )

        should_save = args.save_images and (args.save_max_samples <= 0 or idx <= args.save_max_samples)
        if should_save:
            item_dir = join(args.out_dir, name)
            os.makedirs(item_dir, exist_ok=True)
            save_image(out, join(item_dir, "output.png"), value_range=(-1, 1), normalize=True)
            if args.save_sr_images:
                save_image(prev_sr, join(item_dir, "prev_sr.png"), value_range=(-1, 1), normalize=True)
                save_image(next_sr, join(item_dir, "next_sr.png"), value_range=(-1, 1), normalize=True)

        gt_eval, out_eval, prev_eval, next_eval = align_for_metrics(
            gt.to(pipeline.device),
            out,
            prev_sr,
            next_sr,
        )
        metrics_msg = {}
        for metric in args.metrics:
            score = getattr(utility, f"calc_{metric.lower()}")(gt_eval, out_eval, [prev_eval, next_eval])[0].item()
            results[metric].append(score)
            metrics_msg[metric] = round(score, 3)
        print(name, metrics_msg)

    average = {metric: round(float(np.mean(scores)), 3) for metric, scores in results.items()}
    print("Average", average)
    if args.summary_json:
        summary = {
            "split": args.split,
            "num_samples": len(triplet_folders),
            "average": average,
            "use_ddim": args.use_ddim,
            "ddim_eta": args.ddim_eta,
            "seed": args.seed,
            "use_ema": args.use_ema,
            "strict_checkpoint": args.strict_checkpoint,
            "use_flow_guidance": bool(ldm_config.model.params.get("use_flow_guidance", False)),
            "flow_guidance_strength": float(ldm_config.model.params.get("flow_guidance_strength", 0.0)),
            "rvrt_flow_mode": args.rvrt_flow_mode,
            "rvrt_raft_variant": args.rvrt_raft_variant,
            "rvrt_raft_ckpt": args.rvrt_raft_ckpt,
            "flow_condition_mode": str(ldm_config.model.params.get("flow_condition_mode", "fused")),
            "flow_backend": str(ldm_config.model.params.get("flow_backend", "farneback")),
            "flow_raft_variant": str(ldm_config.model.params.get("flow_raft_variant", "large")),
            "flow_raft_ckpt": ldm_config.model.params.get("flow_raft_ckpt", None),
            "save_images": args.save_images,
            "save_sr_images": args.save_sr_images,
            "save_max_samples": args.save_max_samples,
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
