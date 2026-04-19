import argparse
import json
import os
from os.path import join

import torch
from torchvision import transforms
from torchvision.utils import save_image
from omegaconf import OmegaConf

from evaluate_rvrt_pisa_dual_lora import (
    collect_triplet_folders,
    infer_dual_lora_ranks,
    load_pisa_dual_lora_metadata,
    load_triplet,
)
from ldm.models.rvrt_pisa_dual_lora_pipeline import RVRTPiSADualLoRAPipeline


def summarize_branch_weights(ldm_ckpt):
    state = torch.load(ldm_ckpt, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    summary = {}
    for branch in ("pixel", "semantic"):
        keys = [k for k in state_dict if f"{branch}_lora_" in k]
        tensors = [state_dict[k].float() for k in keys]
        total_abs = float(sum(t.abs().sum().item() for t in tensors)) if tensors else 0.0
        max_abs = float(max((t.abs().max().item() for t in tensors), default=0.0))
        nonzero_tensors = int(sum(1 for t in tensors if t.abs().sum().item() > 0))
        summary[branch] = {
            "num_tensors": len(tensors),
            "nonzero_tensors": nonzero_tensors,
            "total_abs": total_abs,
            "max_abs": max_abs,
        }
    return summary


def prepare_config(ldm_config_path, ldm_ckpt, pixel_groups=None, semantic_groups=None, target_suffixes=None):
    ldm_config = OmegaConf.load(ldm_config_path)
    pixel_rank, semantic_rank = infer_dual_lora_ranks(ldm_ckpt)
    metadata = load_pisa_dual_lora_metadata(ldm_ckpt)
    if pixel_rank is not None:
        ldm_config.model.params.pixel_lora_rank = pixel_rank
    if semantic_rank is not None:
        ldm_config.model.params.semantic_lora_rank = semantic_rank
    if "pixel_lora_groups" in metadata:
        ldm_config.model.params.pixel_lora_groups = list(metadata["pixel_lora_groups"])
    if "semantic_lora_groups" in metadata:
        ldm_config.model.params.semantic_lora_groups = list(metadata["semantic_lora_groups"])
    if "lora_target_suffixes" in metadata:
        ldm_config.model.params.lora_target_suffixes = list(metadata["lora_target_suffixes"])
    if pixel_groups:
        ldm_config.model.params.pixel_lora_groups = list(pixel_groups)
    if semantic_groups:
        ldm_config.model.params.semantic_lora_groups = list(semantic_groups)
    if target_suffixes:
        ldm_config.model.params.lora_target_suffixes = list(target_suffixes)
    return ldm_config


def compare_outputs(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_root_lr = args.dataset_root_lr
    if args.split:
        dataset_root_lr = join(dataset_root_lr, args.split)
    triplet_folders = collect_triplet_folders(dataset_root_lr, list_file=args.list_file, max_samples=args.max_samples)
    if not triplet_folders:
        raise FileNotFoundError(f"No supported triplet folders found under {dataset_root_lr}")

    if args.sample_name:
        matched = [item for item in triplet_folders if item[0] == args.sample_name]
        if not matched:
            raise FileNotFoundError(f"Sample {args.sample_name} was not found")
        name, folder = matched[0]
    else:
        sample_index = min(max(args.sample_index, 0), len(triplet_folders) - 1)
        name, folder = triplet_folders[sample_index]

    prev_lr, _, next_lr = load_triplet(folder, transform)
    ldm_config = prepare_config(
        args.ldm_config,
        args.ldm_ckpt,
        pixel_groups=args.pixel_lora_groups,
        semantic_groups=args.semantic_lora_groups,
        target_suffixes=args.lora_target_suffixes,
    )

    pipeline_a = RVRTPiSADualLoRAPipeline(
        ldm_config=ldm_config,
        ldm_ckpt=args.ldm_ckpt,
        sr_mode=args.sr_mode,
        scale=args.scale,
        rvrt_root=args.rvrt_root,
        rvrt_task=args.rvrt_task,
        rvrt_ckpt=args.rvrt_ckpt,
        pixel_scale=args.pixel_scale_a,
        semantic_scale=args.semantic_scale_a,
        use_ema=not args.use_raw_weights,
    )
    out_a, prev_sr, next_sr = pipeline_a.interpolate(
        prev_lr,
        next_lr,
        use_ddim=args.use_ddim,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        seed=args.seed,
    )

    pipeline_b = RVRTPiSADualLoRAPipeline(
        ldm_config=ldm_config,
        ldm_ckpt=args.ldm_ckpt,
        sr_mode=args.sr_mode,
        scale=args.scale,
        rvrt_root=args.rvrt_root,
        rvrt_task=args.rvrt_task,
        rvrt_ckpt=args.rvrt_ckpt,
        pixel_scale=args.pixel_scale_b,
        semantic_scale=args.semantic_scale_b,
        use_ema=not args.use_raw_weights,
    )
    out_b, _, _ = pipeline_b.interpolate(
        prev_lr,
        next_lr,
        use_ddim=args.use_ddim,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        seed=args.seed,
    )

    diff = (out_a - out_b).abs()
    result = {
        "sample_name": name,
        "scale_a": {"pixel": args.pixel_scale_a, "semantic": args.semantic_scale_a},
        "scale_b": {"pixel": args.pixel_scale_b, "semantic": args.semantic_scale_b},
        "use_ddim": args.use_ddim,
        "ddim_eta": args.ddim_eta,
        "seed": args.seed,
        "mean_abs_diff": float(diff.mean().item()),
        "max_abs_diff": float(diff.max().item()),
        "sum_abs_diff": float(diff.sum().item()),
    }

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_image(out_a, join(args.save_dir, "output_a.png"), value_range=(-1, 1), normalize=True)
        save_image(out_b, join(args.save_dir, "output_b.png"), value_range=(-1, 1), normalize=True)
        save_image(diff / diff.max().clamp_min(1.0e-8), join(args.save_dir, "output_diff.png"))
        save_image(prev_sr, join(args.save_dir, "prev_sr.png"), value_range=(-1, 1), normalize=True)
        save_image(next_sr, join(args.save_dir, "next_sr.png"), value_range=(-1, 1), normalize=True)

    return result


def main():
    parser = argparse.ArgumentParser(description="Inspect PiSA Dual-LoRA checkpoint weights and output differences")
    parser.add_argument("--ldm_ckpt", required=True)
    parser.add_argument("--ldm_config", default=None)
    parser.add_argument("--dataset_root_lr", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--list_file", default=None)
    parser.add_argument("--sample_name", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--sr_mode", choices=["bicubic", "rvrt"], default="rvrt")
    parser.add_argument("--rvrt_root", default=None)
    parser.add_argument("--rvrt_task", default="002_RVRT_videosr_bi_Vimeo_14frames")
    parser.add_argument("--rvrt_ckpt", default=None)
    parser.add_argument("--pixel_lora_groups", nargs="*", default=None)
    parser.add_argument("--semantic_lora_groups", nargs="*", default=None)
    parser.add_argument("--lora_target_suffixes", nargs="*", default=None)
    parser.add_argument("--pixel_scale_a", type=float, default=1.0)
    parser.add_argument("--semantic_scale_a", type=float, default=1.0)
    parser.add_argument("--pixel_scale_b", type=float, default=1.0)
    parser.add_argument("--semantic_scale_b", type=float, default=0.0)
    parser.add_argument("--use_ddim", dest="use_ddim", action="store_true")
    parser.add_argument("--use_ddpm", dest="use_ddim", action="store_false")
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_raw_weights", action="store_true")
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.set_defaults(use_ddim=True)
    args = parser.parse_args()

    summary = {
        "ldm_ckpt": args.ldm_ckpt,
        "weight_summary": summarize_branch_weights(args.ldm_ckpt),
    }
    print("==== weight summary ====")
    print(json.dumps(summary["weight_summary"], indent=2, ensure_ascii=True))

    if args.ldm_config and args.dataset_root_lr:
        comparison = compare_outputs(args)
        summary["output_comparison"] = comparison
        print("==== output comparison ====")
        print(json.dumps(comparison, indent=2, ensure_ascii=True))
    else:
        print("Skip output comparison because ldm_config or dataset_root_lr was not provided.")

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
