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
from ldm.models.even_residual_corrector import load_even_corrector
from ldm.models.flow_guidance import build_flow_aligned_input_pair
from ldm.models.rvrt_ldmvfi_pipeline import RVRTLDMVFIPipeline


def config_accepts_rvrt_frontend_params(ldm_config):
    target = str(ldm_config.model.get("target", ""))
    return "LatentDiffusionVFIRVRTFlowGuided" in target or "rvrt_flow_guided_ddpm" in target


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
    if "rvrt_adapter_ckpt" in metadata:
        params.rvrt_adapter_ckpt = metadata["rvrt_adapter_ckpt"]
    if "rvrt_flow_adapter_hidden_channels" in metadata:
        params.rvrt_flow_adapter_hidden_channels = int(metadata["rvrt_flow_adapter_hidden_channels"])
    if "rvrt_flow_adapter_zero_init_last" in metadata:
        params.rvrt_flow_adapter_zero_init_last = bool(metadata["rvrt_flow_adapter_zero_init_last"])
    if "rvrt_flow_adapter_max_residue_magnitude" in metadata:
        params.rvrt_flow_adapter_max_residue_magnitude = float(metadata["rvrt_flow_adapter_max_residue_magnitude"])
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


def load_sequence(folder, frame_ids, transform):
    names = [f"im{i}.png" for i in frame_ids]
    paths = [join(folder, name) for name in names]
    if not all(os.path.exists(path) for path in paths):
        return None
    frames = [transform(Image.open(path)) for path in paths]
    return torch.stack(frames, dim=0).unsqueeze(0)


def load_lr_sequence(folder, transform):
    return load_sequence(folder, list(range(1, 8)), transform)


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


def compute_metrics(metrics, gt, out, prev=None, nxt=None, device=None):
    if device is None:
        device = out.device
    if prev is None:
        prev = out
    if nxt is None:
        nxt = out
    gt_eval, out_eval, prev_eval, next_eval = align_for_metrics(
        gt.to(device),
        out.to(device),
        prev.to(device),
        nxt.to(device),
    )
    scores = {}
    for metric in metrics:
        score = getattr(utility, f"calc_{metric.lower()}")(gt_eval, out_eval, [prev_eval, next_eval])
        scores[metric] = float(score.mean().item())
    return scores


def init_results(groups, metrics):
    return {group: {metric: [] for metric in metrics} for group in groups}


def extend_results(results, group, frame_scores, metrics):
    for scores in frame_scores:
        for metric in metrics:
            results[group][metric].append(scores[metric])


def average_results(results):
    return {
        group: {
            metric: round(float(np.mean(values)), 3) if values else None
            for metric, values in metric_values.items()
        }
        for group, metric_values in results.items()
    }


def result_counts(results):
    return {
        group: {metric: len(values) for metric, values in metric_values.items()}
        for group, metric_values in results.items()
    }


def mean_scores(frame_scores, metrics):
    return {
        metric: round(float(np.mean([scores[metric] for scores in frame_scores])), 3)
        for metric in metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RVRT + LDMVFI on Vimeo sequences")
    parser.add_argument("--ldm_config", required=True)
    parser.add_argument("--ldm_ckpt", required=True)
    parser.add_argument("--dataset_root_hr", required=True)
    parser.add_argument("--dataset_root_lr", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--list_file", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--sr_mode", choices=["bicubic", "rvrt"], default="bicubic")
    parser.add_argument(
        "--eval_pipeline",
        choices=["triplet", "sr_then_vfi", "vfi_then_sr"],
        default="sr_then_vfi",
    )
    parser.add_argument("--rvrt_root", default=None)
    parser.add_argument("--rvrt_task", default="002_RVRT_videosr_bi_Vimeo_14frames")
    parser.add_argument("--rvrt_ckpt", default=None)
    parser.add_argument(
        "--rvrt_flow_mode",
        choices=["spynet", "zero", "raft", "spynet_raft_residual"],
        default="spynet",
    )
    parser.add_argument("--rvrt_raft_variant", choices=["small", "large"], default="large")
    parser.add_argument("--rvrt_raft_ckpt", default=None)
    parser.add_argument("--rvrt_use_flow_adapter", type=int, choices=[0, 1], default=None)
    parser.add_argument("--rvrt_adapter_ckpt", default=None)
    parser.add_argument("--rvrt_flow_adapter_hidden_channels", type=int, default=None)
    parser.add_argument("--rvrt_flow_adapter_zero_init_last", type=int, choices=[0, 1], default=None)
    parser.add_argument("--rvrt_flow_adapter_max_residue_magnitude", type=float, default=None)
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
    parser.add_argument("--even_corrector_ckpt", default=None)
    parser.add_argument("--even_corrector_hidden_channels", type=int, default=32)
    parser.add_argument("--even_corrector_num_blocks", type=int, default=4)
    parser.add_argument("--even_corrector_max_residue", type=float, default=0.25)
    parser.add_argument("--even_corrector_mode", choices=["residual", "fusion"], default="residual")
    parser.add_argument("--even_corrector_fusion_init_pred_logit", type=float, default=8.0)
    parser.add_argument("--even_corrector_use_flow_inputs", type=int, choices=[0, 1], default=0)
    parser.add_argument("--even_corrector_flow_backend", choices=["farneback", "raft"], default="farneback")
    parser.add_argument("--even_corrector_flow_raft_variant", choices=["small", "large"], default="large")
    parser.add_argument("--even_corrector_flow_raft_ckpt", default=None)
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
    accepts_rvrt_frontend_params = config_accepts_rvrt_frontend_params(ldm_config)
    if accepts_rvrt_frontend_params:
        restore_flow_guidance_metadata(ldm_config, args.ldm_ckpt)
        ldm_config.model.params.sr_frontend_mode = args.sr_mode
        ldm_config.model.params.rvrt_root = args.rvrt_root
        ldm_config.model.params.rvrt_task = args.rvrt_task
        ldm_config.model.params.rvrt_ckpt = args.rvrt_ckpt
        ldm_config.model.params.rvrt_flow_mode = args.rvrt_flow_mode
        ldm_config.model.params.rvrt_raft_variant = args.rvrt_raft_variant
        ldm_config.model.params.rvrt_raft_ckpt = args.rvrt_raft_ckpt
        ldm_config.model.params.rvrt_adapter_ckpt = args.rvrt_adapter_ckpt
        if args.rvrt_use_flow_adapter is not None:
            ldm_config.model.params.rvrt_use_flow_adapter = bool(args.rvrt_use_flow_adapter)
        if args.rvrt_flow_adapter_hidden_channels is not None:
            ldm_config.model.params.rvrt_flow_adapter_hidden_channels = args.rvrt_flow_adapter_hidden_channels
        if args.rvrt_flow_adapter_zero_init_last is not None:
            ldm_config.model.params.rvrt_flow_adapter_zero_init_last = bool(args.rvrt_flow_adapter_zero_init_last)
        if args.rvrt_flow_adapter_max_residue_magnitude is not None:
            ldm_config.model.params.rvrt_flow_adapter_max_residue_magnitude = args.rvrt_flow_adapter_max_residue_magnitude
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
        rvrt_adapter_ckpt=args.rvrt_adapter_ckpt,
        rvrt_use_flow_adapter=bool(args.rvrt_use_flow_adapter) if args.rvrt_use_flow_adapter is not None else False,
        rvrt_flow_adapter_hidden_channels=args.rvrt_flow_adapter_hidden_channels or 16,
        rvrt_flow_adapter_zero_init_last=bool(args.rvrt_flow_adapter_zero_init_last)
        if args.rvrt_flow_adapter_zero_init_last is not None
        else True,
        rvrt_flow_adapter_max_residue_magnitude=args.rvrt_flow_adapter_max_residue_magnitude
        if args.rvrt_flow_adapter_max_residue_magnitude is not None
        else 1.0,
        use_ema=args.use_ema,
        strict_checkpoint=args.strict_checkpoint,
    )
    even_corrector = None
    if args.even_corrector_ckpt:
        even_corrector = load_even_corrector(
            args.even_corrector_ckpt,
            pipeline.device,
            hidden_channels=args.even_corrector_hidden_channels,
            num_blocks=args.even_corrector_num_blocks,
            max_residue=args.even_corrector_max_residue,
            corrector_mode=args.even_corrector_mode,
            fusion_init_pred_logit=args.even_corrector_fusion_init_pred_logit,
        )
        print(f"Loaded even-frame corrector from {args.even_corrector_ckpt}")
    result_groups = ["target"] if args.eval_pipeline == "triplet" else ["all7", "odd4", "even3"]
    results = init_results(result_groups, args.metrics)
    dataset_root_hr = args.dataset_root_hr
    dataset_root_lr = args.dataset_root_lr
    if args.split:
        dataset_root_lr = join(dataset_root_lr, args.split)

    triplet_folders = collect_triplet_folders(dataset_root_lr, list_file=args.list_file, max_samples=args.max_samples)
    if not triplet_folders:
        raise FileNotFoundError(f"No supported triplet folders found under {dataset_root_lr}")
    print(f"Found {len(triplet_folders)} sequences")

    for idx, (name, folder) in enumerate(triplet_folders, start=1):
        print(f"[{idx}/{len(triplet_folders)}] {name}")
        hr_folder = join(dataset_root_hr, *name.split("/"))

        should_save = args.save_images and (args.save_max_samples <= 0 or idx <= args.save_max_samples)
        item_dir = join(args.out_dir, name)
        if should_save:
            os.makedirs(item_dir, exist_ok=True)

        if args.eval_pipeline == "triplet":
            prev_lr, _, next_lr = load_triplet(folder, transform)
            lr_sequence = load_lr_sequence(folder, transform)
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
            frame_scores = [
                compute_metrics(args.metrics, gt, out, prev_sr, next_sr, device=pipeline.device)
            ]
            extend_results(results, "target", frame_scores, args.metrics)
            print(name, {"target": mean_scores(frame_scores, args.metrics)})
            if should_save:
                save_image(out, join(item_dir, "output.png"), value_range=(-1, 1), normalize=True)
                if args.save_sr_images:
                    save_image(prev_sr, join(item_dir, "prev_sr.png"), value_range=(-1, 1), normalize=True)
                    save_image(next_sr, join(item_dir, "next_sr.png"), value_range=(-1, 1), normalize=True)
        else:
            lr_odd = load_sequence(folder, [1, 3, 5, 7], transform)
            gt_full = load_sequence(hr_folder, list(range(1, 8)), transform)
            if lr_odd is None or gt_full is None:
                raise FileNotFoundError(f"Missing Vimeo frames in {folder} or {hr_folder}")

            with torch.no_grad():
                if args.eval_pipeline == "sr_then_vfi":
                    sr_odd = pipeline.super_resolve_sequence(lr_odd)
                    even_preds = []
                    for pair_idx in range(3):
                        pred = pipeline.interpolate_condition_frames(
                            sr_odd[:, pair_idx],
                            sr_odd[:, pair_idx + 1],
                            use_ddim=args.use_ddim,
                            ddim_steps=args.ddim_steps,
                            ddim_eta=args.ddim_eta,
                            seed=args.seed + pair_idx,
                        )
                        if even_corrector is not None:
                            warped_prev = None
                            warped_next = None
                            if args.even_corrector_use_flow_inputs:
                                warped_prev, warped_next = build_flow_aligned_input_pair(
                                    sr_odd[:, pair_idx],
                                    sr_odd[:, pair_idx + 1],
                                    sr_odd[:, pair_idx],
                                    sr_odd[:, pair_idx + 1],
                                    backend=args.even_corrector_flow_backend,
                                    raft_variant=args.even_corrector_flow_raft_variant,
                                    raft_ckpt=args.even_corrector_flow_raft_ckpt,
                                )
                            pred = even_corrector(
                                sr_odd[:, pair_idx],
                                pred,
                                sr_odd[:, pair_idx + 1],
                                warped_prev,
                                warped_next,
                            )
                        even_preds.append(pred)
                    full_preds = [
                        sr_odd[:, 0],
                        even_preds[0],
                        sr_odd[:, 1],
                        even_preds[1],
                        sr_odd[:, 2],
                        even_preds[2],
                        sr_odd[:, 3],
                    ]
                else:
                    pred_lr = []
                    for pair_idx in range(3):
                        pred = pipeline.interpolate_condition_frames(
                            lr_odd[:, pair_idx].to(pipeline.device),
                            lr_odd[:, pair_idx + 1].to(pipeline.device),
                            use_ddim=args.use_ddim,
                            ddim_steps=args.ddim_steps,
                            ddim_eta=args.ddim_eta,
                            seed=args.seed + pair_idx,
                        )
                        pred_lr.append(pred.cpu())
                    lr_full = torch.stack(
                        [
                            lr_odd[:, 0],
                            pred_lr[0],
                            lr_odd[:, 1],
                            pred_lr[1],
                            lr_odd[:, 2],
                            pred_lr[2],
                            lr_odd[:, 3],
                        ],
                        dim=1,
                    )
                    sr_full = pipeline.super_resolve_sequence(lr_full)
                    full_preds = [sr_full[:, frame_idx] for frame_idx in range(7)]

            all_scores = []
            for frame_idx, frame_id in enumerate(range(1, 8)):
                prev_idx = max(frame_idx - 1, 0)
                next_idx = min(frame_idx + 1, 6)
                scores = compute_metrics(
                    args.metrics,
                    gt_full[:, frame_idx],
                    full_preds[frame_idx],
                    full_preds[prev_idx],
                    full_preds[next_idx],
                    device=pipeline.device,
                )
                all_scores.append(scores)
                if should_save:
                    save_image(
                        full_preds[frame_idx],
                        join(item_dir, f"output_im{frame_id}.png"),
                        value_range=(-1, 1),
                        normalize=True,
                    )
                if should_save and args.save_sr_images and frame_id % 2 == 1:
                    save_image(
                        full_preds[frame_idx],
                        join(item_dir, f"sr_im{frame_id}.png"),
                        value_range=(-1, 1),
                        normalize=True,
                    )

            odd_scores = [all_scores[i] for i in (0, 2, 4, 6)]
            even_scores = [all_scores[i] for i in (1, 3, 5)]
            extend_results(results, "all7", all_scores, args.metrics)
            extend_results(results, "odd4", odd_scores, args.metrics)
            extend_results(results, "even3", even_scores, args.metrics)
            print(
                name,
                {
                    "all7": mean_scores(all_scores, args.metrics),
                    "odd4": mean_scores(odd_scores, args.metrics),
                    "even3": mean_scores(even_scores, args.metrics),
                },
            )

    averages = average_results(results)
    average_key = "target" if args.eval_pipeline == "triplet" else "all7"
    average = averages[average_key]
    for group, group_average in averages.items():
        print(f"Average {group}", group_average)
    if args.summary_json:
        summary = {
            "split": args.split,
            "num_samples": len(triplet_folders),
            "eval_pipeline": args.eval_pipeline,
            "average": average,
            "averages": averages,
            "metric_frame_counts": result_counts(results),
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
            "rvrt_adapter_ckpt": args.rvrt_adapter_ckpt,
            "rvrt_use_flow_adapter": bool(args.rvrt_use_flow_adapter) if args.rvrt_use_flow_adapter is not None else False,
            "rvrt_flow_adapter_hidden_channels": args.rvrt_flow_adapter_hidden_channels,
            "rvrt_flow_adapter_zero_init_last": bool(args.rvrt_flow_adapter_zero_init_last)
            if args.rvrt_flow_adapter_zero_init_last is not None
            else None,
            "rvrt_flow_adapter_max_residue_magnitude": args.rvrt_flow_adapter_max_residue_magnitude,
            "even_corrector_ckpt": args.even_corrector_ckpt,
            "even_corrector_mode": args.even_corrector_mode,
            "even_corrector_use_flow_inputs": bool(args.even_corrector_use_flow_inputs),
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
