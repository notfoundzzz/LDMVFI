import argparse
import json
import os
import random
from os.path import join

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from evaluate_rvrt_ldmvfi import (
    collect_triplet_folders,
    compute_metrics,
    config_accepts_rvrt_frontend_params,
    infer_lora_rank_from_checkpoint,
    restore_flow_guidance_metadata,
)
from ldm.models.even_residual_corrector import EvenFrameResidualCorrector
from ldm.models.flow_guidance import build_flow_aligned_input_pair
from ldm.models.rvrt_ldmvfi_pipeline import RVRTLDMVFIPipeline


def charbonnier_loss(pred, target, eps=1e-6):
    return torch.sqrt((pred - target) ** 2 + eps).mean()


def load_sequence(folder, frame_ids, transform):
    frames = [transform(Image.open(join(folder, f"im{frame_id}.png"))) for frame_id in frame_ids]
    return torch.stack(frames, dim=0)


class VimeoOddEvenDataset(Dataset):
    def __init__(self, dataset_root_hr, dataset_root_lr, split, list_file, max_samples=0):
        self.dataset_root_hr = dataset_root_hr
        self.dataset_root_lr = join(dataset_root_lr, split) if split else dataset_root_lr
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.folders = collect_triplet_folders(self.dataset_root_lr, list_file=list_file, max_samples=max_samples)
        if not self.folders:
            raise FileNotFoundError(f"No Vimeo samples found for split={split}, list_file={list_file}")

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        key, lr_folder = self.folders[index]
        hr_folder = join(self.dataset_root_hr, *key.split("/"))
        return {
            "key": key,
            "lr_odd": load_sequence(lr_folder, [1, 3, 5, 7], self.transform),
            "gt_even": load_sequence(hr_folder, [2, 4, 6], self.transform),
        }


def build_pipeline(args):
    ldm_config = OmegaConf.load(args.ldm_config)
    if config_accepts_rvrt_frontend_params(ldm_config):
        restore_flow_guidance_metadata(ldm_config, args.ldm_ckpt)
        params = ldm_config.model.params
        params.sr_frontend_mode = "rvrt"
        params.rvrt_root = args.rvrt_root
        params.rvrt_task = args.rvrt_task
        params.rvrt_ckpt = args.rvrt_ckpt
        params.rvrt_flow_mode = args.rvrt_flow_mode
        params.rvrt_raft_variant = args.rvrt_raft_variant
        params.rvrt_raft_ckpt = args.rvrt_raft_ckpt
        params.rvrt_use_flow_adapter = bool(args.rvrt_use_flow_adapter)
    inferred_lora_rank = infer_lora_rank_from_checkpoint(args.ldm_ckpt)
    if inferred_lora_rank is not None:
        ldm_config.model.params.lora_rank = inferred_lora_rank
    return RVRTLDMVFIPipeline(
        ldm_config=ldm_config,
        ldm_ckpt=args.ldm_ckpt,
        sr_mode="rvrt",
        scale=4,
        rvrt_root=args.rvrt_root,
        rvrt_task=args.rvrt_task,
        rvrt_ckpt=args.rvrt_ckpt,
        rvrt_flow_mode=args.rvrt_flow_mode,
        rvrt_raft_variant=args.rvrt_raft_variant,
        rvrt_raft_ckpt=args.rvrt_raft_ckpt,
        rvrt_use_flow_adapter=bool(args.rvrt_use_flow_adapter),
        use_ema=not args.use_raw_weights,
        strict_checkpoint=not args.allow_incomplete_ckpt,
    )


@torch.no_grad()
def make_even_batch(pipeline, lr_odd, seed, use_ddim, ddim_steps, ddim_eta):
    sr_odd = pipeline.super_resolve_sequence(lr_odd)
    even_preds = []
    for pair_idx in range(3):
        pred = pipeline.interpolate_condition_frames(
            sr_odd[:, pair_idx],
            sr_odd[:, pair_idx + 1],
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            seed=seed + pair_idx,
        )
        even_preds.append(pred)
    prev = torch.stack([sr_odd[:, 0], sr_odd[:, 1], sr_odd[:, 2]], dim=1)
    pred = torch.stack(even_preds, dim=1)
    nxt = torch.stack([sr_odd[:, 1], sr_odd[:, 2], sr_odd[:, 3]], dim=1)
    return sr_odd, prev, pred, nxt


def flatten_even(prev, pred, nxt, gt):
    b, t, c, h, w = pred.shape
    return (
        prev.reshape(b * t, c, h, w),
        pred.reshape(b * t, c, h, w),
        nxt.reshape(b * t, c, h, w),
        gt.reshape(b * t, c, h, w),
    )


@torch.no_grad()
def make_flow_inputs(prev_f, nxt_f, args):
    if not args.use_flow_inputs:
        return None, None
    return build_flow_aligned_input_pair(
        prev_f,
        nxt_f,
        prev_f,
        nxt_f,
        backend=args.flow_backend,
        raft_variant=args.flow_raft_variant,
        raft_ckpt=args.flow_raft_ckpt,
    )


@torch.no_grad()
def validate(pipeline, corrector, loader, args, device):
    corrector.eval()
    losses = []
    scores = []
    for batch_idx, batch in enumerate(loader):
        lr_odd = batch["lr_odd"].to(device)
        gt_even = batch["gt_even"].to(device)
        _, prev, pred, nxt = make_even_batch(
            pipeline,
            lr_odd,
            seed=args.seed + batch_idx * 3,
            use_ddim=args.use_ddim,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
        )
        prev_f, pred_f, nxt_f, gt_f = flatten_even(prev, pred, nxt, gt_even)
        warped_prev, warped_next = make_flow_inputs(prev_f, nxt_f, args)
        refined = corrector(prev_f, pred_f, nxt_f, warped_prev, warped_next)
        losses.append(charbonnier_loss(refined, gt_f).item())
        for i in range(refined.shape[0]):
            scores.append(compute_metrics(args.metrics, gt_f[i : i + 1], refined[i : i + 1], device=device))
    averages = {
        metric: round(float(np.mean([score[metric] for score in scores])), 4)
        for metric in args.metrics
    }
    return float(np.mean(losses)), averages


def save_checkpoint(corrector, optimizer, args, step, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "state_dict": corrector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": {
            "hidden_channels": args.hidden_channels,
            "num_blocks": args.num_blocks,
            "max_residue": args.max_residue,
            "use_flow_inputs": bool(args.use_flow_inputs),
        },
    }
    torch.save(payload, join(out_dir, "last_even_corrector.pth"))
    torch.save(payload, join(out_dir, f"step_{step:06d}_even_corrector.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldm_config", default="/data/Shenzhen/zhahongli/LDMVFI/configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml")
    parser.add_argument("--ldm_ckpt", default="/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt")
    parser.add_argument("--dataset_root_hr", default="/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sequences")
    parser.add_argument("--dataset_root_lr", default="/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sequences_LR")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--train_list", default="/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sep_trainlist.txt")
    parser.add_argument("--val_splits", default="slow_test,medium_test,fast_test")
    parser.add_argument("--val_lists", default="")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=10)
    parser.add_argument("--rvrt_root", default="/data/Shenzhen/zhahongli/RVRT_flow_ablate")
    parser.add_argument("--rvrt_task", default="002_RVRT_videosr_bi_Vimeo_14frames")
    parser.add_argument("--rvrt_ckpt", default="/data/Shenzhen/zhahongli/RVRT/model_zoo/rvrt/002_RVRT_videosr_bi_Vimeo_14frames.pth")
    parser.add_argument("--rvrt_flow_mode", choices=["spynet", "zero", "raft", "spynet_raft_residual"], default="spynet")
    parser.add_argument("--rvrt_raft_variant", choices=["small", "large"], default="large")
    parser.add_argument("--rvrt_raft_ckpt", default=None)
    parser.add_argument("--rvrt_use_flow_adapter", type=int, default=0)
    parser.add_argument("--out_dir", default="/data/Shenzhen/zhahongli/LDMVFI/experiments/even_residual_corrector_spynet")
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--max_residue", type=float, default=0.25)
    parser.add_argument("--use_flow_inputs", type=int, default=0)
    parser.add_argument("--flow_backend", choices=["farneback", "raft"], default="farneback")
    parser.add_argument("--flow_raft_variant", choices=["small", "large"], default="large")
    parser.add_argument("--flow_raft_ckpt", default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--use_ddim", type=int, default=1)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--metrics", nargs="+", default=["PSNR", "SSIM", "LPIPS"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_raw_weights", action="store_true")
    parser.add_argument("--allow_incomplete_ckpt", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(join(args.out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if not args.val_lists:
        val_lists = [
            join(os.path.dirname(args.train_list), f"sep_{split.strip()}list.txt")
            for split in args.val_splits.split(",")
            if split.strip()
        ]
        args.val_lists = ",".join(val_lists)

    print("building frozen RVRT+LDMVFI pipeline...", flush=True)
    pipeline = build_pipeline(args)
    corrector = EvenFrameResidualCorrector(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        max_residue=args.max_residue,
        use_flow_inputs=bool(args.use_flow_inputs),
    ).to(device)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_set = VimeoOddEvenDataset(
        args.dataset_root_hr,
        args.dataset_root_lr,
        args.train_split,
        args.train_list,
        max_samples=args.max_train_samples,
    )
    val_sets = [
        VimeoOddEvenDataset(args.dataset_root_hr, args.dataset_root_lr, split.strip(), val_list.strip(), args.max_val_samples)
        for split, val_list in zip(args.val_splits.split(","), args.val_lists.split(","))
        if split.strip() and val_list.strip()
    ]
    val_items = []
    for val_set in val_sets:
        val_items.extend(val_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_items, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print(f"datasets ready: train={len(train_set)} val={len(val_items)}", flush=True)

    step = 0
    while step < args.max_steps:
        for batch in train_loader:
            corrector.train()
            lr_odd = batch["lr_odd"].to(device)
            gt_even = batch["gt_even"].to(device)
            _, prev, pred, nxt = make_even_batch(
                pipeline,
                lr_odd,
                seed=args.seed + step * 3,
                use_ddim=bool(args.use_ddim),
                ddim_steps=args.ddim_steps,
                ddim_eta=args.ddim_eta,
            )
            prev_f, pred_f, nxt_f, gt_f = flatten_even(prev, pred, nxt, gt_even)
            warped_prev, warped_next = make_flow_inputs(prev_f, nxt_f, args)
            refined = corrector(prev_f, pred_f, nxt_f, warped_prev, warped_next)
            loss = charbonnier_loss(refined, gt_f)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1
            if step == 1 or step % 25 == 0:
                print(f"step={step} loss={loss.item():.6f}", flush=True)
            if step % args.val_interval == 0 or step == args.max_steps:
                val_loss, val_scores = validate(pipeline, corrector, val_loader, args, device)
                print(f"validation step={step} loss={val_loss:.6f} even3={val_scores}", flush=True)
            if step % args.save_interval == 0 or step == args.max_steps:
                save_checkpoint(corrector, optimizer, args, step, args.out_dir)
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
