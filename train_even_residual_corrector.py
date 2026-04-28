import argparse
import json
import os
import random
import time
from os.path import join

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
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


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("local_rank", 0)))
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def is_main_process(rank):
    return rank == 0


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
    raw_corrector = corrector.module if isinstance(corrector, DDP) else corrector
    payload = {
        "state_dict": raw_corrector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": {
            "hidden_channels": args.hidden_channels,
            "num_blocks": args.num_blocks,
            "max_residue": args.max_residue,
            "use_flow_inputs": bool(args.use_flow_inputs),
            "corrector_mode": getattr(args, "corrector_mode", "residual"),
            "fusion_init_pred_logit": getattr(args, "fusion_init_pred_logit", 8.0),
        },
    }
    torch.save(payload, join(out_dir, "last_even_corrector.pth"))
    torch.save(payload, join(out_dir, f"step_{step:06d}_even_corrector.pth"))


def estimate_remaining_events(step, max_steps, interval):
    if interval <= 0:
        return 0
    return sum(1 for next_step in range(step + 1, max_steps + 1) if next_step % interval == 0 or next_step == max_steps)


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
    parser.add_argument("--corrector_mode", choices=["residual", "fusion"], default="residual")
    parser.add_argument("--fusion_init_pred_logit", type=float, default=8.0)
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
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = init_distributed()
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process(rank):
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

    if is_main_process(rank):
        print(f"world_size={world_size}", flush=True)
        print("building frozen RVRT+LDMVFI pipeline...", flush=True)
    pipeline = build_pipeline(args)
    corrector = EvenFrameResidualCorrector(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        max_residue=args.max_residue,
        use_flow_inputs=bool(args.use_flow_inputs),
        corrector_mode=args.corrector_mode,
        fusion_init_pred_logit=args.fusion_init_pred_logit,
    ).to(device)

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
    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_items, batch_size=1, shuffle=False, num_workers=args.num_workers)
    if distributed:
        corrector = DDP(corrector, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if is_main_process(rank):
        print(f"datasets ready: train={len(train_set)} val={len(val_items)}", flush=True)

    step = 0
    epoch = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0
    train_elapsed = 0.0
    last_log_train_elapsed = 0.0
    val_elapsed_total = 0.0
    val_runs = 0
    save_elapsed_total = 0.0
    save_runs = 0
    while step < args.max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            step_start = time.time()
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
            train_elapsed += time.time() - step_start
            if is_main_process(rank) and (step == 1 or step % 25 == 0):
                now = time.time()
                elapsed = now - start_time
                avg_train_step_time = train_elapsed / max(step, 1)
                recent_train_step_time = (train_elapsed - last_log_train_elapsed) / max(step - last_log_step, 1)
                remaining_steps = max(args.max_steps - step, 0)
                train_eta = avg_train_step_time * remaining_steps
                remaining_val_runs = estimate_remaining_events(step, args.max_steps, args.val_interval)
                remaining_save_runs = estimate_remaining_events(step, args.max_steps, args.save_interval)
                avg_val_time = val_elapsed_total / max(val_runs, 1)
                avg_save_time = save_elapsed_total / max(save_runs, 1)
                eta_with_val = train_eta + avg_val_time * remaining_val_runs + avg_save_time * remaining_save_runs
                print(
                    f"step={step}/{args.max_steps} loss={loss.item():.6f} "
                    f"elapsed={elapsed/60:.1f}m train_elapsed={train_elapsed/60:.1f}m "
                    f"avg_train_step={avg_train_step_time:.2f}s recent_train_step={recent_train_step_time:.2f}s "
                    f"train_eta={train_eta/60:.1f}m eta_with_val={eta_with_val/60:.1f}m",
                    flush=True,
                )
                last_log_time = now
                last_log_step = step
                last_log_train_elapsed = train_elapsed
            if step % args.val_interval == 0 or step == args.max_steps:
                if is_main_process(rank):
                    val_start = time.time()
                    val_loss, val_scores = validate(
                        pipeline,
                        corrector.module if isinstance(corrector, DDP) else corrector,
                        val_loader,
                        args,
                        device,
                    )
                    val_elapsed = time.time() - val_start
                    val_elapsed_total += val_elapsed
                    val_runs += 1
                    print(
                        f"validation step={step} loss={val_loss:.6f} even3={val_scores} "
                        f"val_time={val_elapsed/60:.1f}m",
                        flush=True,
                    )
                if distributed:
                    dist.barrier()
            if step % args.save_interval == 0 or step == args.max_steps:
                save_start = time.time()
                if is_main_process(rank):
                    save_checkpoint(corrector, optimizer, args, step, args.out_dir)
                if distributed:
                    dist.barrier()
                save_elapsed = time.time() - save_start
                if is_main_process(rank):
                    save_elapsed_total += save_elapsed
                    save_runs += 1
            if step >= args.max_steps:
                break
        epoch += 1
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
