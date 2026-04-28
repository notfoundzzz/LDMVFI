import argparse
import glob
import json
import os
import random
import time
from os.path import join

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from evaluate_rvrt_ldmvfi import compute_metrics
from ldm.models.even_residual_corrector import EvenFrameResidualCorrector
from train_even_residual_corrector import (
    charbonnier_loss,
    estimate_remaining_events,
    init_distributed,
    is_main_process,
    save_checkpoint,
)


class CachedEvenCorrectorDataset(Dataset):
    def __init__(self, cache_root, splits, max_samples=0, require_flow_inputs=False):
        self.cache_root = cache_root
        self.splits = [split.strip() for split in splits.split(",") if split.strip()]
        self.require_flow_inputs = bool(require_flow_inputs)
        self.paths = []
        for split in self.splits:
            split_paths = sorted(glob.glob(join(cache_root, split, "*.pt")))
            if max_samples > 0:
                split_paths = split_paths[:max_samples]
            self.paths.extend(split_paths)
        if not self.paths:
            raise FileNotFoundError(f"No cached .pt files found under {cache_root} for splits={self.splits}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        payload = torch.load(path, map_location="cpu")
        item = {
            "path": path,
            "key": payload.get("key", os.path.splitext(os.path.basename(path))[0]),
            "split": payload.get("split", os.path.basename(os.path.dirname(path))),
            "prev": payload["prev"].float(),
            "pred": payload["pred"].float(),
            "next": payload["next"].float(),
            "gt": payload["gt"].float(),
        }
        if "warped_prev" in payload and "warped_next" in payload:
            item["warped_prev"] = payload["warped_prev"].float()
            item["warped_next"] = payload["warped_next"].float()
        elif self.require_flow_inputs:
            item["warped_prev"] = payload["prev"].float()
            item["warped_next"] = payload["next"].float()
        return item


def flatten_cached_batch(batch, device, use_flow_inputs):
    prev, pred, nxt, gt = (
        batch["prev"].to(device),
        batch["pred"].to(device),
        batch["next"].to(device),
        batch["gt"].to(device),
    )
    b, t, c, h, w = pred.shape
    prev_f = prev.reshape(b * t, c, h, w)
    pred_f = pred.reshape(b * t, c, h, w)
    nxt_f = nxt.reshape(b * t, c, h, w)
    gt_f = gt.reshape(b * t, c, h, w)
    warped_prev = None
    warped_next = None
    if use_flow_inputs:
        warped_prev = batch["warped_prev"].to(device).reshape(b * t, c, h, w)
        warped_next = batch["warped_next"].to(device).reshape(b * t, c, h, w)
    return prev_f, pred_f, nxt_f, gt_f, warped_prev, warped_next


@torch.no_grad()
def validate(corrector, loader, args, device):
    corrector.eval()
    losses = []
    scores = []
    for batch in loader:
        prev_f, pred_f, nxt_f, gt_f, warped_prev, warped_next = flatten_cached_batch(
            batch, device, bool(args.use_flow_inputs)
        )
        refined = corrector(prev_f, pred_f, nxt_f, warped_prev, warped_next)
        losses.append(charbonnier_loss(refined, gt_f).item())
        for i in range(refined.shape[0]):
            scores.append(compute_metrics(args.metrics, gt_f[i : i + 1], refined[i : i + 1], device=device))
    averages = {metric: round(float(np.mean([score[metric] for score in scores])), 4) for metric in args.metrics}
    return float(np.mean(losses)), averages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--train_splits", default="train")
    parser.add_argument("--val_splits", default="slow_test,medium_test,fast_test")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--out_dir", default="/data/Shenzhen/zhahongli/LDMVFI/experiments/even_residual_corrector_cached")
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--max_residue", type=float, default=0.25)
    parser.add_argument("--use_flow_inputs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--metrics", nargs="+", default=["PSNR", "SSIM", "LPIPS"])
    parser.add_argument("--seed", type=int, default=1234)
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

    train_set = CachedEvenCorrectorDataset(
        args.cache_root,
        args.train_splits,
        max_samples=args.max_train_samples,
        require_flow_inputs=bool(args.use_flow_inputs),
    )
    val_set = CachedEvenCorrectorDataset(
        args.cache_root,
        args.val_splits,
        max_samples=args.max_val_samples,
        require_flow_inputs=bool(args.use_flow_inputs),
    )
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
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    corrector = EvenFrameResidualCorrector(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        max_residue=args.max_residue,
        use_flow_inputs=bool(args.use_flow_inputs),
    ).to(device)
    if distributed:
        corrector = DDP(corrector, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if is_main_process(rank):
        print(f"world_size={world_size}", flush=True)
        print(f"cache_root={args.cache_root}", flush=True)
        print(f"datasets ready: train={len(train_set)} val={len(val_set)}", flush=True)
        print(
            f"corrector hidden={args.hidden_channels} blocks={args.num_blocks} "
            f"max_residue={args.max_residue} use_flow_inputs={bool(args.use_flow_inputs)}",
            flush=True,
        )

    step = 0
    epoch = 0
    start_time = time.time()
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
            prev_f, pred_f, nxt_f, gt_f, warped_prev, warped_next = flatten_cached_batch(
                batch, device, bool(args.use_flow_inputs)
            )
            refined = corrector(prev_f, pred_f, nxt_f, warped_prev, warped_next)
            loss = charbonnier_loss(refined, gt_f)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), max_norm=1.0)
            optimizer.step()

            step += 1
            train_elapsed += time.time() - step_start
            if is_main_process(rank) and (step == 1 or step % 25 == 0):
                elapsed = time.time() - start_time
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
                    f"avg_train_step={avg_train_step_time:.3f}s recent_train_step={recent_train_step_time:.3f}s "
                    f"train_eta={train_eta/60:.1f}m eta_with_val={eta_with_val/60:.1f}m",
                    flush=True,
                )
                last_log_step = step
                last_log_train_elapsed = train_elapsed

            if step % args.val_interval == 0 or step == args.max_steps:
                if is_main_process(rank):
                    val_start = time.time()
                    val_loss, val_scores = validate(
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
