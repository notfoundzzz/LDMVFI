import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate_rvrt_ldmvfi import compute_metrics


class CachedEvenDataset(Dataset):
    def __init__(self, cache_root, splits, max_samples=0, shard_id=0, num_shards=1):
        if num_shards < 1:
            raise ValueError("--num_shards must be >= 1")
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"--shard_id must be in [0, {num_shards}), got {shard_id}")
        self.paths = []
        self.splits = [split.strip() for split in splits.split(",") if split.strip()]
        self.shard_id = shard_id
        self.num_shards = num_shards
        for split in self.splits:
            split_paths = sorted(glob.glob(join(cache_root, split, "*.pt")))
            if max_samples > 0:
                split_paths = split_paths[:max_samples]
            if num_shards > 1:
                split_paths = [path for idx, path in enumerate(split_paths) if idx % num_shards == shard_id]
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
            "split": payload.get("split", os.path.basename(os.path.dirname(path))),
            "key": payload.get("key", os.path.splitext(os.path.basename(path))[0]),
            "prev": payload["prev"].float(),
            "pred": payload["pred"].float(),
            "next": payload["next"].float(),
            "gt": payload["gt"].float(),
        }
        if "warped_prev" in payload and "warped_next" in payload:
            item["warped_prev"] = payload["warped_prev"].float()
            item["warped_next"] = payload["warped_next"].float()
        return item


def flatten_time(tensor):
    b, t, c, h, w = tensor.shape
    return tensor.reshape(b * t, c, h, w)


def mean_scores(scores, metrics):
    return {metric: round(float(np.mean([score[metric] for score in scores])), 4) for metric in metrics}


def append_metrics(store, name, metrics, gt, out, device):
    for i in range(out.shape[0]):
        store[name].append(compute_metrics(metrics, gt[i : i + 1], out[i : i + 1], device=device))


def pixel_oracle(candidates, gt):
    names = list(candidates.keys())
    stack = torch.stack([candidates[name] for name in names], dim=0)
    errors = torch.mean((stack - gt.unsqueeze(0)) ** 2, dim=2, keepdim=True)
    best = torch.argmin(errors, dim=0)
    gather_index = best.unsqueeze(0).expand(1, -1, 3, -1, -1)
    out = torch.gather(stack, dim=0, index=gather_index).squeeze(0)
    counts = torch.bincount(best.flatten().cpu(), minlength=len(names)).tolist()
    return out, dict(zip(names, counts))


def frame_oracle(candidates, gt):
    names = list(candidates.keys())
    stack = torch.stack([candidates[name] for name in names], dim=0)
    errors = torch.mean((stack - gt.unsqueeze(0)) ** 2, dim=(2, 3, 4))
    best = torch.argmin(errors, dim=0)
    gather_index = best.view(1, -1, 1, 1, 1).expand(1, -1, 3, gt.shape[-2], gt.shape[-1])
    out = torch.gather(stack, dim=0, index=gather_index).squeeze(0)
    counts = torch.bincount(best.cpu(), minlength=len(names)).tolist()
    return out, dict(zip(names, counts))


def residual_clip_oracle(pred, gt, max_residue):
    return torch.clamp(pred + torch.clamp(gt - pred, -max_residue, max_residue), -1.0, 1.0)


def collect_residual_stats(abs_residual, max_residues):
    flat = abs_residual.detach().float().cpu().flatten()
    quantiles = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.95, 0.99])).tolist()
    stats = {
        "mae": float(flat.mean().item()),
        "p50": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
    }
    for max_residue in max_residues:
        stats[f"within_{max_residue}"] = float((flat <= max_residue).float().mean().item())
    return stats


def merge_residual_stats(stats_list):
    merged = {}
    for key in stats_list[0].keys():
        merged[key] = round(float(np.mean([stats[key] for stats in stats_list])), 6)
    return merged


def sum_residual_stats(stats_list):
    return {key: float(sum(stats[key] for stats in stats_list)) for key in stats_list[0].keys()}


def metric_sums_and_counts(metric_store, metrics):
    sums = {}
    counts = {}
    for name, scores in metric_store.items():
        sums[name] = {metric: float(sum(score[metric] for score in scores)) for metric in metrics}
        counts[name] = {metric: len(scores) for metric in metrics}
    return sums, counts


def normalize_counts(counts, total):
    return {name: round(float(count) / float(max(total, 1)), 4) for name, count in counts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--splits", default="slow_test,medium_test,fast_test")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--metrics", nargs="+", default=["PSNR", "SSIM", "LPIPS"])
    parser.add_argument("--max_residues", default="0.25,0.5")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--summary_json", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_residues = [float(value) for value in args.max_residues.split(",") if value.strip()]
    dataset = CachedEvenDataset(
        args.cache_root,
        args.splits,
        max_samples=args.max_samples,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metric_store = defaultdict(list)
    residual_stats = []
    pixel_counts = Counter()
    frame_counts = Counter()
    total_pixels = 0
    total_frames = 0
    has_warp = None

    print(f"cache_root={args.cache_root}", flush=True)
    print(
        f"splits={args.splits} samples={len(dataset)} shard={args.shard_id}/{args.num_shards} device={device}",
        flush=True,
    )

    for batch_idx, batch in enumerate(loader, start=1):
        gt = flatten_time(batch["gt"]).to(device)
        pred = flatten_time(batch["pred"]).to(device)
        prev = flatten_time(batch["prev"]).to(device)
        nxt = flatten_time(batch["next"]).to(device)
        candidates = {
            "pred": pred,
            "neighbor_blend": torch.clamp(0.5 * (prev + nxt), -1.0, 1.0),
        }
        if "warped_prev" in batch and "warped_next" in batch:
            warped_prev = flatten_time(batch["warped_prev"]).to(device)
            warped_next = flatten_time(batch["warped_next"]).to(device)
            candidates["warped_prev"] = warped_prev
            candidates["warped_next"] = warped_next
            candidates["warp_blend"] = torch.clamp(0.5 * (warped_prev + warped_next), -1.0, 1.0)
            has_warp = True
        elif has_warp is None:
            has_warp = False

        for name, out in candidates.items():
            append_metrics(metric_store, name, args.metrics, gt, out, device)

        for max_residue in max_residues:
            name = f"residual_clip_oracle_{max_residue}"
            append_metrics(metric_store, name, args.metrics, gt, residual_clip_oracle(pred, gt, max_residue), device)

        pix_oracle, pix_counts = pixel_oracle(candidates, gt)
        frm_oracle, frm_counts = frame_oracle(candidates, gt)
        append_metrics(metric_store, "pixel_oracle", args.metrics, gt, pix_oracle, device)
        append_metrics(metric_store, "frame_oracle", args.metrics, gt, frm_oracle, device)
        pixel_counts.update(pix_counts)
        frame_counts.update(frm_counts)
        total_pixels += int(gt.shape[0] * gt.shape[-2] * gt.shape[-1])
        total_frames += int(gt.shape[0])
        residual_stats.append(collect_residual_stats(torch.abs(gt - pred), max_residues))

        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"processed {batch_idx}/{len(loader)} batches", flush=True)

    metric_sums, metric_counts = metric_sums_and_counts(metric_store, args.metrics)
    residual_summary = merge_residual_stats(residual_stats)
    summary = {
        "cache_root": args.cache_root,
        "splits": args.splits,
        "samples": len(dataset),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "has_warp": bool(has_warp),
        "metrics": {name: mean_scores(scores, args.metrics) for name, scores in metric_store.items()},
        "metric_sums": metric_sums,
        "metric_counts": metric_counts,
        "pixel_oracle_choice_rate": normalize_counts(pixel_counts, total_pixels),
        "pixel_oracle_counts": dict(pixel_counts),
        "total_pixels": total_pixels,
        "frame_oracle_choice_rate": normalize_counts(frame_counts, total_frames),
        "frame_oracle_counts": dict(frame_counts),
        "total_frames": total_frames,
        "pred_abs_residual_stats": residual_summary,
        "pred_abs_residual_stats_sum": sum_residual_stats(residual_stats),
        "pred_abs_residual_stats_count": len(residual_stats),
    }

    print("==== metrics ====", flush=True)
    for name, scores in summary["metrics"].items():
        print(f"{name}: {scores}", flush=True)
    print("==== oracle choice rates ====", flush=True)
    print(f"pixel: {summary['pixel_oracle_choice_rate']}", flush=True)
    print(f"frame: {summary['frame_oracle_choice_rate']}", flush=True)
    print("==== pred residual stats ====", flush=True)
    print(summary["pred_abs_residual_stats"], flush=True)

    if args.summary_json:
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote summary={args.summary_json}", flush=True)


if __name__ == "__main__":
    main()
