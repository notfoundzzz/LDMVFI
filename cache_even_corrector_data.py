import argparse
import json
import os
import traceback
import time
from os.path import join

import torch
import torch.distributed as dist
from torchvision import transforms

from evaluate_rvrt_ldmvfi import collect_triplet_folders
from train_even_residual_corrector import (
    build_pipeline,
    flatten_even,
    init_distributed,
    is_main_process,
    load_sequence,
    make_even_batch,
    make_flow_inputs,
)


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_lr_root(dataset_root_lr, split, layout):
    split_root = join(dataset_root_lr, split) if split else dataset_root_lr
    if layout == "split_dir":
        return split_root
    if layout == "flat":
        return dataset_root_lr
    return split_root if os.path.isdir(split_root) else dataset_root_lr


def safe_key(key):
    return key.replace("/", "_").replace("\\", "_")


def to_cache_tensor(tensor, dtype):
    tensor = tensor.detach().cpu().contiguous()
    if dtype == "float16":
        return tensor.half()
    return tensor.float()


def save_cache_item(path, payload):
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def build_split_specs(args):
    specs = []
    if args.cache_train:
        specs.append(
            {
                "split": args.train_split,
                "list_file": args.train_list,
                "max_samples": args.max_train_samples,
            }
        )
    if args.cache_val:
        val_splits = parse_csv(args.val_splits)
        if args.val_lists:
            val_lists = parse_csv(args.val_lists)
        else:
            val_lists = [join(os.path.dirname(args.train_list), f"sep_{split}list.txt") for split in val_splits]
        if len(val_splits) != len(val_lists):
            raise ValueError(f"val_splits count {len(val_splits)} != val_lists count {len(val_lists)}")
        for split, list_file in zip(val_splits, val_lists):
            specs.append(
                {
                    "split": split,
                    "list_file": list_file,
                    "max_samples": args.max_val_samples,
                }
            )
    return specs


def collect_split_items(args, split, list_file, max_samples):
    lr_root = resolve_lr_root(args.dataset_root_lr, split, args.lr_split_layout)
    folders = collect_triplet_folders(lr_root, list_file=list_file, max_samples=max_samples)
    if not folders:
        raise FileNotFoundError(f"No Vimeo samples found for split={split}, lr_root={lr_root}, list_file={list_file}")
    return lr_root, folders


@torch.no_grad()
def cache_one_sample(args, pipeline, transform, split, key, lr_folder, device, sample_index):
    hr_folder = join(args.dataset_root_hr, *key.split("/"))
    lr_odd = load_sequence(lr_folder, [1, 3, 5, 7], transform).unsqueeze(0).to(device)
    gt_even = load_sequence(hr_folder, [2, 4, 6], transform).unsqueeze(0).to(device)
    _, prev, pred, nxt = make_even_batch(
        pipeline,
        lr_odd,
        seed=args.seed + sample_index * 3,
        use_ddim=bool(args.use_ddim),
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
    )
    prev_f, pred_f, nxt_f, gt_f = flatten_even(prev, pred, nxt, gt_even)
    warped_prev, warped_next = make_flow_inputs(prev_f, nxt_f, args)

    payload = {
        "key": key,
        "split": split,
        "prev": to_cache_tensor(prev.squeeze(0), args.cache_dtype),
        "pred": to_cache_tensor(pred.squeeze(0), args.cache_dtype),
        "next": to_cache_tensor(nxt.squeeze(0), args.cache_dtype),
        "gt": to_cache_tensor(gt_even.squeeze(0), args.cache_dtype),
        "meta": {
            "seed": args.seed + sample_index * 3,
            "use_ddim": bool(args.use_ddim),
            "ddim_steps": args.ddim_steps,
            "ddim_eta": args.ddim_eta,
            "rvrt_flow_mode": args.rvrt_flow_mode,
            "rvrt_use_flow_adapter": bool(args.rvrt_use_flow_adapter),
            "use_flow_inputs": bool(args.use_flow_inputs),
            "flow_backend": args.flow_backend,
        },
    }
    if warped_prev is not None and warped_next is not None:
        payload["warped_prev"] = to_cache_tensor(warped_prev.reshape(1, 3, *warped_prev.shape[1:]).squeeze(0), args.cache_dtype)
        payload["warped_next"] = to_cache_tensor(warped_next.reshape(1, 3, *warped_next.shape[1:]).squeeze(0), args.cache_dtype)
    return payload


def main():
    run_cache()


def run_cache():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldm_config", default="/data/Shenzhen/zhahongli/LDMVFI/configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml")
    parser.add_argument("--ldm_ckpt", default="/data/Shenzhen/zhahongli/models/ldmvfi/ldmvfi-vqflow-f32-c256-concat_max.ckpt")
    parser.add_argument("--dataset_root_hr", default="/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sequences")
    parser.add_argument("--dataset_root_lr", default="/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sequences_LR")
    parser.add_argument("--lr_split_layout", choices=["auto", "split_dir", "flat"], default="auto")
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--train_list", default="/data/Shenzhen/zzff/STVSR/data/vimeo_septuplet/sep_trainlist.txt")
    parser.add_argument("--val_splits", default="slow_test,medium_test,fast_test")
    parser.add_argument("--val_lists", default="")
    parser.add_argument("--cache_train", type=int, default=1)
    parser.add_argument("--cache_val", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=10)
    parser.add_argument("--cache_root", default="/data/Shenzhen/zhahongli/LDMVFI/cache/even_corrector_spynet_ddim200")
    parser.add_argument("--cache_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--rvrt_root", default="/data/Shenzhen/zhahongli/RVRT_flow_ablate")
    parser.add_argument("--rvrt_task", default="002_RVRT_videosr_bi_Vimeo_14frames")
    parser.add_argument("--rvrt_ckpt", default="/data/Shenzhen/zhahongli/RVRT/model_zoo/rvrt/002_RVRT_videosr_bi_Vimeo_14frames.pth")
    parser.add_argument("--rvrt_flow_mode", choices=["spynet", "zero", "raft", "spynet_raft_residual"], default="spynet")
    parser.add_argument("--rvrt_raft_variant", choices=["small", "large"], default="large")
    parser.add_argument("--rvrt_raft_ckpt", default=None)
    parser.add_argument("--rvrt_use_flow_adapter", type=int, default=0)
    parser.add_argument("--use_flow_inputs", type=int, default=0)
    parser.add_argument("--flow_backend", choices=["farneback", "raft"], default="farneback")
    parser.add_argument("--flow_raft_variant", choices=["small", "large"], default="large")
    parser.add_argument("--flow_raft_ckpt", default=None)
    parser.add_argument("--use_ddim", type=int, default=1)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_raw_weights", action="store_true")
    parser.add_argument("--allow_incomplete_ckpt", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    try:
        os.makedirs(args.cache_root, exist_ok=True)
        if is_main_process(rank):
            with open(join(args.cache_root, "args.json"), "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2)
            print(f"world_size={world_size}", flush=True)
            print(f"cache_root={args.cache_root}", flush=True)
        print(f"rank={rank} local_rank={local_rank} device={device} building frozen RVRT+LDMVFI pipeline...", flush=True)
        pipeline = build_pipeline(args)
        print(f"rank={rank} pipeline ready", flush=True)

        split_specs = build_split_specs(args)
        rank_manifest = join(args.cache_root, f"manifest_rank{rank:03d}.jsonl")
        with open(rank_manifest, "w", encoding="utf-8") as manifest:
            for spec in split_specs:
                split = spec["split"]
                lr_root, folders = collect_split_items(args, split, spec["list_file"], spec["max_samples"])
                split_dir = join(args.cache_root, split)
                os.makedirs(split_dir, exist_ok=True)
                assigned = [(idx, item) for idx, item in enumerate(folders) if idx % world_size == rank]
                print(
                    f"rank={rank} split={split} lr_root={lr_root} total={len(folders)} assigned={len(assigned)}",
                    flush=True,
                )
                start = time.time()
                done = 0
                skipped = 0
                for local_pos, (sample_index, (key, lr_folder)) in enumerate(assigned, start=1):
                    out_path = join(split_dir, f"{safe_key(key)}.pt")
                    if args.skip_existing and os.path.exists(out_path):
                        skipped += 1
                    else:
                        payload = cache_one_sample(args, pipeline, transform, split, key, lr_folder, device, sample_index)
                        save_cache_item(out_path, payload)
                        done += 1
                    record = {"split": split, "key": key, "path": out_path, "rank": rank}
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest.flush()
                    if local_pos == 1 or local_pos % args.log_interval == 0 or local_pos == len(assigned):
                        elapsed = time.time() - start
                        avg = elapsed / max(local_pos, 1)
                        eta = avg * max(len(assigned) - local_pos, 0)
                        print(
                            f"rank={rank} split={split} {local_pos}/{len(assigned)} "
                            f"done={done} skipped={skipped} elapsed={elapsed/60:.1f}m "
                            f"avg={avg:.2f}s eta={eta/60:.1f}m",
                            flush=True,
                        )
    except Exception:
        print(f"rank={rank} cache failed with exception:", flush=True)
        traceback.print_exc()
        raise

    if distributed:
        dist.barrier()
    if is_main_process(rank):
        manifest_path = join(args.cache_root, "manifest.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as out_manifest:
            for manifest_rank in range(world_size):
                path = join(args.cache_root, f"manifest_rank{manifest_rank:03d}.jsonl")
                if not os.path.exists(path):
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        out_manifest.write(line)
        print(f"wrote manifest={manifest_path}", flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
