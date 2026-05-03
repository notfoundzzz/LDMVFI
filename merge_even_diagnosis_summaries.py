import argparse
import glob
import json
import os
from collections import Counter, defaultdict


def load_summaries(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shard summaries matched: {pattern}")
    summaries = []
    for path in paths:
        with open(path, "r", encoding="utf-8-sig") as f:
            item = json.load(f)
        item["_path"] = path
        summaries.append(item)
    return summaries


def merge_metrics(summaries):
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    for item in summaries:
        if "metric_sums" in item and "metric_counts" in item:
            for name, metric_values in item["metric_sums"].items():
                for metric, value in metric_values.items():
                    sums[name][metric] += float(value)
                    counts[name][metric] += int(item["metric_counts"][name][metric])
        else:
            # Backward-compatible fallback for older one-shot summaries.
            for name, metric_values in item["metrics"].items():
                for metric, value in metric_values.items():
                    count = int(item.get("samples", 1))
                    sums[name][metric] += float(value) * count
                    counts[name][metric] += count

    merged = {}
    for name, metric_values in sums.items():
        merged[name] = {}
        for metric, value in metric_values.items():
            count = counts[name][metric]
            merged[name][metric] = round(float(value) / float(count), 4) if count else None
    return merged, {name: dict(metric_counts) for name, metric_counts in counts.items()}


def normalize_counts(counter, total):
    return {name: round(float(count) / float(max(total, 1)), 4) for name, count in counter.items()}


def merge_counter(summaries, key):
    counter = Counter()
    for item in summaries:
        counter.update({name: int(value) for name, value in item.get(key, {}).items()})
    return counter


def merge_residual_stats(summaries):
    sums = defaultdict(float)
    count = 0
    for item in summaries:
        if "pred_abs_residual_stats_sum" in item:
            shard_count = int(item.get("pred_abs_residual_stats_count", 0))
            for key, value in item["pred_abs_residual_stats_sum"].items():
                sums[key] += float(value)
            count += shard_count
        elif "pred_abs_residual_stats" in item:
            shard_count = int(item.get("samples", 1))
            for key, value in item["pred_abs_residual_stats"].items():
                sums[key] += float(value) * shard_count
            count += shard_count
    return {key: round(float(value) / float(max(count, 1)), 6) for key, value in sums.items()}, count


def main():
    parser = argparse.ArgumentParser(description="Merge sharded diagnose_even_corrector_cache.py summaries.")
    parser.add_argument("--inputs", required=True, help="Glob pattern for shard summary JSON files.")
    parser.add_argument("--output", required=True, help="Merged summary JSON path.")
    args = parser.parse_args()

    summaries = load_summaries(args.inputs)
    metrics, metric_counts = merge_metrics(summaries)
    pixel_counts = merge_counter(summaries, "pixel_oracle_counts")
    frame_counts = merge_counter(summaries, "frame_oracle_counts")
    total_pixels = sum(int(item.get("total_pixels", 0)) for item in summaries)
    total_frames = sum(int(item.get("total_frames", 0)) for item in summaries)
    residual_stats, residual_count = merge_residual_stats(summaries)
    samples = sum(int(item.get("samples", 0)) for item in summaries)

    merged = {
        "cache_root": summaries[0].get("cache_root", ""),
        "splits": summaries[0].get("splits", ""),
        "samples": samples,
        "num_shards": len(summaries),
        "has_warp": any(bool(item.get("has_warp", False)) for item in summaries),
        "metrics": metrics,
        "metric_counts": metric_counts,
        "pixel_oracle_choice_rate": normalize_counts(pixel_counts, total_pixels),
        "pixel_oracle_counts": dict(pixel_counts),
        "total_pixels": total_pixels,
        "frame_oracle_choice_rate": normalize_counts(frame_counts, total_frames),
        "frame_oracle_counts": dict(frame_counts),
        "total_frames": total_frames,
        "pred_abs_residual_stats": residual_stats,
        "pred_abs_residual_stats_count": residual_count,
        "shard_summaries": [item["_path"] for item in summaries],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"merged {len(summaries)} summaries -> {args.output}")
    print("==== metrics ====")
    for name, scores in merged["metrics"].items():
        print(f"{name}: {scores}")
    print("==== oracle choice rates ====")
    print(f"pixel: {merged['pixel_oracle_choice_rate']}")
    print(f"frame: {merged['frame_oracle_choice_rate']}")
    print("==== pred residual stats ====")
    print(merged["pred_abs_residual_stats"])


if __name__ == "__main__":
    main()
