import argparse
import os
from functools import partial

import torch
from omegaconf import OmegaConf

from ldm.path_setup import ensure_local_dependency_paths
from ldm.data import testsets_stsr
from ldm.models.diffusion.ddim import DDIMSampler
from main import instantiate_from_config

ensure_local_dependency_paths()


parser = argparse.ArgumentParser(description="Space-time super-resolution evaluation")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--dataset", type=str, default="Ucf_STSR")
parser.add_argument("--metrics", nargs="+", type=str, default=["PSNR", "SSIM", "LPIPS"])
parser.add_argument("--data_dir", type=str, default="D:\\")
parser.add_argument("--out_dir", type=str, default="eval_results_stsr")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--scale_factor", type=int, default=2)
parser.add_argument("--use_ddim", action="store_true")
parser.add_argument("--ddim_eta", type=float, default=1.0)
parser.add_argument("--ddim_steps", type=int, default=200)


def main():
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print("Model loaded successfully")

    if args.use_ddim and hasattr(model, "sample_ddpm"):
        ddim = DDIMSampler(model)
        sample_func = partial(ddim.sample, S=args.ddim_steps, eta=args.ddim_eta, verbose=False)
    elif hasattr(model, "sample_ddpm"):
        sample_func = partial(model.sample_ddpm, return_intermediates=False, verbose=False)
    else:
        sample_func = None

    os.makedirs(args.out_dir, exist_ok=True)
    print("Testing on dataset:", args.dataset)
    test_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(test_dir, exist_ok=True)

    if args.dataset.startswith("Ucf"):
        db_folder = "ucf"
    elif args.dataset.startswith("Snufilm"):
        db_folder = "snufilm"
    else:
        db_folder = args.dataset.replace("_STSR", "").lower()
    test_db = getattr(testsets_stsr, args.dataset)(os.path.join(args.data_dir, db_folder), scale_factor=args.scale_factor)
    test_db.eval(
        model,
        sample_func,
        metrics=args.metrics,
        output_dir=test_dir,
        resume=args.resume,
        sampler_kwargs={"use_ddim": args.use_ddim, "ddim_steps": args.ddim_steps, "ddim_eta": args.ddim_eta},
    )


if __name__ == "__main__":
    main()
