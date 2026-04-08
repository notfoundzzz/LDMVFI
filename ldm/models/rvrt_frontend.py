import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ldm.path_setup import ensure_repo_path


RVRT_TASK_CONFIGS = {
    "002_RVRT_videosr_bi_Vimeo_14frames": {
        "upscale": 4,
        "clip_size": 2,
        "img_size": [2, 64, 64],
        "window_size": [2, 8, 8],
        "num_blocks": [1, 2, 1],
        "depths": [2, 2, 2],
        "embed_dims": [144, 144, 144],
        "num_heads": [6, 6, 6],
        "inputconv_groups": [1, 1, 1, 1, 1, 1],
        "deformable_groups": 12,
        "attention_heads": 12,
        "attention_window": [3, 3],
        "cpu_cache_length": 100,
        "scale": 4,
    },
    "003_RVRT_videosr_bd_Vimeo_14frames": {
        "upscale": 4,
        "clip_size": 2,
        "img_size": [2, 64, 64],
        "window_size": [2, 8, 8],
        "num_blocks": [1, 2, 1],
        "depths": [2, 2, 2],
        "embed_dims": [144, 144, 144],
        "num_heads": [6, 6, 6],
        "inputconv_groups": [1, 1, 1, 1, 1, 1],
        "deformable_groups": 12,
        "attention_heads": 12,
        "attention_window": [3, 3],
        "cpu_cache_length": 100,
        "scale": 4,
    },
}


class BicubicVideoSR(torch.nn.Module):
    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale

    def forward(self, lq_frames):
        b, t, c, h, w = lq_frames.shape
        x = lq_frames.view(b * t, c, h, w)
        x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return torch.clamp(x.view(b, t, c, x.shape[-2], x.shape[-1]), 0.0, 1.0)


class RVRTVideoSR(torch.nn.Module):
    def __init__(
        self,
        rvrt_root,
        task="002_RVRT_videosr_bi_Vimeo_14frames",
        model_path=None,
        tile=(0, 0, 0),
        tile_overlap=(2, 20, 20),
    ):
        super().__init__()
        if task not in RVRT_TASK_CONFIGS:
            raise ValueError(f"Unsupported RVRT task: {task}")
        ensure_repo_path(rvrt_root)
        from models.network_rvrt import RVRT as net
        from main_test_rvrt import test_video

        cfg = RVRT_TASK_CONFIGS[task].copy()
        self.scale = cfg.pop("scale")
        self.model = net(**cfg)
        self.test_video_fn = test_video
        self.args = SimpleNamespace(
            task=task,
            tile=list(tile),
            tile_overlap=list(tile_overlap),
            scale=self.scale,
            window_size=[2, 8, 8],
            nonblind_denoising=False,
        )

        model_path = model_path or os.path.join(rvrt_root, "model_zoo", "rvrt", f"{task}.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RVRT checkpoint not found at {model_path}. "
                "Download it from the RVRT releases into model_zoo/rvrt."
            )
        pretrained = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(pretrained["params"] if "params" in pretrained else pretrained, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, lq_frames):
        return self.test_video_fn(lq_frames, self.model, self.args)


def build_sr_frontend(sr_mode, scale, rvrt_root=None, rvrt_task=None, rvrt_ckpt=None, tile=(0, 0, 0), tile_overlap=(2, 20, 20)):
    if sr_mode == "bicubic":
        return BicubicVideoSR(scale=scale)
    if sr_mode == "rvrt":
        if not rvrt_root:
            raise ValueError("rvrt_root is required when sr_mode=rvrt")
        return RVRTVideoSR(
            rvrt_root=rvrt_root,
            task=rvrt_task or "002_RVRT_videosr_bi_Vimeo_14frames",
            model_path=rvrt_ckpt,
            tile=tile,
            tile_overlap=tile_overlap,
        )
    raise ValueError(f"Unsupported sr_mode: {sr_mode}")
