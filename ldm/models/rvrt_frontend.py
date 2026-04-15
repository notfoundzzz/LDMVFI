import os
from types import SimpleNamespace
from contextlib import nullcontext

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

        cfg = RVRT_TASK_CONFIGS[task].copy()
        self.scale = cfg.pop("scale")
        self.model = net(**cfg)
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

    def _grad_context(self):
        return nullcontext() if any(p.requires_grad for p in self.model.parameters()) else torch.no_grad()

    def forward(self, lq_frames):
        with self._grad_context():
            return self.test_video(lq_frames)

    def test_video(self, lq):
        num_frame_testing = self.args.tile[0]
        if num_frame_testing:
            sf = self.args.scale
            num_frame_overlapping = self.args.tile_overlap[0]
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if self.args.nonblind_denoising else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d - num_frame_testing, stride)) + [max(0, d - num_frame_testing)]
            E = torch.zeros(b, d, c, h * sf, w * sf, device=lq.device, dtype=lq.dtype)
            W = torch.zeros(b, d, 1, 1, 1, device=lq.device, dtype=lq.dtype)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx : d_idx + num_frame_testing, ...]
                out_clip = self.test_clip(lq_clip)
                out_clip_mask = torch.ones(
                    (b, min(num_frame_testing, d), 1, 1, 1), device=lq.device, dtype=lq.dtype
                )

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping // 2 :, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping // 2 :, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, : num_frame_overlapping // 2, ...] *= 0
                        out_clip_mask[:, : num_frame_overlapping // 2, ...] *= 0

                E[:, d_idx : d_idx + num_frame_testing, ...].add_(out_clip)
                W[:, d_idx : d_idx + num_frame_testing, ...].add_(out_clip_mask)
            return E.div_(W)

        window_size = self.args.window_size
        d_old = lq.size(1)
        d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
        lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
        output = self.test_clip(lq)
        return output[:, :d_old, :, :, :]

    def test_clip(self, lq):
        sf = self.args.scale
        window_size = self.args.window_size
        size_patch_testing = self.args.tile[1]
        assert size_patch_testing % window_size[-1] == 0, "testing patch size should be a multiple of window_size."

        if size_patch_testing:
            overlap_size = self.args.tile_overlap[1]
            not_overlap_border = True

            b, d, c, h, w = lq.size()
            c = c - 1 if self.args.nonblind_denoising else c
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h - size_patch_testing, stride)) + [max(0, h - size_patch_testing)]
            w_idx_list = list(range(0, w - size_patch_testing, stride)) + [max(0, w - size_patch_testing)]
            E = torch.zeros(b, d, c, h * sf, w * sf, device=lq.device, dtype=lq.dtype)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = lq[..., h_idx : h_idx + size_patch_testing, w_idx : w_idx + size_patch_testing]
                    out_patch = self.model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    if not_overlap_border:
                        if h_idx < h_idx_list[-1]:
                            out_patch[..., -overlap_size // 2 :, :] *= 0
                            out_patch_mask[..., -overlap_size // 2 :, :] *= 0
                        if w_idx < w_idx_list[-1]:
                            out_patch[..., :, -overlap_size // 2 :] *= 0
                            out_patch_mask[..., :, -overlap_size // 2 :] *= 0
                        if h_idx > h_idx_list[0]:
                            out_patch[..., : overlap_size // 2, :] *= 0
                            out_patch_mask[..., : overlap_size // 2, :] *= 0
                        if w_idx > w_idx_list[0]:
                            out_patch[..., :, : overlap_size // 2] *= 0
                            out_patch_mask[..., :, : overlap_size // 2] *= 0

                    E[..., h_idx * sf : (h_idx + size_patch_testing) * sf, w_idx * sf : (w_idx + size_patch_testing) * sf].add_(out_patch)
                    W[..., h_idx * sf : (h_idx + size_patch_testing) * sf, w_idx * sf : (w_idx + size_patch_testing) * sf].add_(out_patch_mask)
            return E.div_(W)

        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = self.model(lq)
        return output[:, :, :, : h_old * sf, : w_old * sf]


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
