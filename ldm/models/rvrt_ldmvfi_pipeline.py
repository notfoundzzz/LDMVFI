import torch

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.rvrt_frontend import build_sr_frontend
from ldm.modules.condition_adapter import SRConditionAdapter
from ldm.util import instantiate_from_config


class RVRTLDMVFIPipeline:
    def __init__(
        self,
        ldm_config,
        ldm_ckpt,
        sr_mode="bicubic",
        scale=4,
        rvrt_root=None,
        rvrt_task="002_RVRT_videosr_bi_Vimeo_14frames",
        rvrt_ckpt=None,
        adapter_mode="none",
        adapter_hidden_channels=32,
        adapter_res_blocks=2,
        adapter_scale=1.0,
        device=None,
        tile=(0, 0, 0),
        tile_overlap=(2, 20, 20),
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter_mode = adapter_mode
        self.sr_frontend = build_sr_frontend(
            sr_mode=sr_mode,
            scale=scale,
            rvrt_root=rvrt_root,
            rvrt_task=rvrt_task,
            rvrt_ckpt=rvrt_ckpt,
            tile=tile,
            tile_overlap=tile_overlap,
        )
        self.sr_frontend = self.sr_frontend.to(self.device).eval()
        self.condition_adapter = None
        if adapter_mode == "sr_residual":
            self.condition_adapter = SRConditionAdapter(
                in_channels=9,
                cond_channels=6,
                hidden_channels=adapter_hidden_channels,
                num_res_blocks=adapter_res_blocks,
                scale=adapter_scale,
            ).to(self.device).eval()

        self.model = instantiate_from_config(ldm_config.model)
        state = torch.load(ldm_ckpt, map_location="cpu")
        self.model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def super_resolve_neighbors(self, prev_lr, next_lr):
        seq = torch.stack([prev_lr, next_lr], dim=1).to(self.device)
        out = self.sr_frontend(seq)
        return torch.clamp(out[:, 0], -1.0, 1.0), torch.clamp(out[:, 1], -1.0, 1.0)

    @torch.no_grad()
    def interpolate(self, prev_lr, next_lr, use_ddim=True, ddim_steps=200, ddim_eta=1.0):
        prev_lr = prev_lr.to(self.device)
        next_lr = next_lr.to(self.device)

        # RVRT expects [0,1], LDMVFI expects [-1,1]
        prev_01 = (prev_lr + 1.0) / 2.0
        next_01 = (next_lr + 1.0) / 2.0
        prev_sr, next_sr = self.super_resolve_neighbors(prev_01, next_01)
        prev_sr = prev_sr * 2.0 - 1.0
        next_sr = next_sr * 2.0 - 1.0

        xc = {"prev_frame": prev_sr, "next_frame": next_sr}
        c, phi_prev_list, phi_next_list = self.model.get_learned_conditioning(xc)
        adapter_summary = None
        if self.condition_adapter is not None:
            adapter_delta = self.condition_adapter(prev_sr, next_sr, target_hw=(c.shape[-2], c.shape[-1]))
            c = c + adapter_delta
            adapter_summary = adapter_delta.abs().mean(dim=1, keepdim=True)
        shape = (self.model.channels, c.shape[2], c.shape[3])
        if use_ddim:
            ddim = DDIMSampler(self.model)
            latent, _ = ddim.sample(ddim_steps, c.shape[0], shape, c, eta=ddim_eta, verbose=False)
        else:
            latent = self.model.sample_ddpm(conditioning=c, batch_size=c.shape[0], shape=shape, x_T=None, verbose=False)
        if isinstance(latent, tuple):
            latent = latent[0]
        out = self.model.decode_first_stage(latent, xc, phi_prev_list, phi_next_list)
        return torch.clamp(out, min=-1.0, max=1.0), prev_sr, next_sr, adapter_summary
