from contextlib import nullcontext

import torch

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.flow_guidance import build_flow_guided_middle_prior
from ldm.models.rvrt_frontend import build_sr_frontend
from ldm.util import instantiate_from_config


class RVRTPiSADualLoRAPipeline:
    """@brief Standalone evaluation pipeline for PiSA-style dual LoRA.

    @example Set `pixel_scale=1.0` and `semantic_scale=0.0`
    to observe the pixel branch effect alone.
    """

    def __init__(
        self,
        ldm_config,
        ldm_ckpt,
        sr_mode="bicubic",
        scale=4,
        rvrt_root=None,
        rvrt_task="002_RVRT_videosr_bi_Vimeo_14frames",
        rvrt_ckpt=None,
        pixel_scale=1.0,
        semantic_scale=1.0,
        device=None,
        tile=(0, 0, 0),
        tile_overlap=(2, 20, 20),
        use_ema=True,
        strict_checkpoint=True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pixel_scale = float(pixel_scale)
        self.semantic_scale = float(semantic_scale)
        self.use_ema = bool(use_ema)
        self.strict_checkpoint = bool(strict_checkpoint)
        self.last_flow_prior = None

        self.model = instantiate_from_config(ldm_config.model)
        state = torch.load(ldm_ckpt, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys while loading PiSA Dual-LoRA model: {unexpected[:10]}")
        missing_lora = [key for key in missing if "lora_" in key]
        if missing_lora:
            raise ValueError(f"Checkpoint is missing PiSA Dual-LoRA weights: {missing_lora[:10]}")
        disallowed_missing = [key for key in missing if not self._is_allowed_missing_key(key)]
        if self.strict_checkpoint and disallowed_missing:
            raise ValueError(
                "Checkpoint is missing required keys for PiSA Dual-LoRA evaluation. "
                f"First keys: {disallowed_missing[:10]}"
            )
        self.model = self.model.to(self.device).eval()
        self.sr_frontend, self.sr_frontend_source = self._build_eval_frontend(
            sr_mode=sr_mode,
            scale=scale,
            rvrt_root=rvrt_root,
            rvrt_task=rvrt_task,
            rvrt_ckpt=rvrt_ckpt,
            tile=tile,
            tile_overlap=tile_overlap,
        )
        print(f"Evaluation RVRT source: {self.sr_frontend_source}")

    def _is_allowed_missing_key(self, key):
        if key.startswith("model_ema.") and not self.use_ema:
            return True
        return False

    def _build_eval_frontend(self, sr_mode, scale, rvrt_root, rvrt_task, rvrt_ckpt, tile, tile_overlap):
        model_frontend = getattr(self.model, "rvrt_frontend", None)
        if sr_mode == "rvrt" and model_frontend is not None:
            frontend = model_frontend.to(self.device).eval()
            return frontend, "checkpoint rvrt_frontend"

        frontend = build_sr_frontend(
            sr_mode=sr_mode,
            scale=scale,
            rvrt_root=rvrt_root,
            rvrt_task=rvrt_task,
            rvrt_ckpt=rvrt_ckpt,
            tile=tile,
            tile_overlap=tile_overlap,
        )
        return frontend.to(self.device).eval(), "external sr_frontend"

    def _sampling_scope(self, seed):
        if seed is None:
            return nullcontext()

        devices = [self.device] if self.device.type == "cuda" else []
        return torch.random.fork_rng(devices=devices)

    def _make_initial_latent(self, shape, seed):
        with self._sampling_scope(seed):
            if seed is not None:
                torch.manual_seed(seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
            return torch.randn(shape, device=self.device)

    def _sample_prediction(
        self,
        xc,
        conditioning,
        phi_prev_list,
        phi_next_list,
        use_ddim,
        ddim_steps,
        ddim_eta,
        x_T,
        pixel_scale,
        semantic_scale,
    ):
        if hasattr(self.model, "set_pisa_dual_lora_scales"):
            self.model.set_pisa_dual_lora_scales(pixel_scale=pixel_scale, semantic_scale=semantic_scale)
        shape = (self.model.channels, conditioning.shape[2], conditioning.shape[3])
        sample_x_T = x_T.clone() if x_T is not None else None
        if use_ddim:
            ddim = DDIMSampler(self.model)
            latent, _ = ddim.sample(
                ddim_steps,
                conditioning.shape[0],
                shape,
                conditioning,
                eta=ddim_eta,
                verbose=False,
                x_T=sample_x_T,
            )
        else:
            latent = self.model.sample_ddpm(
                conditioning=conditioning,
                batch_size=conditioning.shape[0],
                shape=shape,
                x_T=sample_x_T,
                verbose=False,
            )
        if isinstance(latent, tuple):
            latent = latent[0]
        return self.model.decode_first_stage(latent, xc, phi_prev_list, phi_next_list)

    @torch.no_grad()
    def super_resolve_neighbors(self, prev_lr, next_lr):
        seq = torch.stack([prev_lr, next_lr], dim=1).to(self.device)
        out = self.sr_frontend(seq)
        return out[:, 0], out[:, 1]

    @torch.no_grad()
    def interpolate(self, prev_lr, next_lr, use_ddim=True, ddim_steps=200, ddim_eta=0.0, seed=None):
        prev_lr = prev_lr.to(self.device)
        next_lr = next_lr.to(self.device)
        prev_01 = (prev_lr + 1.0) / 2.0
        next_01 = (next_lr + 1.0) / 2.0
        prev_sr, next_sr = self.super_resolve_neighbors(prev_01, next_01)
        prev_sr = torch.clamp(prev_sr * 2.0 - 1.0, min=-1.0, max=1.0)
        next_sr = torch.clamp(next_sr * 2.0 - 1.0, min=-1.0, max=1.0)

        scope = self.model.ema_scope() if self.use_ema and hasattr(self.model, "ema_scope") else nullcontext()
        with scope:
            xc = {"prev_frame": prev_sr, "next_frame": next_sr}
            if getattr(self.model, "use_flow_guidance", False):
                self.last_flow_prior = build_flow_guided_middle_prior(
                    prev_lr,
                    next_lr,
                    prev_sr,
                    next_sr,
                    backend=getattr(self.model, "flow_backend", "farneback"),
                    raft_variant=getattr(self.model, "flow_raft_variant", "large"),
                    raft_ckpt=getattr(self.model, "flow_raft_ckpt", None),
                )
                xc[getattr(self.model, "cond_flow_key", "flow_prior")] = self.last_flow_prior
            else:
                self.last_flow_prior = None
            c, phi_prev_list, phi_next_list = self.model.get_learned_conditioning(xc)
            latent_shape = (c.shape[0], self.model.channels, c.shape[2], c.shape[3])
            x_T = self._make_initial_latent(latent_shape, seed)

            pixel_pred = self._sample_prediction(
                xc,
                c,
                phi_prev_list,
                phi_next_list,
                use_ddim,
                ddim_steps,
                ddim_eta,
                x_T,
                pixel_scale=1.0,
                semantic_scale=0.0,
            )
            if self.semantic_scale == 0.0:
                out = self.pixel_scale * pixel_pred
            else:
                combined_pred = self._sample_prediction(
                    xc,
                    c,
                    phi_prev_list,
                    phi_next_list,
                    use_ddim,
                    ddim_steps,
                    ddim_eta,
                    x_T,
                    pixel_scale=1.0,
                    semantic_scale=1.0,
                )
                semantic_residual = combined_pred - pixel_pred
                out = self.pixel_scale * pixel_pred + self.semantic_scale * semantic_residual
        return torch.clamp(out, min=-1.0, max=1.0), prev_sr, next_sr
