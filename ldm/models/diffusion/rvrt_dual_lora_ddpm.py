import torch
from torch.optim.lr_scheduler import LambdaLR

from ldm.models.diffusion.ddpm import LatentDiffusionVFI
from ldm.models.rvrt_frontend import build_sr_frontend
from ldm.modules.dual_lora import (
    inject_dual_lora_modules,
    normalize_lora_patterns,
    pixel_lora_parameters,
    semantic_lora_parameters,
    set_dual_lora_scales,
)
from ldm.modules.ema import LitEma
from ldm.util import instantiate_from_config


class LatentDiffusionVFIRVRTDualLoRA(LatentDiffusionVFI):
    def __init__(
        self,
        rvrt_root,
        rvrt_task="002_RVRT_videosr_bi_Vimeo_14frames",
        rvrt_ckpt=None,
        rvrt_tile=(0, 0, 0),
        rvrt_tile_overlap=(2, 20, 20),
        pixel_lora_rank=8,
        pixel_lora_alpha=16.0,
        semantic_lora_rank=8,
        semantic_lora_alpha=16.0,
        pixel_lora_patterns=None,
        semantic_lora_patterns=None,
        semantic_start_step=20000,
        pixel_scale=1.0,
        semantic_scale=1.0,
        lr_prev_key="prev_frame_lr",
        lr_next_key="next_frame_lr",
        cond_prev_key="prev_frame",
        cond_next_key="next_frame",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr_prev_key = lr_prev_key
        self.lr_next_key = lr_next_key
        self.cond_prev_key = cond_prev_key
        self.cond_next_key = cond_next_key
        self.semantic_start_step = semantic_start_step
        self.pixel_scale = pixel_scale
        self.semantic_scale = semantic_scale
        self._dual_stage = None
        self.pixel_lora_patterns = normalize_lora_patterns(pixel_lora_patterns, ("input_blocks", "output_blocks"))
        self.semantic_lora_patterns = normalize_lora_patterns(semantic_lora_patterns, ("middle_block",))

        self.rvrt_frontend = build_sr_frontend(
            sr_mode="rvrt",
            scale=4,
            rvrt_root=rvrt_root,
            rvrt_task=rvrt_task,
            rvrt_ckpt=rvrt_ckpt,
            tile=tuple(rvrt_tile),
            tile_overlap=tuple(rvrt_tile_overlap),
        )
        self.rvrt_frontend.eval()
        for param in self.rvrt_frontend.parameters():
            param.requires_grad = False

        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False

        self.dual_lora_targets, self.pixel_targets, self.semantic_targets = inject_dual_lora_modules(
            self.model,
            pixel_rank=pixel_lora_rank,
            pixel_alpha=pixel_lora_alpha,
            semantic_rank=semantic_lora_rank,
            semantic_alpha=semantic_lora_alpha,
            pixel_patterns=self.pixel_lora_patterns,
            semantic_patterns=self.semantic_lora_patterns,
        )
        if not self.dual_lora_targets:
            raise RuntimeError("No Dual-LoRA target modules were found in diffusion model")
        print(
            f"Injected Dual-LoRA into {len(self.dual_lora_targets)} modules "
            f"(pixel={len(self.pixel_targets)}, semantic={len(self.semantic_targets)})"
        )

        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self._set_dual_lora_stage("pixel" if self.semantic_start_step > 0 else "both")

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["dual_lora_metadata"] = {
            "pixel_lora_patterns": list(self.pixel_lora_patterns),
            "semantic_lora_patterns": list(self.semantic_lora_patterns),
            "semantic_start_step": int(self.semantic_start_step),
            "pixel_scale": float(self.pixel_scale),
            "semantic_scale": float(self.semantic_scale),
        }

    def on_fit_start(self):
        super().on_fit_start()
        self.rvrt_frontend = self.rvrt_frontend.to(self.device).eval()

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.semantic_start_step > 0 and self.global_step >= self.semantic_start_step and self._dual_stage != "both":
            self._set_dual_lora_stage("both")
        parent_hook = getattr(super(), "on_train_batch_start", None)
        if parent_hook is None:
            return None
        try:
            return parent_hook(batch, batch_idx, dataloader_idx)
        except TypeError:
            return parent_hook(batch, batch_idx)

    def _set_dual_lora_stage(self, stage: str):
        if stage == self._dual_stage:
            return
        if stage == "pixel":
            set_dual_lora_scales(self.model, pixel_scale=self.pixel_scale, semantic_scale=0.0)
            print("Dual-LoRA stage: pixel-only")
        elif stage == "both":
            set_dual_lora_scales(self.model, pixel_scale=self.pixel_scale, semantic_scale=self.semantic_scale)
            print("Dual-LoRA stage: pixel + semantic")
        else:
            raise ValueError(f"Unsupported Dual-LoRA stage: {stage}")
        self._dual_stage = stage

    def set_dual_lora_scales(self, pixel_scale=1.0, semantic_scale=1.0):
        self.pixel_scale = pixel_scale
        self.semantic_scale = semantic_scale
        set_dual_lora_scales(self.model, pixel_scale=pixel_scale, semantic_scale=semantic_scale)

    @torch.no_grad()
    def _super_resolve_neighbors(self, batch, bs=None):
        prev_lr = super(LatentDiffusionVFI, self).get_input(batch, self.lr_prev_key).to(self.device)
        next_lr = super(LatentDiffusionVFI, self).get_input(batch, self.lr_next_key).to(self.device)
        if bs is not None:
            prev_lr = prev_lr[:bs]
            next_lr = next_lr[:bs]

        prev_01 = (prev_lr + 1.0) / 2.0
        next_01 = (next_lr + 1.0) / 2.0
        seq = torch.stack([prev_01, next_01], dim=1)
        out = self.rvrt_frontend(seq)
        prev_sr = torch.clamp(out[:, 0] * 2.0 - 1.0, min=-1.0, max=1.0)
        next_sr = torch.clamp(out[:, 1] * 2.0 - 1.0, min=-1.0, max=1.0)
        return prev_sr, next_sr

    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        return_phi=False,
        bs=None,
    ):
        x = super(LatentDiffusionVFI, self).get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        with torch.no_grad():
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            prev_sr, next_sr = self._super_resolve_neighbors(batch, bs=bs)
            xc = {self.cond_prev_key: prev_sr, self.cond_next_key: next_sr}
            c, phi_prev_list, phi_next_list = self.get_learned_conditioning(xc)

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z, xc, phi_prev_list, phi_next_list)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        if return_phi:
            out.append(phi_prev_list)
            out.append(phi_next_list)
        return out

    def configure_optimizers(self):
        pixel_params = list(pixel_lora_parameters(self.model))
        semantic_params = list(semantic_lora_parameters(self.model))
        if not pixel_params and not semantic_params:
            raise RuntimeError("No Dual-LoRA parameters found")
        params = []
        if pixel_params:
            params.append({"params": pixel_params, "lr": self.learning_rate, "name": "pixel_lora"})
        if semantic_params:
            params.append({"params": semantic_params, "lr": self.learning_rate, "name": "semantic_lora"})
        if self.learn_logvar:
            params.append({"params": [self.logvar], "lr": self.learning_rate, "name": "logvar"})
        opt = torch.optim.AdamW(params, lr=self.learning_rate)
        print(
            f"Optimizer groups: pixel={len(pixel_params)} params, semantic={len(semantic_params)} params, "
            f"lr={self.learning_rate:.2e}"
        )
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = [{"scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule), "interval": "step", "frequency": 1}]
            return [opt], scheduler
        return opt
