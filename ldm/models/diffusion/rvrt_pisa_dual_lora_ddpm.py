from contextlib import nullcontext

import torch
from torch.optim.lr_scheduler import LambdaLR

from ldm.models.diffusion.ddpm import LatentDiffusionVFI
from ldm.models.rvrt_frontend import build_sr_frontend
from ldm.modules.ema import LitEma
from ldm.modules.pisa_dual_lora import (
    inject_pisa_dual_lora_modules,
    pixel_lora_parameters,
    semantic_lora_parameters,
    set_pisa_dual_lora_scales,
)
from ldm.util import instantiate_from_config


class LatentDiffusionVFIRVRTPiSADualLoRA(LatentDiffusionVFI):
    """@brief RVRT + LDMVFI training class with PiSA-style dual LoRA.

    @example When `dual_lora_train_mode=joint`, both pixel and semantic
    branches are trained together.
    """

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
        pixel_lora_groups=None,
        semantic_lora_groups=None,
        lora_target_suffixes=None,
        dual_lora_train_mode="joint",
        semantic_start_step=0,
        pixel_scale=1.0,
        semantic_scale=1.0,
        rvrt_train_mode="frozen",
        rvrt_train_patterns=None,
        rvrt_lr=1.0e-5,
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
        self.rvrt_train_mode = rvrt_train_mode
        self.rvrt_lr = rvrt_lr
        self.dual_lora_train_mode = dual_lora_train_mode
        self.semantic_start_step = int(semantic_start_step)
        self.pixel_scale = float(pixel_scale)
        self.semantic_scale = float(semantic_scale)
        self._active_stage = None
        self.pixel_lora_groups = self._normalize_items(pixel_lora_groups, ("encoder", "decoder", "others"))
        self.semantic_lora_groups = self._normalize_items(semantic_lora_groups, ("encoder", "decoder", "others"))
        self.lora_target_suffixes = self._normalize_items(
            lora_target_suffixes,
            (
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "qkv",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "in_layers.2",
                "out_layers.3",
                "skip_connection",
            ),
        )

        self.rvrt_frontend = build_sr_frontend(
            sr_mode="rvrt",
            scale=4,
            rvrt_root=rvrt_root,
            rvrt_task=rvrt_task,
            rvrt_ckpt=rvrt_ckpt,
            tile=tuple(rvrt_tile),
            tile_overlap=tuple(rvrt_tile_overlap),
        )
        self.rvrt_trainable_patterns = self._normalize_rvrt_train_patterns(rvrt_train_patterns)
        self._configure_rvrt_training()

        for param in self.first_stage_model.parameters():
            param.requires_grad = False
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False

        self.dual_lora_targets, self.pixel_targets, self.semantic_targets = inject_pisa_dual_lora_modules(
            self.model,
            pixel_rank=pixel_lora_rank,
            pixel_alpha=pixel_lora_alpha,
            semantic_rank=semantic_lora_rank,
            semantic_alpha=semantic_lora_alpha,
            pixel_groups=self.pixel_lora_groups,
            semantic_groups=self.semantic_lora_groups,
            target_suffixes=self.lora_target_suffixes,
        )
        if not self.dual_lora_targets:
            raise RuntimeError("No PiSA Dual-LoRA target modules were found in diffusion model")
        print(
            f"Injected PiSA Dual-LoRA into {len(self.dual_lora_targets)} modules "
            f"(pixel={len(self.pixel_targets)}, semantic={len(self.semantic_targets)})"
        )
        print(f"Pixel groups: {self.pixel_lora_groups}")
        print(f"Semantic groups: {self.semantic_lora_groups}")

        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self._set_training_stage("joint" if self.dual_lora_train_mode == "joint" else "pixel")

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["pisa_dual_lora_metadata"] = {
            "pixel_lora_groups": list(self.pixel_lora_groups),
            "semantic_lora_groups": list(self.semantic_lora_groups),
            "lora_target_suffixes": list(self.lora_target_suffixes),
            "dual_lora_train_mode": self.dual_lora_train_mode,
            "semantic_start_step": self.semantic_start_step,
            "pixel_scale": self.pixel_scale,
            "semantic_scale": self.semantic_scale,
        }

    def on_fit_start(self):
        super().on_fit_start()
        self.rvrt_frontend = self.rvrt_frontend.to(self.device)
        if self._rvrt_requires_grad():
            self.rvrt_frontend.train()
        else:
            self.rvrt_frontend.eval()

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.dual_lora_train_mode == "staged":
            next_stage = "joint" if self.global_step >= self.semantic_start_step else "pixel"
            self._set_training_stage(next_stage)
        parent_hook = getattr(super(), "on_train_batch_start", None)
        if parent_hook is None:
            return None
        try:
            return parent_hook(batch, batch_idx, dataloader_idx)
        except TypeError:
            return parent_hook(batch, batch_idx)

    def _normalize_rvrt_train_patterns(self, patterns):
        if patterns is None:
            return (
                "conv_before_upsample",
                "upsample",
                "conv_up1",
                "conv_up2",
                "conv_hr",
                "conv_last",
                "tail",
                "reconstruction",
            )
        if isinstance(patterns, str):
            patterns = [p.strip() for p in patterns.split(",")]
        return tuple(p for p in patterns if p)

    def _normalize_items(self, items, default):
        if items is None:
            return tuple(default)
        if isinstance(items, str):
            items = [item.strip() for item in items.split(",")]
        return tuple(item for item in items if item)

    def _rvrt_requires_grad(self):
        return self.rvrt_train_mode != "frozen"

    def _configure_rvrt_training(self):
        for param in self.rvrt_frontend.parameters():
            param.requires_grad = False

        if not self._rvrt_requires_grad():
            self.rvrt_frontend.eval()
            print("RVRT frontend frozen")
            return

        trainable_names = []
        for name, param in self.rvrt_frontend.named_parameters():
            if any(pattern in name for pattern in self.rvrt_trainable_patterns):
                param.requires_grad = True
                trainable_names.append(name)

        if not trainable_names:
            raise RuntimeError(
                "RVRT partial finetuning was requested, but no parameters matched "
                f"patterns={self.rvrt_trainable_patterns}"
            )

        self.rvrt_frontend.train()
        print(
            f"RVRT frontend partial finetune enabled: {len(trainable_names)} params matched "
            f"patterns={self.rvrt_trainable_patterns}. First names: {trainable_names[:10]}"
        )

    def _set_training_stage(self, stage: str):
        if stage == self._active_stage:
            return
        if stage == "pixel":
            set_pisa_dual_lora_scales(self.model, pixel_scale=self.pixel_scale, semantic_scale=0.0)
            print("PiSA Dual-LoRA stage: pixel-only")
        elif stage == "joint":
            set_pisa_dual_lora_scales(self.model, pixel_scale=self.pixel_scale, semantic_scale=self.semantic_scale)
            print("PiSA Dual-LoRA stage: pixel + semantic")
        else:
            raise ValueError(f"Unsupported PiSA Dual-LoRA stage: {stage}")
        self._active_stage = stage

    def set_pisa_dual_lora_scales(self, pixel_scale=1.0, semantic_scale=1.0):
        self.pixel_scale = float(pixel_scale)
        self.semantic_scale = float(semantic_scale)
        set_pisa_dual_lora_scales(self.model, pixel_scale=self.pixel_scale, semantic_scale=self.semantic_scale)

    def _super_resolve_neighbors(self, batch, bs=None):
        prev_lr = super(LatentDiffusionVFI, self).get_input(batch, self.lr_prev_key).to(self.device)
        next_lr = super(LatentDiffusionVFI, self).get_input(batch, self.lr_next_key).to(self.device)
        if bs is not None:
            prev_lr = prev_lr[:bs]
            next_lr = next_lr[:bs]

        prev_01 = (prev_lr + 1.0) / 2.0
        next_01 = (next_lr + 1.0) / 2.0
        seq = torch.stack([prev_01, next_01], dim=1)
        grad_context = nullcontext() if self._rvrt_requires_grad() else torch.no_grad()
        with grad_context:
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
            raise RuntimeError("No PiSA Dual-LoRA parameters found")

        params = []
        if pixel_params:
            params.append({"params": pixel_params, "lr": self.learning_rate, "name": "pixel_lora"})
        if semantic_params:
            params.append({"params": semantic_params, "lr": self.learning_rate, "name": "semantic_lora"})

        rvrt_params = [p for p in self.rvrt_frontend.parameters() if p.requires_grad]
        if rvrt_params:
            params.append({"params": rvrt_params, "lr": self.rvrt_lr, "name": "rvrt"})

        if self.learn_logvar:
            params.append({"params": [self.logvar], "lr": self.learning_rate, "name": "logvar"})

        opt = torch.optim.AdamW(params, lr=self.learning_rate)
        print(
            f"Optimizer groups: pixel={len(pixel_params)} params, semantic={len(semantic_params)} params, "
            f"rvrt={len(rvrt_params)} params, lr={self.learning_rate:.2e}"
        )
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler = [{"scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule), "interval": "step", "frequency": 1}]
            return [opt], scheduler
        return opt
